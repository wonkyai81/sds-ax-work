"""도메인 타입과 결정적 도구: 합계·예산 판정·가장 근접한 안 선택·텍스트/HTML 렌더링 (LLM 호출 없음)."""
import re
from html import escape
from typing import Literal, TypedDict

MAX_READJUST = 3
VALUE_PLAN_RATIO = 0.8


class TripRequest(TypedDict):
    """여행 요청: 여행지, 기간(일), 사용자 예산(원)."""

    destination: str
    days: int
    budget_krw: int


class ItineraryItem(TypedDict):
    """일정 항목 하나: 일차·시간대·활동·원화 비용·교통·비고."""

    day: int
    time_slot: Literal["아침", "오후", "저녁"]
    activity: str
    cost_krw: int
    transport: str
    note: str


PlanType = Literal["맛집형", "명소형", "가성비형"]
PLAN_TYPES: tuple[str, ...] = ("맛집형", "명소형", "가성비형")


class PlanState(TypedDict):
    """안 하나의 원본 상태: 안 종류와 시도별 항목 목록(최대 1 + 재조정 3회)."""

    plan_type: PlanType
    attempts: list[list[ItineraryItem]]


def target_budget(plan_type: str, budget_krw: int) -> int:
    """안 종류별 목표 예산(원)을 돌려준다. 가성비형은 사용자 예산의 80%."""
    if plan_type == "가성비형":
        return int(budget_krw * VALUE_PLAN_RATIO)
    return budget_krw


def total_cost(items: list[ItineraryItem]) -> int:
    """항목별 비용(원)의 합계를 계산한다."""
    return sum(int(item["cost_krw"]) for item in items)


def check_budget(plan_type: str, items: list[ItineraryItem], budget_krw: int) -> dict:
    """합계와 목표 예산을 비교해 {within, total, overage}를 돌려준다. 합계 = 목표면 충족."""
    total = total_cost(items)
    overage = total - target_budget(plan_type, budget_krw)
    return {"within": overage <= 0, "total": total, "overage": max(overage, 0)}


def closest_attempt(attempts: list[list[ItineraryItem]], plan_type: str, budget_krw: int) -> list[ItineraryItem]:
    """시도 중 초과 금액이 가장 작은 시도를 고른다(동률이면 나중 시도). 빈 목록이면 빈 목록."""
    if not attempts:
        return []
    best = min(
        range(len(attempts)),
        key=lambda i: (check_budget(plan_type, attempts[i], budget_krw)["overage"], -i),
    )
    return attempts[best]


def readjust_count(attempts: list) -> int:
    """재조정 횟수는 시도 수 - 1 로 계산한다."""
    return max(len(attempts) - 1, 0)


def needs_readjust(plan_type: str, attempts: list[list[ItineraryItem]], budget_krw: int) -> bool:
    """마지막 시도가 초과이고 재조정 횟수가 3회 미만이면 True."""
    last = check_budget(plan_type, attempts[-1], budget_krw)
    return (not last["within"]) and readjust_count(attempts) < MAX_READJUST


def final_items(plan_type: str, attempts: list[list[ItineraryItem]], budget_krw: int) -> list[ItineraryItem]:
    """보여줄 항목을 고른다: 예산을 충족한 마지막 시도, 없으면 가장 근접한 시도."""
    for attempt in reversed(attempts):
        if check_budget(plan_type, attempt, budget_krw)["within"]:
            return attempt
    return closest_attempt(attempts, plan_type, budget_krw)


_SECRET_RE = re.compile(
    r"시스템\s*프롬프트|api\s*키|api\s*key|aws|bedrock|환경\s*변수|비밀\s*번호|자격\s*증명|시크릿|secret|토큰|\.env",
    re.IGNORECASE,
)
_PERSONAL_RE = re.compile(r"다른\s*(?:사용자|사람|유저)|그\s*사람|저장해\s*둔|연락처|전화\s*번호|이메일\s*주소|개인\s*정보")
_LOOKUP_RE = re.compile(r"실시간|최저가|정확한\s*[\w\s]{0,10}?(?:요금|가격|금액|비용)|예약")
_LUXURY_RE = re.compile(r"최고급|5\s*성급|오성급|럭셔리|초호화")
_CHEAP_RE = re.compile(r"최소|최저|저렴|가장\s*싸|아주\s*싸")
TRIP_FIELD_LABELS = {"destination": "여행지", "days": "기간", "budget_krw": "예산"}
CONFLICT_NOTICE = "안내: 요청하신 조건(최고급·5성급과 최소 비용)이 서로 충돌해 예산을 우선해 안을 만들었습니다."

_GUARD_MESSAGES = {
    "secret_request": "시스템 설정, 프롬프트, API 키 같은 비밀 값은 알려드릴 수 없습니다. 여행지·기간·예산을 알려 주시면 일정을 만들어 드립니다.",
    "personal_data_request": "다른 사용자의 계획이나 개인정보는 제공할 수 없습니다. 이 서비스는 로그인과 저장 기능이 없어 사용자별 정보를 보관하지 않습니다.",
    "unsupported_lookup": "v1은 실시간 가격이나 특정 업체의 정확한 요금을 조회하지 않고 AI 추정값만 씁니다. 여행지·기간·예산을 알려 주시면 추정 비용으로 일정을 만들어 드립니다.",
}


def missing_fields(trip: dict) -> list[str]:
    """여행 요청에서 빠진 항목(여행지·기간·예산)의 한글 이름 목록을 돌려준다."""
    return [label for key, label in TRIP_FIELD_LABELS.items() if not trip.get(key)]


def merge_trip(known: dict | None, parsed: dict) -> dict:
    """보관해 둔 값(known)에 새로 파싱한 값을 덮어 합친다. 새 값이 비어 있으면 기존 값을 유지한다."""
    merged = dict(known or {})
    for key in TRIP_FIELD_LABELS:
        if parsed.get(key):
            merged[key] = parsed[key]
        else:
            merged.setdefault(key, parsed.get(key))
    return merged


def has_conflict(question: str) -> bool:
    """최고급 조건과 최소 비용 조건이 함께 나오면 충돌로 본다."""
    return bool(_LUXURY_RE.search(question) and _CHEAP_RE.search(question))


def classify_request(question: str, known_trip: dict | None = None, new_trip: bool = False) -> dict | None:
    """요청을 규칙으로 검사한다(LLM 미사용). 거절·안내가 필요하면 {code, message[, missing]}, 통과면 None.

    1~2번(비밀·개인정보)은 모든 요청에, 3~4번(실시간 조회·필수 정보 부족)은 new_trip=True 일 때만 적용한다.
    known_trip 은 파싱·병합이 끝난 여행 정보다.
    """
    for code, pattern in (("secret_request", _SECRET_RE), ("personal_data_request", _PERSONAL_RE)):
        if pattern.search(question):
            return {"code": code, "message": _GUARD_MESSAGES[code]}
    if not new_trip:
        return None
    if _LOOKUP_RE.search(question):
        return {"code": "unsupported_lookup", "message": _GUARD_MESSAGES["unsupported_lookup"]}
    missing = missing_fields(known_trip or {})
    if missing:
        names = "·".join(missing)
        return {"code": "missing_fields", "missing": missing, "message": f"{names}을(를) 알려 주세요. 알려 주시면 이어서 일정을 만들겠습니다."}
    return None


def _won(value: int) -> str:
    """금액을 '1,234원' 형태로 만든다."""
    return f"{value:,}원"


def _summary_line(plan_type: str, items: list[ItineraryItem], budget_krw: int) -> str:
    """안 하나의 한 줄 요약(합계·목표·충족 여부)."""
    result = check_budget(plan_type, items, budget_krw)
    target = target_budget(plan_type, budget_krw)
    status = "예산 충족" if result["within"] else f"예산 미충족(초과 {_won(result['overage'])})"
    return f"- {plan_type}: 합계 {_won(result['total'])} / 목표 {_won(target)} — {status}"


def render_plain_answer(
    trip: TripRequest,
    plans: dict[str, PlanState],
    selected: str | None = None,
    items: list[ItineraryItem] | None = None,
) -> str:
    """plainAnswer 텍스트를 만든다. 선택 전이면 3안 요약, 선택 후면 선택 안 요약."""
    budget = trip["budget_krw"]
    if selected is None:
        lines = [f"{trip['destination']} {trip['days']}일, 예산 {_won(budget)} 기준 안입니다. 하나를 골라 주세요."]
        for plan_type in PLAN_TYPES:
            if plan_type in plans:
                lines.append(_summary_line(plan_type, final_items(plan_type, plans[plan_type]["attempts"], budget), budget))
        return "\n".join(lines)
    chosen = items if items is not None else final_items(selected, plans[selected]["attempts"], budget)
    lines = [f"선택한 안: {selected}", _summary_line(selected, chosen, budget)]
    if not check_budget(selected, chosen, budget)["within"]:
        lines.append("경고: 예산을 충족하지 못한 안 중 가장 근접한 안입니다.")
    lines.append("수정하고 싶은 부분을 말씀해 주세요. 종료하려면 '종료'를 보내세요.")
    return "\n".join(lines)


def render_report_html(plan_type: str, items: list[ItineraryItem], budget_krw: int) -> str:
    """선택 안의 HTML 리포트를 만든다. LLM 문자열은 모두 이스케이프한다."""
    result = check_budget(plan_type, items, budget_krw)
    target = target_budget(plan_type, budget_krw)
    balance = budget_krw - result["total"]
    rows = "".join(
        "<tr>"
        f"<td>{int(i['day'])}일차</td><td>{escape(str(i['time_slot']))}</td>"
        f"<td>{escape(str(i['activity']))}</td><td>{_won(int(i['cost_krw']))}</td>"
        f"<td>{escape(str(i['transport']))}</td><td>{escape(str(i['note']))}</td>"
        "</tr>"
        for i in sorted(items, key=lambda x: (int(x["day"]), ["아침", "오후", "저녁"].index(x["time_slot"])
                                              if x["time_slot"] in ("아침", "오후", "저녁") else 3))
    )
    warning = ""
    if not result["within"]:
        warning = (
            f"<p class='warning'>경고: 예산 미충족 — 목표 {_won(target)}를 {_won(result['overage'])} 초과했습니다. "
            "재조정 후에도 맞추지 못한 안 중 가장 근접한 안입니다.</p>"
        )
    return (
        f"<section class='report'><h2>{escape(plan_type)} 일정</h2>"
        "<p class='notice'>일정과 가격은 AI 추정값이며 실제와 다를 수 있습니다.</p>"
        "<table><thead><tr><th>일차</th><th>시간대</th><th>활동</th><th>비용(원)</th><th>교통수단</th><th>비고</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        f"<p>비용 합계: {_won(result['total'])}</p>"
        f"<p>사용자 예산: {_won(budget_krw)} / 목표 예산: {_won(target)}</p>"
        f"<p>예산 대비 잔액: {balance:,}원</p>{warning}</section>"
    )
