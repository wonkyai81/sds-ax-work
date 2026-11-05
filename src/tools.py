"""도메인 타입과 결정적 도구: 합계·예산 판정·가장 근접한 안 선택·텍스트/HTML 렌더링 (LLM 호출 없음)."""
import asyncio
import json
import re
import shutil
import threading
from contextlib import AsyncExitStack
from datetime import date, timedelta
from html import escape
from typing import Any, Literal, NotRequired, TypedDict

MAX_READJUST = 3
VALUE_PLAN_RATIO = 0.8


class TripRequest(TypedDict):
    """여행 요청: 여행지, 기간(일), 인원(성인 기준), 사용자 예산(원, 없으면 None = 금액 제한 없음)."""

    destination: str
    days: int
    people: int
    budget_krw: int | None
    origin: NotRequired[str | None]  # 출발지. 없으면 기본값(서울)을 쓴다
    start_date: NotRequired[str | None]  # 사용자가 명시한 여행 시작일(YYYY-MM-DD). 없으면 오늘+30일을 쓴다


class ItineraryItem(TypedDict):
    """일정 항목 하나: 일차·시간대·활동·원화 비용·교통·비고."""

    day: int
    time_slot: Literal["아침", "오후", "저녁"]
    activity: str
    cost_krw: int
    transport: str
    note: str
    origin: NotRequired[str]  # 출발지
    arrival: NotRequired[str]  # 도착지
    duration: NotRequired[str]  # 이동 소요시간(예: "약 30분")
    depart_time: NotRequired[str]  # 출발(시작) 시각 HH:MM
    arrive_time: NotRequired[str]  # 도착(종료) 시각 HH:MM
    per_person: NotRequired[bool]  # 입장권·식사·항공권처럼 1인 단위 비용이면 True
    unit_cost_krw: NotRequired[int]  # per_person 일 때 1인 가격(원). 전체 비용 cost_krw 는 코드가 인원을 곱해 계산한다


PlanType = Literal["맛집형", "명소형", "가성비형"]
PLAN_TYPES: tuple[str, ...] = ("맛집형", "명소형", "가성비형")


class PlanState(TypedDict):
    """안 하나의 원본 상태: 안 종류와 시도별 항목 목록(최대 1 + 재조정 3회)."""

    plan_type: PlanType
    attempts: list[list[ItineraryItem]]


def target_budget(plan_type: str, budget_krw: int | None) -> int | None:
    """안 종류별 목표 예산(원)을 돌려준다. 가성비형은 사용자 예산의 80%. 예산이 없으면 None(제한 없음)."""
    if budget_krw is None:
        return None
    if plan_type == "가성비형":
        return int(budget_krw * VALUE_PLAN_RATIO)
    return budget_krw


def total_cost(items: list[ItineraryItem]) -> int:
    """항목별 비용(원)의 합계를 계산한다."""
    return sum(int(item["cost_krw"]) for item in items)


def apply_headcount(items: list[ItineraryItem], people: int | None) -> list[ItineraryItem]:
    """per_person 항목의 전체 비용을 '1인 가격 x 인원'으로 코드가 다시 계산한다(원본 목록은 바꾸지 않는다)."""
    result = []
    for item in items:
        new = dict(item)
        unit = int(new.get("unit_cost_krw") or 0)
        if new.get("per_person") and unit > 0 and people and people > 0:
            new["cost_krw"] = unit * people
        result.append(new)
    return result


def check_budget(plan_type: str, items: list[ItineraryItem], budget_krw: int | None) -> dict:
    """합계와 목표 예산을 비교해 {within, total, overage}를 돌려준다. 합계 = 목표면 충족."""
    total = total_cost(items)
    target = target_budget(plan_type, budget_krw)
    if target is None:  # 예산이 없으면 판정 대상이 없어 항상 충족으로 본다
        return {"within": True, "total": total, "overage": 0}
    overage = total - target
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
TRIP_FIELD_LABELS = {"destination": "여행지", "days": "기간", "people": "인원"}  # 필수 입력. 예산은 선택
OPTIONAL_TRIP_FIELDS = ("budget_krw", "origin", "start_date")
DEFAULT_LEAD_DAYS = 30  # 날짜를 명시하지 않으면 오늘 + 30일(D+30)을 여행 시작일로 조회한다
DEFAULT_ORIGIN = "서울"  # 출발지 입력이 없을 때 쓰는 기본 출발지


def trip_origin(trip: dict) -> tuple[str, bool]:
    """여행 요청의 출발지와 기본값 여부를 돌려준다. 입력이 없으면 ('서울', True)."""
    origin = str(trip.get("origin") or "").strip()
    return (origin, False) if origin else (DEFAULT_ORIGIN, True)


def origin_label(trip: dict) -> str:
    """화면 표시용 출발지 문자열. 기본값이면 '(기본값)'을 붙인다."""
    origin, is_default = trip_origin(trip)
    return f"{origin}(기본값)" if is_default else origin


def trip_dates(trip: dict, today: date | None = None) -> tuple[date, date, bool]:
    """MCP 조회와 화면에 쓸 (시작일, 종료일, 기본값 여부). 사용자가 명시한 미래 날짜가 있으면 그것을, 없으면 오늘+30일을 쓴다."""
    today = today or date.today()
    start = None
    try:
        start = date.fromisoformat(str(trip["start_date"])) if trip.get("start_date") else None
    except ValueError:
        start = None
    is_default = start is None or start < today
    if is_default:
        start = today + timedelta(days=DEFAULT_LEAD_DAYS)
    days = trip.get("days") if isinstance(trip.get("days"), int) and trip["days"] > 0 else 1
    return start, start + timedelta(days=days - 1), is_default


def period_label(trip: dict, today: date | None = None) -> str:
    """화면 표시용 조회 기준 날짜 문자열. 입력한 날짜인지 기본값(D+30)인지 함께 알려 준다."""
    start, end, is_default = trip_dates(trip, today)
    return f"{start.isoformat()} ~ {end.isoformat()} ({'입력이 없어 D+30 기본값' if is_default else '입력'})"


def normalize_time(value: Any) -> str:
    """'9:05', '09:05' 같은 시각을 'HH:MM'으로 맞춘다. 형식이 틀리면 빈 문자열(깨진 값을 보여주지 않는다)."""
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", str(value or ""))
    if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
        return ""
    return f"{int(m.group(1)):02d}:{m.group(2)}"
CONFLICT_NOTICE = "안내: 요청하신 조건(최고급·5성급과 최소 비용)이 서로 충돌해 예산을 우선해 안을 만들었습니다."

_GUARD_MESSAGES = {
    "secret_request": "시스템 설정, 프롬프트, API 키 같은 비밀 값은 알려드릴 수 없습니다. 여행지·기간·인원(예산은 생략 가능)을 알려 주시면 일정을 만들어 드립니다.",
    "personal_data_request": "다른 사용자의 계획이나 개인정보는 제공할 수 없습니다. 이 서비스는 로그인과 저장 기능이 없어 사용자별 정보를 보관하지 않습니다.",
    "unsupported_lookup": "v1은 실시간 가격이나 특정 업체의 정확한 요금을 조회하지 않고 AI 추정값만 씁니다. 여행지·기간·인원(예산은 생략 가능)을 알려 주시면 추정 비용으로 일정을 만들어 드립니다.",
}


def missing_fields(trip: dict) -> list[str]:
    """여행 요청에서 빠진 필수 항목(여행지·기간·인원)의 한글 이름 목록을 돌려준다. 예산은 선택이라 검사하지 않는다."""
    return [label for key, label in TRIP_FIELD_LABELS.items() if not trip.get(key)]


def merge_trip(known: dict | None, parsed: dict) -> dict:
    """보관해 둔 값(known)에 새로 파싱한 값을 덮어 합친다. 새 값이 비어 있으면 기존 값을 유지한다."""
    merged = dict(known or {})
    for key in (*TRIP_FIELD_LABELS, *OPTIONAL_TRIP_FIELDS):
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


SEARCH_TOOL_NAMES = ("maps_search_places", "maps_search_nearby", "maps_place_details", "maps_explore_area", "maps_geocode")
ROUTE_TOOL_NAMES = ("maps_directions", "maps_distance_matrix")
PRICE_TOOL_NAMES = ("searchTnas", "searchStays", "flightsFareCalendar", "searchInternationalFlights", "searchDomesticFlights")
MCP_CALL_TIMEOUT = 60
MAX_COMPACT_ITEMS = 8


def _widget_texts(node: Any, acc: list[str]) -> list[str]:
    """마이리얼트립 화면 위젯 JSON 에서 Text 값만 순서대로 뽑는다."""
    if isinstance(node, dict):
        if node.get("type") == "Text" and "value" in node:
            acc.append(str(node["value"]).strip())
        for value in node.values():
            _widget_texts(value, acc)
    elif isinstance(node, list):
        for value in node:
            _widget_texts(value, acc)
    return acc


def _flight_leg(origin: str, destination: str, leg: dict) -> str:
    """항공 구간 한 줄: '출발지→도착지 날짜 출발시각→날짜 도착시각'."""
    return (
        f"{origin}→{destination} {leg.get('departDate', '')} {leg.get('departTime', '')}"
        f"→{leg.get('arriveDate', '')} {leg.get('arriveTime', '')}"
    )


def compact_price_result(name: str, raw: str) -> str:
    """마이리얼트립 도구 결과(수만 자 JSON)를 모델이 읽을 짧은 목록으로 줄인다. 파싱에 실패하면 앞부분만 돌려준다."""
    try:
        data = json.loads(raw)
        if name in ("searchTnas", "searchStays"):
            items: list[list[str]] = []
            for text in _widget_texts(data, []):
                if text and (text.startswith("⭐") or re.search(r"원(?:~|/박)?$", text)) and items:
                    items[-1].append(text)
                elif text:
                    items.append([text])
            return "\n".join("- " + " | ".join(i) for i in items[:MAX_COMPACT_ITEMS])
        result = data["result"]
        if name == "flightsFareCalendar":
            rows = [
                f"- {i['departureDate']}~{i['returnDate']} {i['airline']} 왕복 {_won(int(i['totalPrice']))}"
                for i in result["items"][:5]
            ]
            return "요금 캘린더(실시간 아님, 예약 시 달라질 수 있음)\n" + "\n".join(rows)
        if name == "searchInternationalFlights":
            rows = []
            for i in result["items"][:3]:
                info = i["travelInfo"]
                kind = "직항" if info.get("isDirect") else f"경유 {info.get('stops', '?')}회"
                legs = i.get("legs") or []
                flight_no = ""
                if legs and legs[0].get("segments"):
                    flight_no = str(legs[0]["segments"][0].get("flightNumber", ""))
                legs_text = " / ".join(
                    f"{label} {_flight_leg(leg['origin'], leg['destination'], leg)}"
                    for label, leg in zip(("가는 편", "오는 편"), legs)
                )
                rows.append(
                    f"- {i['airline']['name']} {flight_no} {kind} {legs_text} / 1인 왕복 합계 {_won(int(i['price']['total']))}"
                )
            return "국제선(성인 1명 기준, 실시간 변동)\n" + "\n".join(rows)
        if name == "searchDomesticFlights":
            rows = []
            for i in result["items"][:3]:
                parts = []
                for label, key in (("가는 편", "outbound"), ("오는 편", "inbound")):
                    leg = i.get(key)
                    if leg:
                        parts.append(f"{label} {leg.get('flightNumber', '')} " + _flight_leg(leg["departCity"], leg["arriveCity"], leg))
                rows.append(f"- {i['airline']['name']} " + " / ".join(parts) + f" / 1인 왕복 합계 {_won(int(i['price']['total']))}")
            return "국내선(성인 1명 기준, 실시간 변동)\n" + "\n".join(rows)
    except (ValueError, KeyError, TypeError):
        pass
    return raw[:3000]


class _BackgroundMCP:
    """MCP 서버 세션을 백그라운드 이벤트 루프에 연결해 두고, 도구를 동기 LangChain 도구로 노출하는 공통 기반."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._stack: AsyncExitStack | None = None

    def start(self):
        """MCP 서버에 연결하고 도구 목록을 불러온다. 실패하면 예외를 그대로 낸다."""
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self._open(), self._loop).result(timeout=180)
        return self

    async def _open(self) -> None:
        """세션을 열고 도구를 동기 래퍼로 바꿔 등록한다."""
        from langchain_mcp_adapters.tools import load_mcp_tools
        from mcp import ClientSession

        self._stack = AsyncExitStack()
        read, write = await self._connect()
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        for tool in await load_mcp_tools(session):
            self._register(tool)

    async def _connect(self) -> tuple[Any, Any]:
        """전송 방식별 연결(읽기, 쓰기 스트림)을 돌려준다."""
        raise NotImplementedError

    def _register(self, tool: Any) -> None:
        """도구를 용도별 목록에 등록한다."""
        raise NotImplementedError

    def _compact(self, name: str, raw: str) -> str:
        """도구 결과 문자열을 가공한다(기본은 그대로)."""
        return raw

    def _to_sync(self, tool: Any):
        """비동기 MCP 도구를 동기 호출 가능한 StructuredTool 로 감싼다(결과 문자열만 돌려줌)."""
        from langchain_core.tools import StructuredTool

        def call(**kwargs: Any) -> str:
            """MCP 도구를 호출하고 결과 텍스트를 돌려준다. 실패하면 오류 문장을 돌려줘 모델이 다음 행동을 정하게 한다."""
            try:
                # callbacks=[]: 바깥 래퍼 도구가 이미 추적되므로 안쪽 MCP 도구는 추적 콜백을 상속하지 않게 한다(중복 기록 방지)
                future = asyncio.run_coroutine_threadsafe(tool.ainvoke(kwargs, config={"callbacks": []}), self._loop)
                result = future.result(timeout=MCP_CALL_TIMEOUT)
                if isinstance(result, list):  # MCP 콘텐츠 블록 목록이면 텍스트만 이어 붙인다
                    text = "\n".join(str(b.get("text", b)) if isinstance(b, dict) else str(b) for b in result)
                else:
                    text = str(result)
                return self._compact(tool.name, text)
            except Exception as exc:  # 조회 실패는 일정 생성을 막지 않는다
                return f"조회 실패: {type(exc).__name__}"

        return StructuredTool(name=tool.name, description=tool.description, args_schema=tool.args_schema, func=call)


class GoogleMapsMCP(_BackgroundMCP):
    """Google Maps MCP 서버(stdio). search_tools 는 국내·해외 장소 검색, route_tools 는 해외 경로·소요시간 조회용이다."""

    def __init__(self, api_key: str):
        super().__init__()
        self._api_key = api_key
        self.search_tools: list = []
        self.route_tools: list = []

    async def _connect(self) -> tuple[Any, Any]:
        """npx 로 Google Maps MCP 서버를 띄워 stdio 로 연결한다."""
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=shutil.which("npx") or "npx",
            args=["-y", "@cablate/mcp-google-map", "--stdio"],
            env={"GOOGLE_MAPS_API_KEY": self._api_key},
        )
        return await self._stack.enter_async_context(stdio_client(params))

    def _register(self, tool: Any) -> None:
        """이름으로 검색용/경로용에 나눠 등록한다."""
        if tool.name in SEARCH_TOOL_NAMES:
            self.search_tools.append(self._to_sync(tool))
        elif tool.name in ROUTE_TOOL_NAMES:
            self.route_tools.append(self._to_sync(tool))


class MyRealTripMCP(_BackgroundMCP):
    """마이리얼트립 MCP 서버(HTTP). price_tools 는 항공권(국내·해외)과 투어·입장권 가격 조회용이며 결과는 짧게 줄여 돌려준다."""

    def __init__(self, url: str):
        super().__init__()
        self._url = url
        self.price_tools: list = []

    async def _connect(self) -> tuple[Any, Any]:
        """HTTP(streamable) 로 연결한다. 인증 키는 필요 없다."""
        from mcp.client.streamable_http import streamablehttp_client

        streams = await self._stack.enter_async_context(streamablehttp_client(self._url))
        return streams[0], streams[1]

    def _register(self, tool: Any) -> None:
        """가격 조회 도구만 등록한다(숙소·예약 관련 도구는 쓰지 않는다)."""
        if tool.name in PRICE_TOOL_NAMES:
            self.price_tools.append(self._to_sync(tool))

    def _compact(self, name: str, raw: str) -> str:
        """큰 JSON 결과를 목록 몇 줄로 줄인다."""
        return compact_price_result(name, raw)


def _won(value: int) -> str:
    """금액을 '1,234원' 형태로 만든다."""
    return f"{value:,}원"


def _summary_line(plan_type: str, items: list[ItineraryItem], budget_krw: int) -> str:
    """안 하나의 한 줄 요약(합계·목표·충족 여부)."""
    result = check_budget(plan_type, items, budget_krw)
    target = target_budget(plan_type, budget_krw)
    if target is None:
        return f"- {plan_type}: 합계 {_won(result['total'])} — 예산 제한 없음"
    status = "예산 충족" if result["within"] else f"예산 미충족(초과 {_won(result['overage'])})"
    return f"- {plan_type}: 합계 {_won(result['total'])} / 목표 {_won(target)} — {status}"


def render_plain_answer(
    trip: TripRequest,
    plans: dict[str, PlanState],
    selected: str | None = None,
    items: list[ItineraryItem] | None = None,
) -> str:
    """plainAnswer 텍스트를 만든다. 선택 전이면 3안 요약, 선택 후면 선택 안 요약."""
    budget = trip.get("budget_krw")
    if selected is None:
        who = f", {trip['people']}명" if trip.get("people") else ""
        basis = f"예산 {_won(budget)}" if budget is not None else "예산 제한 없음"
        lines = [f"{origin_label(trip)} → {trip['destination']} {trip['days']}일{who}, {basis} 기준 안입니다. 하나를 골라 주세요.",
                 f"조회 기준 날짜: {period_label(trip)}"]
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


REPORT_STYLE = (
    "<style>"
    ".report{font-family:system-ui,-apple-system,'Segoe UI','Malgun Gothic',sans-serif;color:#1f2937}"
    ".report .tabs{display:flex;gap:4px;border-bottom:1px solid #e5e7eb;margin:12px 0}"
    ".report .tabs label{padding:8px 14px;cursor:pointer;color:#6b7280;border-bottom:2px solid transparent;font-weight:600}"
    ".report input.tab{display:none}.report .panel{display:none}"
    "#rt1:checked~.tabs label[for=rt1],#rt2:checked~.tabs label[for=rt2],#rt3:checked~.tabs label[for=rt3]"
    "{color:#4f46e5;border-bottom-color:#4f46e5}"
    "#rt1:checked~.p1,#rt2:checked~.p2,#rt3:checked~.p3{display:block}"
    ".report table{width:100%;border-collapse:collapse;font-size:14px}"
    ".report th,.report td{border-bottom:1px solid #f1f5f9;padding:8px;text-align:left;vertical-align:top}"
    ".report th{background:#f8fafc}.report .warning{color:#b91c1c;font-weight:600}"
    ".report .notice{color:#6b7280;font-size:13px}.report .card{border:1px solid #e5e7eb;border-radius:10px;padding:12px;margin:8px 0}"
    ".report .badge{font-size:12px;background:#eef2ff;color:#4338ca;border-radius:999px;padding:2px 8px;margin-right:6px}"
    ".report h3{margin:16px 0 6px}"
    "</style>"
)
GUIDE_MISSING = "<p class='notice'>이 안내는 생성하지 못했습니다. 잠시 후 다시 시도해 주세요.</p>"


def _guide_cards(rows: list[dict], head: str, body: str, tag: str | None = None) -> str:
    """카드 목록 HTML. LLM 문자열은 모두 이스케이프한다."""
    out = []
    for r in rows:
        badge = f"<span class='badge'>{escape(str(r.get(tag, '')))}</span>" if tag and r.get(tag) else ""
        out.append(f"<div class='card'><strong>{badge}{escape(str(r.get(head, '')))}</strong><p>{escape(str(r.get(body, '')))}</p></div>")
    return "".join(out)


def _places_html(guide: dict | None) -> str:
    """탭 2: 맛집·명소 가이드북."""
    places = (guide or {}).get("places") or []
    if not places:
        return GUIDE_MISSING
    parts = []
    for kind in ("맛집", "명소"):
        group = [p for p in places if p.get("kind") == kind]
        if group:
            cards = "".join(
                "<div class='card'><strong>"
                f"<span class='badge'>{escape(str(p.get('area', '')))}</span>{escape(str(p.get('name', '')))}</strong>"
                f"<p>{escape(str(p.get('description', '')))}</p>"
                f"<p class='notice'>팁: {escape(str(p.get('tip', '')))}</p></div>"
                for p in group
            )
            parts.append(f"<h3>{kind}</h3>{cards}")
    return "".join(parts)


def _prep_html(guide: dict | None) -> str:
    """탭 3: 준비물, 유의 사항, 목적지 사전 유의 정보."""
    if not guide or not (guide.get("packing") or guide.get("cautions") or guide.get("destination_notes")):
        return GUIDE_MISSING
    packing = "".join(f"<li>{escape(str(x))}</li>" for x in guide.get("packing") or [])
    return (
        f"<h3>준비물</h3><ul>{packing}</ul>"
        f"<h3>목적지 사전 유의 정보</h3>{_guide_cards(guide.get('destination_notes') or [], 'title', 'detail')}"
        f"<h3>여행 중 유의 사항</h3>{_guide_cards(guide.get('cautions') or [], 'title', 'detail')}"
    )


def _cost_cell(item: ItineraryItem, people: int | None) -> str:
    """비용 칸 문자열. 1인 단위 항목은 '1인 가격 x 인원 = 전체 가격', 공용 항목은 전체 가격만 보여준다."""
    cost = int(item["cost_krw"])
    unit = int(item.get("unit_cost_krw") or 0)
    if item.get("per_person") and unit > 0:
        count = people or max(round(cost / unit), 1)
        return f"{_won(unit)} × {count}명 = {_won(cost)}"
    return _won(cost)


def render_report_html(
    plan_type: str,
    items: list[ItineraryItem],
    budget_krw: int | None,
    guide: dict | None = None,
    price_note: str | None = None,
    people: int | None = None,
    origin_text: str | None = None,
    period_text: str | None = None,
) -> str:
    """선택 안의 3탭 HTML 리포트(일정표 / 맛집·명소 가이드북 / 준비물·유의 사항)를 만든다. 탭은 CSS만으로 전환한다."""
    result = check_budget(plan_type, items, budget_krw)
    target = target_budget(plan_type, budget_krw)
    rows = "".join(
        "<tr>"
        f"<td>{int(i['day'])}일차</td><td>{normalize_time(i.get('depart_time'))}</td>"
        f"<td>{normalize_time(i.get('arrive_time'))}</td><td>{escape(str(i['time_slot']))}</td>"
        f"<td>{escape(str(i['activity']))}</td>"
        f"<td>{escape(str(i.get('origin', '')))}</td><td>{escape(str(i.get('arrival', '')))}</td>"
        f"<td>{escape(_cost_cell(i, people))}</td>"
        f"<td>{escape(str(i['transport']))}</td><td>{escape(str(i.get('duration', '')))}</td><td>{escape(str(i['note']))}</td>"
        "</tr>"
        for i in sorted(items, key=lambda x: (int(x["day"]), ["아침", "오후", "저녁"].index(x["time_slot"])
                                              if x["time_slot"] in ("아침", "오후", "저녁") else 3,
                                              normalize_time(x.get("depart_time")) or "99:99"))
    )
    warning = ""
    if not result["within"]:
        warning = (
            f"<p class='warning'>경고: 예산 미충족 — 목표 {_won(target)}를 {_won(result['overage'])} 초과했습니다. "
            "재조정 후에도 맞추지 못한 안 중 가장 근접한 안입니다.</p>"
        )
    if budget_krw is None:
        budget_lines = "<p>사용자 예산: 제한 없음</p>"
    else:
        budget_lines = (
            f"<p>사용자 예산: {_won(budget_krw)} / 목표 예산: {_won(target)}</p>"
            f"<p>예산 대비 잔액: {budget_krw - result['total']:,}원</p>"
        )
    people_line = (f"<p>출발지: {escape(origin_text)}</p>" if origin_text else "") + (
        f"<p>조회 기준 날짜: {escape(period_text)}</p>" if period_text else ""
    ) + (
        f"<p>여행 인원: {people}명</p>" if people else ""
    )
    per_person_note = (
        "<p class='notice'>1인 단위 항목(입장권·식사·항공권 등)은 '1인 가격 × 인원 = 전체 가격'으로 표시합니다.</p>"
        if any(i.get("per_person") for i in items) else ""
    )
    schedule = (
        f"{people_line}"
        "<table><thead><tr><th>일차</th><th>출발시간</th><th>도착시간</th><th>시간대</th><th>활동</th><th>출발지</th><th>도착지</th><th>비용(원)</th><th>교통수단</th><th>소요시간</th><th>비고</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        f"{per_person_note}"
        f"<p>비용 합계: {_won(result['total'])}</p>"
        f"{budget_lines}{warning}"
    )
    if price_note:
        schedule += (
            "<div class='card'><strong>참고 가격 (마이리얼트립 조회, 실시간 아님)</strong>"
            f"<p style='white-space:pre-wrap'>{escape(price_note)}</p>"
            "<p class='notice'>참고 가격은 1인 기준 시작가이며, 일정 비용에는 선택한 항목의 '1인 가격 × 인원'으로 반영됩니다.</p></div>"
        )
    return (
        f"{REPORT_STYLE}<section class='report'><h2>{escape(plan_type)} 일정</h2>"
        "<p class='notice'>일정과 가격은 AI 추정값이며 실제와 다를 수 있습니다.</p>"
        "<input class='tab' type='radio' name='rt' id='rt1' checked><input class='tab' type='radio' name='rt' id='rt2'>"
        "<input class='tab' type='radio' name='rt' id='rt3'>"
        "<div class='tabs'><label for='rt1'>일정표</label><label for='rt2'>맛집 및 명소 가이드북</label>"
        "<label for='rt3'>준비물·유의 사항</label></div>"
        f"<div class='panel p1'>{schedule}</div><div class='panel p2'>{_places_html(guide)}</div>"
        f"<div class='panel p3'>{_prep_html(guide)}</div></section>"
    )
