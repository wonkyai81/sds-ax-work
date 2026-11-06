"""도메인 타입과 결정적 도구: 합계·예산 판정·가장 근접한 안 선택·텍스트/HTML 렌더링 (LLM 호출 없음)."""
import asyncio
import json
import logging
import math
import re
import shutil
import threading
from contextlib import AsyncExitStack
from datetime import date, datetime, timedelta, timezone
from html import escape
from urllib.parse import quote
from typing import Any, Literal, NotRequired, TypedDict

MAX_READJUST = 3
logger = logging.getLogger(__name__)

NOTE_MAX_CHARS = 40  # 비고(note) 최대 글자 수. 프롬프트로 짧게 쓰게 하고(출력 토큰 절감), 넘으면 코드가 자른다
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
    unit_cost_krw: NotRequired[int]  # per_person 일 때 1인 가격(원), 숙박 항목이면 객실 1박 요금(원). 전체 비용 cost_krw 는 코드가 곱해 계산한다
    local_unit_price: NotRequired[float]  # 현지 통화로 추정한 단가(숫자만). 환율 곱셈은 코드가 한다
    local_currency: NotRequired[str]  # local_unit_price 의 통화 코드(예: JPY)
    rooms: NotRequired[int]  # 숙박 항목: 코드가 계산한 객실 수
    nights: NotRequired[int]  # 숙박 항목: 코드가 계산한 박 수


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


def clip_note(note: Any) -> str:
    """비고를 NOTE_MAX_CHARS 글자 이내로 맞춘다. 넘으면 마지막을 '…'로 줄인다."""
    text = str(note or "").strip()
    return text if len(text) <= NOTE_MAX_CHARS else text[: NOTE_MAX_CHARS - 1].rstrip() + "…"


def apply_headcount(items: list[ItineraryItem], people: int | None) -> list[ItineraryItem]:
    """항목 후처리: per_person 항목의 전체 비용을 '1인 가격 x 인원'으로 코드가 다시 계산하고, 비고를 상한 길이로 맞춘다(원본 목록은 바꾸지 않는다)."""
    result = []
    for item in items:
        new = dict(item)
        if "note" in new:
            new["note"] = clip_note(new["note"])
        unit = int(new.get("unit_cost_krw") or 0)
        if new.get("per_person") and unit > 0 and people and people > 0:
            new["cost_krw"] = unit * people
        result.append(new)
    return result


MEAL_PRICE_BY_LEVEL = {"INEXPENSIVE": 10_000, "MODERATE": 20_000, "EXPENSIVE": 50_000, "VERY_EXPENSIVE": 100_000}  # Google 가격대 → 식사 1인 가격(원). 국가와 무관한 고정 기준표
MEAL_LEVEL_LABEL = {"INEXPENSIVE": "저렴", "MODERATE": "보통", "EXPENSIVE": "비쌈", "VERY_EXPENSIVE": "매우 비쌈"}
MIN_MEAL_PER_PERSON_KRW = 3_000  # 식사 1인 최소 단가. LLM 이 예산을 맞추려고 식사를 0원으로 쓰는 것을 코드가 막는다
MIN_LODGING_ROOM_NIGHT_KRW = 30_000  # 객실 1박 최소 요금. 숙박비를 0원으로 쓰는 것을 코드가 막는다
FLOOR_NOTE = "최소 단가 적용"


def lodging_index(items: list[ItineraryItem]) -> int | None:
    """총 숙박비를 담은 항목의 위치. 활동에 '체크인'이 들어간 첫 항목, 없으면 비용이 있는 첫 '숙박' 분류 항목. 없으면 None."""
    for k, item in enumerate(items):
        if "체크인" in str(item.get("activity", "")):
            return k
    for k, item in enumerate(items):
        if str(item.get("category") or "").strip() == "숙박" and int(item.get("cost_krw") or 0) > 0:
            return k
    return None


def finalize_items(
    items: list[ItineraryItem], people: int | None, days: int | None, fx: dict | None = None, place_stats: dict | None = None,
    price_quotes: dict | None = None, plan_type: str | None = None,
) -> list[ItineraryItem]:
    """LLM 이 준 항목의 금액을 코드가 확정한다(원본은 바꾸지 않는다). 모든 곱셈·환산·최소 단가 비교가 여기서 일어난다.

    0) 마이리얼트립에서 조회한 항공·숙소 가격(price_quotes)이 있으면 해당 항목의 금액·시각을 조회값으로 덮어쓴다(apply_price_quotes).
    1) 현지 통화 단가(local_unit_price)가 있으면 환율을 곱해 원화 단가로 바꾼다(통화 코드가 환율의 통화와 같을 때만).
    2) 숙박 항목은 객실 1박 요금 x 객실 수(2명당 1실, 올림) x 박 수로 총 숙박비를 계산하고, 최소 단가보다 낮으면 최소 단가를 쓴다.
    3) 식당 항목은 Google 가격대(place_stats 의 price_level)를 아는 장소면 LLM 금액 대신 기준표(MEAL_PRICE_BY_LEVEL)의 1인 가격을 쓴다.
       가격대를 모르면 LLM 1인 단가를 쓰되 최소 단가보다 낮으면 최소 단가를 쓴다.
    4) 1인 단위 항목은 1인 가격 x 인원으로 전체 비용을 계산한다(apply_headcount).
    """
    items = apply_price_quotes(items, price_quotes, plan_type)
    rate = float(fx["krw_per_unit"]) if fx and fx.get("krw_per_unit") else None
    currency = str(fx.get("currency", "")).upper() if fx else ""
    people_n = people if people and people > 0 else None
    rooms = (people_n + 1) // 2 if people_n else None
    nights = days - 1 if days and days > 1 else 0
    lodging = lodging_index(items) if rooms and nights else None
    result = []
    for k, item in enumerate(items):
        new = dict(item)
        local = float(new.get("local_unit_price") or 0)
        if local > 0 and rate and str(new.get("local_currency") or "").upper() == currency:
            krw = int(round(local * rate))
            if k == lodging or new.get("per_person"):
                new["unit_cost_krw"] = krw
            else:
                new["cost_krw"] = krw
        floored = False
        if k == lodging:
            unit = int(new.get("unit_cost_krw") or 0)
            if unit <= 0:  # 총액으로 쓴 경우: 객실 1박 요금으로 되돌려 최소 단가와 비교한다
                unit = int(new.get("cost_krw") or 0) // (rooms * nights)
            if unit < MIN_LODGING_ROOM_NIGHT_KRW:
                unit, floored = MIN_LODGING_ROOM_NIGHT_KRW, True
            new.update(unit_cost_krw=unit, cost_krw=unit * rooms * nights, rooms=rooms, nights=nights, per_person=False)
        elif people_n and is_restaurant(new):
            stat = place_stats_for(str(new.get("activity", "")), place_stats)
            level = stat.get("price_level") if stat else None
            if level in MEAL_PRICE_BY_LEVEL:  # 조회한 가격대가 있으면 LLM 추정 대신 기준표 금액으로 확정한다
                new.update(per_person=True, unit_cost_krw=MEAL_PRICE_BY_LEVEL[level])
                new["note"] = f"Google 가격대 {MEAL_LEVEL_LABEL[level]} 기준 {new.get('note') or ''}".strip()
            else:
                unit = int(new.get("unit_cost_krw") or 0) if new.get("per_person") else 0
                if unit <= 0:
                    unit = int(new.get("cost_krw") or 0) // people_n
                if unit < MIN_MEAL_PER_PERSON_KRW:
                    new.update(per_person=True, unit_cost_krw=MIN_MEAL_PER_PERSON_KRW)
                    floored = True
        if floored:
            new["note"] = f"{FLOOR_NOTE} {new.get('note') or ''}".strip()
        result.append(new)
    return apply_headcount(result, people)


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

                def leg_text(label: str, leg: dict) -> str:
                    """구간별 편명을 따로 적는다(가는 편과 오는 편의 편명이 다르다)."""
                    segments = leg.get("segments") or [{}]
                    return f"{label} {segments[0].get('flightNumber', '')} {_flight_leg(leg['origin'], leg['destination'], leg)}"

                legs_text = " / ".join(leg_text(label, leg) for label, leg in zip(("가는 편", "오는 편"), legs))
                rows.append(
                    f"- {i['airline']['name']} {kind} {legs_text} / 1인 왕복 합계 {_won(int(i['price']['total']))}"
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


def _passenger_count(args: dict | None) -> int:
    """항공 조회 인자의 승객 수(성인+아동). 없으면 1. 조회 합계가 전체 승객 총액이라 1인 가격으로 나눌 때 쓴다."""
    passengers = (args or {}).get("passengers") or {}
    try:
        return max(1, int(passengers.get("adults", 1)) + int(passengers.get("children", 0)))
    except (TypeError, ValueError):
        return 1


def parse_price_quotes(name: str, raw: str, args: dict | None = None) -> dict:
    """마이리얼트립 항공·숙소 조회 원본에서 코드가 쓸 구조화 값을 뽑는다. 읽지 못하면 빈 사전.

    항공: {"flights": [{airline, direct, price_krw(1인 왕복), out/back: {date, depart, arrive, no}}]}
    숙소: {"stays": [{name, price_krw(객실 1박)}]}
    """
    try:
        data = json.loads(raw)
        if name in ("searchInternationalFlights", "searchDomesticFlights"):
            people = _passenger_count(args)
            flights = []
            for item in data["result"]["items"]:
                if name == "searchInternationalFlights":
                    legs = []
                    for leg in (item.get("legs") or [])[:2]:
                        segment = (leg.get("segments") or [{}])[0]
                        number = f"{segment.get('airlineCode', '')}{segment.get('flightNumber', '')}"
                        legs.append({"date": leg.get("departDate"), "depart": leg.get("departTime"), "arrive": leg.get("arriveTime"), "no": number})
                    direct = bool(item.get("travelInfo", {}).get("isDirect"))
                else:
                    legs = [
                        {"date": leg.get("departDate"), "depart": leg.get("departTime"), "arrive": leg.get("arriveTime"), "no": leg.get("flightNumber", "")}
                        for leg in (item.get("outbound"), item.get("inbound")) if leg
                    ]
                    direct = None
                total = int(item["price"]["total"])
                flights.append({
                    "airline": str(item["airline"]["name"]), "direct": direct, "price_krw": -(-total // people),
                    "out": legs[0] if legs else {}, "back": legs[1] if len(legs) > 1 else {},
                })
            return {"flights": flights} if flights else {}
        if name == "searchStays":
            groups: list[list[str]] = []
            for text in _widget_texts(data, []):
                if text and (text.startswith("⭐") or re.search(r"원(?:~|/박)?$", text)) and groups:
                    groups[-1].append(text)
                elif text:
                    groups.append([text])
            stays = []
            for group in groups:
                price = next((int(m.group(1).replace(",", "")) for t in group if (m := re.fullmatch(r"([\d,]+)원/박", t))), None)
                if price:
                    stays.append({"name": group[0], "price_krw": price})
            return {"stays": stays} if stays else {}
    except (ValueError, KeyError, TypeError, AttributeError):
        pass
    return {}


def record_price_quotes(sink: dict, name: str, raw: str, args: dict | None = None) -> None:
    """조회 결과를 sink 에 합친다. 항공은 마지막 조회로 바꾸고, 숙소는 이름별로 모은다."""
    found = parse_price_quotes(name, raw, args)
    if found.get("flights"):
        sink["flights"] = found["flights"]
    if found.get("stays"):
        known = {s["name"]: s for s in sink.get("stays", [])}
        known.update({s["name"]: s for s in found["stays"]})
        sink["stays"] = list(known.values())


def _pick_flight_quote(flights: list[dict], plan_type: str | None, hint: str) -> dict:
    """항공 후보 중 하나를 고른다. 항목에 적힌 항공사가 후보에 있으면 그 항공사, 가성비형은 최저가, 그 밖은 직항 중 최저가(직항이 없으면 최저가)."""
    pool = [f for f in flights if f["airline"] and f["airline"] in hint] or flights
    if plan_type != "가성비형":
        pool = [f for f in pool if f.get("direct")] or pool
    return min(pool, key=lambda f: f["price_krw"])


def _short_date(value: Any) -> str:
    """YYYY-MM-DD 를 M/D 로. 형식이 다르면 빈 문자열."""
    m = re.fullmatch(r"\d{4}-(\d{2})-(\d{2})", str(value or ""))
    return f"{int(m.group(1))}/{int(m.group(2))}" if m else ""


def _flight_note(prefix: str, quote: dict, leg: dict) -> str:
    """항공 항목 비고: 항공사·편명·날짜."""
    return " ".join(x for x in (prefix, quote["airline"], str(leg.get("no") or ""), _short_date(leg.get("date"))) if x)


def apply_price_quotes(items: list[ItineraryItem], quotes: dict | None, plan_type: str | None = None) -> list[ItineraryItem]:
    """LLM 이 쓴 항공·숙소 항목의 금액을 마이리얼트립에서 조회한 값으로 코드가 덮어쓴다(원본은 바꾸지 않는다).

    - 가는 항공('항공', '귀국' 없음)은 1인 왕복 요금과 가는 편 시각·편명·날짜, 귀국 항공은 오는 편 시각·편명·날짜로 맞춘다.
    - 숙소는 활동명의 숙소 이름이 조회한 숙소 이름과 맞을 때 그 1박 객실 요금으로 바꾼다. 맞지 않으면(사용자가 직접 고른 숙소 등) LLM 값을 그대로 둔다.
    조회값이 없으면 항목을 그대로 돌려준다.
    """
    result = [dict(i) for i in items]
    flights, stays = (quotes or {}).get("flights") or [], (quotes or {}).get("stays") or []
    if flights:
        flight_items = [k for k, i in enumerate(result) if "항공" in str(i.get("activity", "")) and not is_restaurant(i)]
        out_k = next((k for k in flight_items if "귀국" not in str(result[k].get("activity", ""))), None)
        back_k = next((k for k in flight_items if "귀국" in str(result[k].get("activity", ""))), None)
        lead = result[out_k] if out_k is not None else (result[back_k] if back_k is not None else None)
        if lead is not None:  # 가는 편과 오는 편이 같은 항공권(왕복)이 되도록 한 번만 고른다
            quote = _pick_flight_quote(flights, plan_type, f"{lead.get('activity', '')} {lead.get('note') or ''}")
            for k, prefix, leg_key in ((out_k, "왕복", "out"), (back_k, "왕복 요금에 포함", "back")):
                if k is None:
                    continue
                item, leg = result[k], quote.get(leg_key) or {}
                if leg_key == "out":
                    item.update(per_person=True, unit_cost_krw=quote["price_krw"], local_unit_price=0)
                item.update(depart_time=leg.get("depart") or item.get("depart_time", ""), arrive_time=leg.get("arrive") or item.get("arrive_time", ""))
                item["note"] = _flight_note(prefix, quote, leg)
    idx = lodging_index(result) if stays else None
    if idx is not None:
        target = _norm_name(str(result[idx].get("activity", "")).split(":", 1)[-1])
        stay = next(
            (s for s in stays if len(_norm_name(s["name"])) >= 3 and (_norm_name(s["name"]) in target or (len(target) >= 3 and target in _norm_name(s["name"])))),
            None,
        )
        if stay:
            result[idx].update(unit_cost_krw=stay["price_krw"], local_unit_price=0, per_person=False)
            result[idx]["note"] = f"조회 요금 {stay['price_krw']:,}원/박 기준"
    return result


def _collect_place_stats(node: Any, found: dict[str, dict]) -> None:
    """JSON 에서 이름(name)과 평점(rating)·리뷰 수(total_ratings 등)가 함께 있는 장소를 찾아 found 에 담는다."""
    if isinstance(node, dict):
        name, rating = node.get("name"), node.get("rating")
        reviews = next((node[k] for k in ("total_ratings", "user_ratings_total", "userRatingCount") if k in node), None)
        if isinstance(name, str) and isinstance(rating, (int, float)) and isinstance(reviews, (int, float)):
            found[name] = {"rating": float(rating), "reviews": int(reviews)}
            level = str(node.get("price_level") or "").removeprefix("PRICE_LEVEL_")
            if level in MEAL_PRICE_BY_LEVEL:  # UNSPECIFIED·FREE 등 기준표에 없는 값은 담지 않는다(LLM 추정으로 둔다)
                found[name]["price_level"] = level
        for value in node.values():
            _collect_place_stats(value, found)
    elif isinstance(node, list):
        for value in node:
            _collect_place_stats(value, found)


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

    def _observe(self, name: str, raw: str) -> None:
        """도구 결과를 코드가 따로 기록해야 할 때 쓰는 훅(기본은 아무것도 안 함)."""

    def _to_sync(self, tool: Any, observer: Any | None = None):
        """비동기 MCP 도구를 동기 호출 가능한 StructuredTool 로 감싼다(결과 문자열만 돌려줌).

        observer(이름, 원본 결과, 호출 인자)를 주면 기본 관측 훅(_observe) 대신 그것을 부른다(요청별 기록 보관용).
        """
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
                if observer:
                    observer(tool.name, text, kwargs)
                else:
                    self._observe(tool.name, text)
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
        self.place_stats: dict[str, dict] = {}  # 검색 결과에서 모은 {장소 이름: {rating, reviews}}
        self._stats_lock = threading.Lock()

    def stats_snapshot(self) -> dict[str, dict]:
        """지금까지 모은 장소 평점·리뷰 수의 복사본."""
        with self._stats_lock:
            return dict(self.place_stats)

    def _observe(self, name: str, raw: str) -> None:
        """장소 검색 결과(JSON)에서 이름·평점·리뷰 수를 모아 둔다(식당 평가 기준 판정용)."""
        if name not in SEARCH_TOOL_NAMES:
            return
        try:
            found: dict[str, dict] = {}
            _collect_place_stats(json.loads(raw), found)
        except ValueError:
            return
        if found:
            with self._stats_lock:
                self.place_stats.update(found)

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
        self._raw_price_tools: list = []

    async def _connect(self) -> tuple[Any, Any]:
        """HTTP(streamable) 로 연결한다. 인증 키는 필요 없다."""
        from mcp.client.streamable_http import streamablehttp_client

        streams = await self._stack.enter_async_context(streamablehttp_client(self._url))
        return streams[0], streams[1]

    def _register(self, tool: Any) -> None:
        """가격 조회 도구만 등록한다(숙소·예약 관련 도구는 쓰지 않는다)."""
        if tool.name in PRICE_TOOL_NAMES:
            self._raw_price_tools.append(tool)
            self.price_tools.append(self._to_sync(tool))

    def quote_tools(self, sink: dict) -> list:
        """가격 조회 도구를 감싸되, 항공·숙소 조회 결과의 구조화된 값을 sink 에 모은다. 요청마다 새 sink 를 줘서 세션이 섞이지 않게 한다."""
        return [self._to_sync(t, observer=lambda name, raw, args: record_price_quotes(sink, name, raw, args)) for t in self._raw_price_tools]

    def _compact(self, name: str, raw: str) -> str:
        """큰 JSON 결과를 목록 몇 줄로 줄인다."""
        return compact_price_result(name, raw)


def _won(value: int) -> str:
    """금액을 '1,234원' 형태로 만든다."""
    return f"{value:,}원"


_LOGISTICS_RE = re.compile(r"항공|귀국|체크인|체크아웃|이동|렌터카|복귀|출국|공항|숙소")
_MEAL_RE = re.compile(r"식사|점심|저녁|아침|조식|브런치|디너|런치|레스토랑|식당|맛집|카페|라멘|스시|오마카세|샤브|뷔페|이자카야|야키니쿠|몬자|돈카츠|텐동")
_SIGHT_RE = re.compile(r"관람|전망|산책|야경|일출|일몰|감상|공원|신사|사찰|사원|박물관|미술관|궁|타워|투어|입장|체험")
_SLOTS = ("아침", "오후", "저녁")
_AFTER_MIDNIGHT_BEFORE = "06:00"  # 저녁 시간대 항목이 이 시각 전에 시작하면 자정을 넘긴 것(야간 항공 도착 후 이동·체크인 등)으로 본다


def schedule_sort_key(item: ItineraryItem) -> tuple:
    """일정 정렬 키: (일차, 시간대, 시작 시각). 저녁 시간대의 06:00 전 시각은 자정을 넘긴 것이라 같은 시간대의 뒤로 보낸다.

    예: 18:10 출발 야간 항공(00:10 도착) 뒤의 00:40 공항→숙소 이동은 시각이 더 작아도 항공 다음에 와야 한다.
    """
    slot = item.get("time_slot")
    time = normalize_time(item.get("depart_time")) or "99:99"
    after_midnight = slot == "저녁" and time < _AFTER_MIDNIGHT_BEFORE
    return (int(item["day"]), _SLOTS.index(slot) if slot in _SLOTS else 3, after_midnight, time)
DIFF_MIN_SENTENCES, DIFF_MAX_SENTENCES, DIFF_MAX_CHARS = 2, 3, 250  # 안별 차별점: 2~3문장, 250자 이내
MAIN_SCHEDULE_COUNT = 5
RESTAURANT_MIN_RATING, RESTAURANT_MIN_REVIEWS = 4.5, 5000  # 주요 일정에 식당을 넣을 최소 평점·리뷰 수


def budget_usage_line(plan_type: str, items: list[ItineraryItem], budget_krw: int | None) -> str:
    """예산 대비 사용 비율과 잔액(또는 초과)을 한 줄로 만든다. 가성비형은 목표(예산의 80%) 충족 여부도 붙인다."""
    if budget_krw is None:
        return "예산 제한 없음"
    result = check_budget(plan_type, items, budget_krw)
    total = result["total"]
    pct = total * 100 / budget_krw
    pct_text = f"{math.ceil(pct * 10) / 10:.1f}%" if pct > 100 else f"{int(pct)}%"
    tail = f"잔액 {_won(budget_krw - total)}" if total <= budget_krw else f"{_won(total - budget_krw)} 초과"
    line = f"{pct_text} 사용, {tail}"
    if plan_type == "가성비형":
        line += ", 목표 80% 이내" if result["within"] else ", 목표 80% 초과"
    return line


def _norm_name(text: Any) -> str:
    """장소 이름 비교용: 소문자로 바꾸고 공백·기호를 뺀다."""
    return re.sub(r"[^0-9a-z가-힣]", "", str(text).lower())


def place_stats_for(activity: str, place_stats: dict | None) -> dict | None:
    """활동명에 이름이 들어 있는 장소의 {rating, reviews} 통계를 돌려준다(여러 개면 이름이 가장 긴 것). 모르면 None."""
    target = _norm_name(activity)
    best: tuple[int, dict] | None = None
    for name, stat in (place_stats or {}).items():
        key = _norm_name(name)
        if len(key) >= 4 and key in target and (best is None or len(key) > best[0]):
            best = (len(key), stat)
    return best[1] if best else None


def is_restaurant(item: ItineraryItem) -> bool:
    """식당 항목인지 판별한다. 모델이 쓴 category 가 있으면 그것을, 없으면 식사 관련 단어(관람·산책 같은 단어가 있으면 제외)로 본다."""
    category = str(item.get("category") or "").strip()
    if category:
        return category == "식당"
    activity = str(item.get("activity", ""))
    return bool(_MEAL_RE.search(activity)) and not _SIGHT_RE.search(activity)


def qualified_restaurant_stats(item: ItineraryItem, place_stats: dict | None) -> dict | None:
    """식당이 평점 4.5 이상이고 리뷰 5,000개 이상이면 그 통계를, 아니거나 통계를 모르면 None."""
    stat = place_stats_for(str(item.get("activity", "")), place_stats)
    if stat and stat.get("rating", 0) >= RESTAURANT_MIN_RATING and stat.get("reviews", 0) >= RESTAURANT_MIN_REVIEWS:
        return stat
    return None


def main_schedule(items: list[ItineraryItem], n: int = MAIN_SCHEDULE_COUNT, place_stats: dict | None = None) -> list[str]:
    """주요 일정 n개를 시간순 문장으로 고른다. 명소 위주이고, 식당은 평점·리뷰 기준을 만족하는 것만 최대 1개 넣는다.

    항공·이동·체크인 같은 이동/숙박 항목은 명소가 부족할 때만 채운다. 일차마다 비용이 큰 명소부터 고른다.
    """

    def cost(k: int) -> int:
        return int(items[k].get("cost_krw") or 0)

    def order(k: int) -> tuple:
        return schedule_sort_key(items[k])

    movable = [k for k in range(len(items)) if not _LOGISTICS_RE.search(str(items[k].get("activity", "")))]
    sights = [k for k in movable if not is_restaurant(items[k])]
    stats = {k: qualified_restaurant_stats(items[k], place_stats) for k in movable if is_restaurant(items[k])}
    best_restaurant = sorted((k for k, st in stats.items() if st), key=lambda k: (-stats[k]["reviews"], k))[:1]
    slots = n - len(best_restaurant)
    by_day: dict[int, list[int]] = {}
    for k in sights:
        by_day.setdefault(int(items[k]["day"]), []).append(k)
    chosen = [max(ks, key=lambda k: (cost(k), -k)) for _, ks in sorted(by_day.items())]
    if len(chosen) > slots:
        chosen = sorted(chosen, key=lambda k: (-cost(k), k))[:slots]
    chosen += sorted((k for k in sights if k not in chosen), key=lambda k: (-cost(k), k))[: slots - len(chosen)]
    chosen += best_restaurant
    if len(chosen) < n:  # 명소가 부족할 때만 이동·숙박 항목으로 채운다(기준 미달 식당은 넣지 않는다)
        logistics = [k for k in range(len(items)) if k not in movable and k not in chosen]
        chosen += sorted(logistics, key=lambda k: (-cost(k), k))[: n - len(chosen)]
    lines = []
    for k in sorted(chosen, key=order):
        time = normalize_time(items[k].get("depart_time"))
        activity = str(items[k].get("activity", "")).strip()
        activity = activity if len(activity) <= 40 else activity[:39] + "…"
        extra = f" ★{stats[k]['rating']:g} 리뷰 {stats[k]['reviews']:,}개" if stats.get(k) else ""
        lines.append(f"{int(items[k]['day'])}일차 {time + ' ' if time else ''}{activity} ({_won(cost(k))}){extra}")
    return lines


def split_sentences(text: Any) -> list[str]:
    """문장 끝(. ! ?) 뒤의 공백에서 문장을 나눈다. '4.5점' 같은 소수점은 나누지 않는다."""
    t = " ".join(str(text or "").split())
    return [part for part in re.split(r"(?<=[.!?])\s+", t) if part]


def difference_ok(text: Any) -> bool:
    """차별점이 2~3문장이고 DIFF_MAX_CHARS 이내인지."""
    t = " ".join(str(text or "").split())
    return DIFF_MIN_SENTENCES <= len(split_sentences(t)) <= DIFF_MAX_SENTENCES and len(t) <= DIFF_MAX_CHARS


def clip_difference(text: Any) -> str:
    """차별점을 앞의 3문장, 250자 이내로 줄인다. 첫 문장 하나가 250자를 넘으면 '…'로 줄인다."""
    kept: list[str] = []
    for sentence in split_sentences(text)[:DIFF_MAX_SENTENCES]:
        if kept and len(" ".join([*kept, sentence])) > DIFF_MAX_CHARS:
            break
        kept.append(sentence)
    result = " ".join(kept)
    return result if len(result) <= DIFF_MAX_CHARS else result[: DIFF_MAX_CHARS - 1].rstrip() + "…"


def fallback_difference(plan_type: str, plan_items: dict[str, list[ItineraryItem]]) -> str:
    """LLM 비교를 쓰지 못할 때 쓰는 수치 기반 차별점(2~3문장): 합계 순위, 가장 싼/비싼 안과의 차이, 대표 일정."""
    items = plan_items[plan_type]
    totals = {pt: total_cost(its) for pt, its in plan_items.items()}
    total = totals[plan_type]
    parts = []
    if len(totals) > 1:
        rank = sorted(totals, key=totals.get, reverse=True).index(plan_type) + 1
        parts.append(f"{plan_type}의 합계는 {_won(total)}으로, {len(totals)}개 안 중 비용이 {rank}번째로 큽니다.")
        cheapest = min(totals, key=totals.get)
        priciest = max(totals, key=totals.get)
        if total > totals[cheapest]:
            parts.append(f"가장 저렴한 {cheapest}보다 {_won(total - totals[cheapest])} 더 듭니다.")
        elif total < totals[priciest]:
            parts.append(f"가장 비싼 {priciest}보다 {_won(totals[priciest] - total)} 덜 듭니다.")
        else:
            parts.append("다른 안과 합계가 같습니다.")
    else:
        parts.append(f"{plan_type}의 합계는 {_won(total)}입니다.")
    names = [re.sub(r"^\d{2}:\d{2} ", "", line.split(" ", 1)[1].rsplit(" (", 1)[0]) for line in main_schedule(items, 3)]
    if names:
        parts.append(f"대표 일정은 {', '.join(names)}입니다.")
    return " ".join(parts)


MONEY_RE = re.compile(r"\d[\d,.]*\s*(?:원|만\s*원|천\s*원|억|%|퍼센트|KRW)")


def drop_money_sentences(text: Any) -> str:
    """LLM 이 쓴 문장 중 금액·비율 숫자가 들어간 문장을 뺀다. 합계와 비율은 코드가 계산한 값만 화면에 보이게 하기 위해서다."""
    return " ".join(s for s in split_sentences(text) if not MONEY_RE.search(s))


def fit_difference(text: Any, fallback: str) -> str:
    """차별점을 2~3문장, 250자 이내로 맞춘다. 금액·비율 숫자가 든 문장은 빼고, 길면 앞 3문장만 남기며, 1문장이면 폴백 문장을 이어 붙여 2문장 이상이 되게 한다."""
    fitted = clip_difference(drop_money_sentences(text))
    if len(split_sentences(fitted)) < DIFF_MIN_SENTENCES and fallback:
        extra = [sentence for sentence in split_sentences(fallback) if sentence not in fitted]
        fitted = clip_difference(" ".join([*split_sentences(fitted), *extra[: DIFF_MIN_SENTENCES - len(split_sentences(fitted))]]))
    return fitted


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
    comparison: dict[str, str] | None = None,
    place_stats: dict | None = None,
) -> str:
    """plainAnswer 텍스트를 만든다. 선택 전이면 3안 요약(안마다 예산 대비·차별점·주요 일정 블록), 선택 후면 선택 안 요약."""
    budget = trip.get("budget_krw")
    if selected is None:
        who = f", {trip['people']}명" if trip.get("people") else ""
        basis = f"예산 {_won(budget)}" if budget is not None else "예산 제한 없음"
        lines = [f"{origin_label(trip)} → {trip['destination']} {trip['days']}일{who}, {basis} 기준 안입니다. 하나를 골라 주세요.",
                 f"조회 기준 날짜: {period_label(trip)}"]
        finals = {pt: final_items(pt, plans[pt]["attempts"], budget) for pt in PLAN_TYPES if pt in plans}
        for plan_type, plan_items in finals.items():
            difference = fit_difference((comparison or {}).get(plan_type), fallback_difference(plan_type, finals))
            lines.append(_summary_line(plan_type, plan_items, budget))
            lines.append(f"  예산 대비: {budget_usage_line(plan_type, plan_items, budget)}")
            lines.append(f"  차별점: {difference}")
            lines.append("  주요 일정:")
            lines.extend(f"  · {text}" for text in main_schedule(plan_items, place_stats=place_stats))
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
    ".report .scroll{overflow-x:auto}"
    ".report table{width:100%;border-collapse:collapse;font-size:12.5px}"
    ".report th,.report td{border-bottom:1px solid #f1f5f9;padding:10px 12px;text-align:left;vertical-align:top}"
    ".report th{white-space:nowrap}.report td:nth-child(1),.report td:nth-child(3),.report td:nth-child(4){white-space:nowrap}"
    ".report td:nth-child(5){width:1%;white-space:nowrap}"  # 시간대 열은 내용 폭만큼만 쓴다(아침·오후·저녁)
    ".report td:nth-child(6){min-width:240px}.report td:nth-child(9){min-width:160px}.report td:nth-child(12){min-width:200px}"
    ".report th{background:#f8fafc;font-weight:700;color:#0f172a}.report .warning{color:#b91c1c;font-weight:600}"
    ".report .notice{color:#6b7280;font-size:13px}.report .card{border:1px solid #e5e7eb;border-radius:10px;padding:12px;margin:8px 0}"
    ".report .badge{font-size:12px;background:#eef2ff;color:#4338ca;border-radius:999px;padding:2px 8px;margin-right:6px}"
    ".report .card.has-img{display:flex;gap:14px;align-items:flex-start}.report .card.has-img>div{flex:1;min-width:0}"
    ".report .card figure{margin:0;flex:0 0 160px}.report .card figure img{width:160px;height:110px;object-fit:cover;border-radius:8px;display:block;background:#f1f5f9}"
    ".report .card figcaption{font-size:10px;color:#94a3b8;margin-top:3px;line-height:1.3}"
    "@media (max-width:560px){.report .card.has-img{flex-direction:column}.report .card figure{flex-basis:auto}.report .card figure img{width:100%;height:160px}}"
    ".report h3{margin:16px 0 6px}"
    ".report h2{font-size:22px;font-weight:800;letter-spacing:-.02em;margin:0 0 4px}"
    ".report .chips{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0 14px}"
    ".report .chip{background:#f1f5f9;color:#475569;border-radius:999px;padding:4px 12px;font-size:12px}"
    ".report .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}"
    ".report .stat{background:#f8fafc;border:1px solid #eef2f7;border-radius:14px;padding:12px 14px}"
    ".report .stat.main{background:#eef2ff;border-color:#e0e7ff}"
    ".report .stat.over{background:#fef2f2;border-color:#fee2e2}"
    ".report .stat .k{display:block;font-size:11px;color:#94a3b8;letter-spacing:.04em;margin-bottom:2px}"
    ".report .stat .v{display:block;font-size:14px;font-weight:700;color:#0f172a;font-variant-numeric:tabular-nums}"
    ".report .stat.main .v{color:#4338ca;font-size:15px}.report .stat.over .v{color:#b91c1c}"
    ".report .bar{height:6px;border-radius:999px;background:#e2e8f0;margin:12px 0 16px;overflow:hidden}"
    ".report .bar i{display:block;height:100%;background:#6366f1;border-radius:999px}.report .bar.over i{background:#ef4444}"
    ".report .warning{background:#fef2f2;border-radius:10px;padding:10px 14px;font-size:13px;margin:0 0 14px}"
    ".report .scroll{border:1px solid #eef2f7;border-radius:14px}"
    ".report table{font-size:12.5px}.report tbody tr:hover{background:#fafaff}"
    ".report th{background:#f8fafc;color:#64748b;font-size:11px;font-weight:600;letter-spacing:.03em;border-bottom:1px solid #e5e7eb}"
    ".report td{border-bottom:1px solid #f1f5f9}.report tbody tr:last-child td{border-bottom:0}"
    ".report td:nth-child(2){font-weight:700;color:#4f46e5;white-space:nowrap}"
    ".report td:nth-child(6){font-weight:600;color:#0f172a}"
    ".report td:nth-child(1),.report td:nth-child(7),.report td:nth-child(8),.report td:nth-child(11),.report td:nth-child(12){color:#64748b}"
    ".report .notice{margin:10px 2px}"
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


_DEFAULT_SIGHT_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 110" preserveAspectRatio="xMidYMid slice">'
    '<rect width="160" height="110" fill="#e8ecf7"/><circle cx="118" cy="32" r="12" fill="#fcd9a0"/>'
    '<path d="M0 110 L48 52 L82 90 L108 64 L160 110 Z" fill="#b7c0dc"/>'
    '<path d="M0 110 L36 76 L70 110 Z" fill="#9aa6cc"/></svg>'
)
# 이미지를 못 찾은 명소에 쓰는 기본 이미지(풍경 아이콘). 퍼센트 인코딩해 따옴표·꺾쇠 없이 속성에 안전하게 넣는다
DEFAULT_SIGHT_IMAGE = "data:image/svg+xml;charset=utf-8," + quote(_DEFAULT_SIGHT_SVG)


def _places_html(guide: dict | None) -> str:
    """탭 2: 맛집·명소 가이드북."""
    places = (guide or {}).get("places") or []
    if not places:
        return GUIDE_MISSING
    parts = []
    for kind in ("맛집", "명소"):
        group = [p for p in places if p.get("kind") == kind]
        if group:
            cards = "".join(_place_card(p) for p in group)
            parts.append(f"<h3>{kind}</h3>{cards}")
    return "".join(parts)


def _place_card(place: dict) -> str:
    """가이드북 카드 하나. 대표 이미지(place['image'])가 있으면 왼쪽에 썸네일과 출처를 보여 준다. 모든 값은 이스케이프한다."""
    body = (
        "<strong>"
        f"<span class='badge'>{escape(str(place.get('area', '')))}</span>{escape(str(place.get('name', '')))}</strong>"
        f"<p>{escape(str(place.get('description', '')))}</p>"
        f"<p class='notice'>팁: {escape(str(place.get('tip', '')))}</p>"
    )
    image = place.get("image") or {}
    url = safe_image_url(image.get("url"))
    if not url:
        if place.get("kind") != "명소":  # 맛집은 대표 이미지를 찾기 어려워 이미지 없이 보여 준다
            return f"<div class='card'>{body}</div>"
        return (  # 이미지를 못 찾은 명소는 기본 이미지로 카드 높이와 모양을 맞춘다(주소가 아니라 데이터라 내보낸 파일에서도 보인다)
            "<div class='card has-img'>"
            f"<figure><img src='{DEFAULT_SIGHT_IMAGE}' alt='{escape(str(place.get('name', '')), quote=True)}' class='default-img'>"
            "<figcaption>대표 이미지 없음</figcaption></figure>"
            f"<div>{body}</div></div>"
        )
    credit = f"이미지: 위키백과 「{escape(str(image.get('title', '')))}」"
    return (
        "<div class='card has-img'>"
        f"<figure><img src='{escape(url, quote=True)}' alt='{escape(str(place.get('name', '')), quote=True)}' "
        "loading='lazy' referrerpolicy='no-referrer'>"
        f"<figcaption>{credit}</figcaption></figure><div>{body}</div></div>"
    )


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
    if item.get("rooms") and item.get("nights") and unit > 0:  # 숙박: 객실 1박 요금 x 객실 수 x 박 수
        return f"{_won(unit)} × {item['rooms']}실 × {item['nights']}박 = {_won(cost)}"
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
    start_date: date | None = None,
    fx: dict | None = None,
    weather: dict | None = None,
) -> str:
    """선택 안의 3탭 HTML 리포트(일정표 / 맛집·명소 가이드북 / 준비물·유의 사항)를 만든다. 탭은 CSS만으로 전환한다.

    fx(환율)가 있으면 일정표 아래에 참고 카드로, weather(날씨 예보)가 있으면 준비물·유의 사항 탭 맨 위에 요약 카드로 보여 준다.
    start_date 가 있으면 일차 앞 열에 'M/D'(예: 3/20) 날짜를 보여 준다. 자정을 넘긴 항목(저녁 시간대 06:00 전)은 다음 날짜로 표시한다.
    """

    def date_cell(i: ItineraryItem) -> str:
        if start_date is None:
            return ""
        shown = start_date + timedelta(days=int(i["day"]) - 1 + (1 if schedule_sort_key(i)[2] else 0))
        return f"{shown.month}/{shown.day}"

    result = check_budget(plan_type, items, budget_krw)
    target = target_budget(plan_type, budget_krw)
    rows = "".join(
        "<tr>"
        f"<td>{date_cell(i)}</td><td>{int(i['day'])}일차</td><td>{normalize_time(i.get('depart_time'))}</td>"
        f"<td>{normalize_time(i.get('arrive_time'))}</td><td>{escape(str(i['time_slot']))}</td>"
        f"<td>{escape(str(i['activity']))}</td>"
        f"<td>{escape(str(i.get('origin', '')))}</td><td>{escape(str(i.get('arrival', '')))}</td>"
        f"<td>{escape(_cost_cell(i, people))}</td>"
        f"<td>{escape(str(i['transport']))}</td><td>{escape(str(i.get('duration', '')))}</td><td>{escape(str(i['note']))}</td>"
        "</tr>"
        for i in sorted(items, key=schedule_sort_key)
    )
    warning = ""
    if not result["within"]:
        warning = (
            f"<p class='warning'>경고: 예산 미충족 — 목표 {_won(target)}를 {_won(result['overage'])} 초과했습니다. "
            "재조정 후에도 맞추지 못한 안 중 가장 근접한 안입니다.</p>"
        )
    tiles = f"<div class='stat main'><span class='k'>합계</span><span class='v'>비용 합계: {_won(result['total'])}</span></div>"
    bar = ""
    if budget_krw is None:
        tiles += "<div class='stat'><span class='k'>예산</span><span class='v'>사용자 예산: 제한 없음</span></div>"
    else:
        left = budget_krw - result["total"]
        pct = min(round(result["total"] / budget_krw * 100), 100) if budget_krw > 0 else 100
        tone = "over" if not result["within"] else ""
        tiles += (
            f"<div class='stat'><span class='k'>예산</span><span class='v'>사용자 예산: {_won(budget_krw)}</span></div>"
            f"<div class='stat'><span class='k'>목표</span><span class='v'>목표 예산: {_won(target)}</span></div>"
            f"<div class='stat {tone}'><span class='k'>잔액</span><span class='v'>예산 대비 잔액: {left:,}원</span></div>"
        )
        bar = f"<div class='bar {tone}'><i style='width:{pct}%'></i></div>"
    budget_lines = f"<div class='stats'>{tiles}</div>{bar}"
    chips = "".join(
        f"<span class='chip'>{escape(text)}</span>"
        for text in (
            f"출발지: {origin_text}" if origin_text else "",
            f"조회 기준 날짜: {period_text}" if period_text else "",
            f"여행 인원: {people}명" if people else "",
        )
        if text
    )
    people_line = f"<div class='chips'>{chips}</div>" if chips else ""
    per_person_note = (
        "<p class='notice'>1인 단위 항목(입장권·식사·항공권 등)은 '1인 가격 × 인원 = 전체 가격'으로 표시합니다.</p>"
        if any(i.get("per_person") for i in items) else ""
    )
    schedule = (
        f"{people_line}{budget_lines}{warning}"
        "<div class='scroll'><table><thead><tr><th>날짜</th><th>일차</th><th>출발시간</th><th>도착시간</th><th>시간대</th><th>활동</th><th>출발지</th><th>도착지</th><th>비용(원)</th><th>교통수단</th><th>소요시간</th><th>비고</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
        f"{per_person_note}"
    )
    if price_note:
        schedule += (
            "<div class='card'><strong>참고 가격 (마이리얼트립 조회, 실시간 아님)</strong>"
            f"<p style='white-space:pre-wrap'>{escape(price_note)}</p>"
            "<p class='notice'>참고 가격은 1인 기준 시작가이며, 일정 비용에는 선택한 항목의 '1인 가격 × 인원'으로 반영됩니다.</p></div>"
        )
    if fx_text(fx):
        schedule += (
            "<div class='card'><strong>환율 (Frankfurter, ECB 기준)</strong>"
            f"<p>{escape(fx_text(fx))}</p>"
            "<p class='notice'>현지 통화 가격은 이 환율로 원화로 환산해 일정 비용에 반영했습니다.</p></div>"
        )
    return (
        f"{REPORT_STYLE}<section class='report'><h2>{escape(plan_type)} 일정</h2>"
        "<p class='notice'>일정과 가격은 AI 추정값이며 실제와 다를 수 있습니다.</p>"
        "<input class='tab' type='radio' name='rt' id='rt1' checked><input class='tab' type='radio' name='rt' id='rt2'>"
        "<input class='tab' type='radio' name='rt' id='rt3'>"
        "<div class='tabs'><label for='rt1'>일정표</label><label for='rt2'>맛집 및 명소 가이드북</label>"
        "<label for='rt3'>준비물·유의 사항</label></div>"
        f"<div class='panel p1'>{schedule}</div><div class='panel p2'>{_places_html(guide)}</div>"
        f"<div class='panel p3'>{_weather_html(weather)}{_prep_html(guide)}</div></section>"
    )


NAVER_LOCAL_URL = "https://naverapihub.apigw.ntruss.com/search/v1/local"
NAVER_DRIVING_URL = "https://maps.apigw.ntruss.com/map-direction-15/v1/driving"
NAVER_COORD_SCALE = 10_000_000  # 지역검색의 mapx/mapy 는 경도·위도 x 10^7 정수로 온다
_NON_CAR_RE = re.compile(r"항공|비행|공항|KTX|SRT|기차|열차|귀국|페리|크루즈")  # 자동차 경로로 계산하면 안 되는 이동
_CAR_TRANSPORT = "자동차"


def _norm_name(text: str) -> str:
    """장소 이름 비교용: 소문자로 바꾸고 공백·기호를 지운다."""
    return re.sub(r"[\W_]+", "", str(text or "").lower())


def format_minutes(minutes: int) -> str:
    """분을 '약 28분', '약 2시간 5분', '약 3시간' 형태로 쓴다."""
    minutes = max(int(minutes), 1)
    if minutes < 60:
        return f"약 {minutes}분"
    hours, rest = divmod(minutes, 60)
    return f"약 {hours}시간 {rest}분" if rest else f"약 {hours}시간"


class NaverMaps:
    """국내 이동 계산용 네이버 API 묶음: 지역검색(이름→좌표)과 Directions 15(자동차 경로·시간·요금). 둘만 쓴다.

    키는 호출하는 쪽이 .env 에서 읽어 넘기며 로그나 예외 메시지에 남기지 않는다. 호출이 실패하면 None 을 돌려준다.
    """

    def __init__(self, local_keys: tuple[str, str], maps_keys: tuple[str, str], http: Any | None = None, timeout: float = 15.0):
        import requests

        self._local = {"X-NCP-APIGW-API-KEY-ID": local_keys[0], "X-NCP-APIGW-API-KEY": local_keys[1]}
        self._maps = {"x-ncp-apigw-api-key-id": maps_keys[0], "x-ncp-apigw-api-key": maps_keys[1]}
        self._http = http or requests
        self._timeout = timeout

    def search_place(self, query: str, region: str = "") -> dict | None:
        """이름(상호·명소)으로 장소 1곳을 찾아 {name, category, address, lng, lat} 로 돌려준다.

        제목이 검색어와 겹치거나 주소에 지역명(region)이 들어 있는 결과만 채택한다(region 이 있으면 지역 일치 필수).
        """
        import html

        try:
            resp = self._http.get(NAVER_LOCAL_URL, headers=self._local, params={"query": query, "display": 5}, timeout=self._timeout)
            if not resp.ok:
                return None
            items = resp.json().get("items") or []
        except Exception as exc:  # 조회 실패는 추정값으로 대신한다(키는 남기지 않는다)
            logger.warning("네이버 지역검색 실패: %s", type(exc).__name__)
            return None
        tokens = [_norm_name(re.sub(r"(특별시|광역시|특별자치도|특별자치시|도|시|군)$", "", t)) for t in region.split()]
        tokens = [t for t in tokens if t]
        want = _norm_name(query)
        best, best_score = None, 0
        for it in items:
            title = html.unescape(re.sub(r"<[^>]+>", "", str(it.get("title", "")))).strip()
            address = f"{it.get('roadAddress', '')} {it.get('address', '')}"
            norm_title = _norm_name(title)
            title_hit = bool(norm_title) and (norm_title in want or want in norm_title)
            region_hit = any(t in _norm_name(address) for t in tokens)
            if tokens and not region_hit:
                continue
            if not tokens and not title_hit:
                continue
            score = 2 * title_hit + region_hit
            if score > best_score:
                best, best_score = (it, title), score
        if best is None:
            return None
        it, title = best
        try:
            lng, lat = float(it["mapx"]), float(it["mapy"])
        except (KeyError, TypeError, ValueError):
            return None
        if abs(lng) > 1000:  # 정수 표기(x10^7)면 도 단위로 되돌린다
            lng, lat = lng / NAVER_COORD_SCALE, lat / NAVER_COORD_SCALE
        return {"name": title, "category": str(it.get("category", "")), "address": str(it.get("roadAddress") or it.get("address") or ""), "lng": lng, "lat": lat}

    def drive_route(self, start: tuple[float, float], goal: tuple[float, float]) -> dict | None:
        """자동차 경로를 조회해 {distance_m, minutes, toll, fuel, taxi} 로 돌려준다. 실패하면 None.

        start·goal 은 (경도, 위도). 소요시간은 응답의 밀리초를 분으로 바꾼다. 실시간 교통이 반영돼 호출마다 조금 다르다.
        """
        params = {"start": f"{start[0]},{start[1]}", "goal": f"{goal[0]},{goal[1]}"}
        try:
            resp = self._http.get(NAVER_DRIVING_URL, headers=self._maps, params=params, timeout=self._timeout)
            if not resp.ok:
                return None
            data = resp.json()
            if data.get("code") != 0:
                return None
            summary = data["route"]["traoptimal"][0]["summary"]
            return {
                "distance_m": int(summary["distance"]),
                "minutes": max(round(int(summary["duration"]) / 60000), 1),
                "toll": int(summary.get("tollFare") or 0),
                "fuel": int(summary.get("fuelPrice") or 0),
                "taxi": int(summary.get("taxiFare") or 0),
            }
        except Exception as exc:
            logger.warning("네이버 경로 조회 실패: %s", type(exc).__name__)
            return None


def apply_domestic_routes(items: list[dict], queries: dict[int, str], naver: Any, region: str) -> tuple[list[dict], int]:
    """국내 일정에서 같은 날 연속한 장소 사이의 자동차 경로를 조회해 항목을 채운다. (새 항목 목록, 채운 항목 수)를 돌려준다.

    queries 는 {항목 번호: 장소 검색어}. 채우는 값은 출발지·도착지·소요시간·교통수단(자동차)이고, 분류가 '이동' 인 항목은
    비용을 통행료+연료비(공동 비용)로 바꾸고 비고에 거리·택시 요금을 적는다. 항공·기차·귀국 항목과 조회에 실패한 구간은 그대로 둔다.
    """
    out = [dict(i) for i in items]
    cache: dict[str, dict | None] = {}
    last_by_day: dict[Any, dict] = {}
    filled = 0
    for idx, item in enumerate(out):
        query = queries.get(idx)
        if not query:
            continue
        if query not in cache:
            cache[query] = naver.search_place(query, region)
        place = cache[query]
        if place is None:
            continue
        prev = last_by_day.get(item.get("day"))
        last_by_day[item.get("day")] = place
        if prev is None or (prev["lng"], prev["lat"]) == (place["lng"], place["lat"]):
            continue
        if _NON_CAR_RE.search(f"{item.get('activity', '')} {item.get('transport', '')}"):
            continue
        route = naver.drive_route((prev["lng"], prev["lat"]), (place["lng"], place["lat"]))
        if route is None:
            continue
        item["origin"], item["arrival"] = prev["name"], place["name"]
        item["duration"] = format_minutes(route["minutes"])
        item["transport"] = _CAR_TRANSPORT
        if item.get("category") == "이동":
            item["cost_krw"] = route["toll"] + route["fuel"]
            item["per_person"], item["unit_cost_krw"] = False, 0
            item["note"] = clip_note(f"{route['distance_m'] / 1000:.1f}km·택시 약 {route['taxi']:,}원")
        filled += 1
    return out, filled


FRANKFURTER_URL = "https://api.frankfurter.dev/v1/latest"
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/forecast"
FORECAST_DAYS = 5  # OpenWeatherMap 무료 예보는 오늘부터 5일치(3시간 간격)다


class FrankfurterRates:
    """환율 조회: Frankfurter(ECB 기준, 무료·키 불필요). 현지 통화 1단위가 몇 원인지만 돌려준다. 실패하면 None."""

    def __init__(self, http: Any | None = None, timeout: float = 10.0):
        import requests

        self._http = http or requests
        self._timeout = timeout

    def rate_to_krw(self, currency: str) -> dict | None:
        """통화 코드(ISO 4217, 예: SGD) 1단위의 원화 환율을 {currency, krw_per_unit, date} 로 돌려준다. 지원하지 않는 통화·실패는 None."""
        code = str(currency or "").strip().upper()
        if not re.fullmatch(r"[A-Z]{3}", code):
            return None
        if code == "KRW":
            return {"currency": "KRW", "krw_per_unit": 1.0, "date": date.today().isoformat()}
        try:
            resp = self._http.get(FRANKFURTER_URL, params={"base": code, "symbols": "KRW"}, timeout=self._timeout)
            if not resp.ok:
                return None
            data = resp.json()
            return {"currency": code, "krw_per_unit": float(data["rates"]["KRW"]), "date": str(data.get("date", ""))}
        except Exception as exc:  # 조회 실패는 LLM 추정 환율로 대신한다
            logger.warning("환율 조회 실패: %s", type(exc).__name__)
            return None


class OpenWeatherForecast:
    """날씨 조회: OpenWeatherMap 5일 예보(무료 플랜, 분당·월간 호출 한도 있음). 키는 로그나 예외 메시지에 남기지 않는다. 실패하면 None."""

    def __init__(self, api_key: str, http: Any | None = None, timeout: float = 10.0):
        import requests

        self._key = api_key
        self._http = http or requests
        self._timeout = timeout

    def daily_forecast(self, query: str) -> dict | None:
        """도시 검색어('Singapore,SG' 형식)의 예보를 현지 날짜별 {date, min, max, description, pop} 목록으로 묶어 돌려준다. 실패는 None."""
        from collections import Counter

        params = {"q": query, "appid": self._key, "units": "metric", "lang": "kr"}
        try:
            resp = self._http.get(OPENWEATHER_URL, params=params, timeout=self._timeout)
            if not resp.ok:
                return None
            data = resp.json()
            offset = int(data.get("city", {}).get("timezone") or 0)  # 도시 시간대(UTC 초)로 현지 날짜를 정한다
            by_day: dict[str, list[dict]] = {}
            for entry in data["list"]:
                local = datetime.fromtimestamp(int(entry["dt"]) + offset, tz=timezone.utc).date()
                by_day.setdefault(local.isoformat(), []).append(entry)
            days = []
            for day, entries in sorted(by_day.items()):
                desc = Counter(str(e["weather"][0]["description"]) for e in entries).most_common(1)[0][0]
                days.append({
                    "date": day,
                    "min": round(min(float(e["main"]["temp_min"]) for e in entries)),
                    "max": round(max(float(e["main"]["temp_max"]) for e in entries)),
                    "description": desc,
                    "pop": round(max(float(e.get("pop") or 0) for e in entries) * 100),
                })
            return {"city": str(data.get("city", {}).get("name") or query), "days": days}
        except Exception as exc:
            logger.warning("날씨 조회 실패: %s", type(exc).__name__)
            return None


def trip_weather(forecast: dict | None, start: date, end: date) -> dict | None:
    """예보 중 여행 기간(start~end)에 걸치는 날을 고른다. 하나도 없으면 in_range=False 로 두고 가까운 예보 전체를 참고용으로 남긴다."""
    if not forecast or not forecast.get("days"):
        return None
    picked = [d for d in forecast["days"] if start.isoformat() <= d["date"] <= end.isoformat()]
    return {"city": forecast["city"], "in_range": bool(picked), "days": picked or forecast["days"]}


def make_condition_tools(rates: Any | None = None, weather: Any | None = None) -> list:
    """환율·날씨 LangChain Tool 목록을 만든다. 연결된 클라이언트(rates·weather)에 해당하는 도구만 들어간다.

    도구는 실패해도 예외 대신 {"error": ...} 를 돌려줘서 일정 생성이 멈추지 않는다.
    """
    from langchain_core.tools import tool

    tools: list = []
    if rates is not None:

        @tool
        def get_exchange_rate(currency: str) -> dict:
            """현지 통화 1단위가 몇 원인지 환율을 조회한다(Frankfurter, ECB 기준). currency 는 SGD·JPY·USD 같은 ISO 4217 코드."""
            found = rates.rate_to_krw(currency)
            return found if found else {"error": f"{currency} 환율을 조회하지 못했습니다"}

        tools.append(get_exchange_rate)
    if weather is not None:

        @tool
        def get_weather_forecast(city: str, start_date: str, end_date: str) -> dict:
            """도시의 5일 날씨 예보를 조회한다(OpenWeatherMap). city 는 'Singapore,SG' 형식, 날짜는 YYYY-MM-DD(여행 기간)."""
            try:
                start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
            except ValueError:
                return {"error": "날짜 형식이 올바르지 않습니다"}
            found = trip_weather(weather.daily_forecast(city), start, end)
            return found if found else {"error": f"{city} 날씨를 조회하지 못했습니다"}

        tools.append(get_weather_forecast)
    return tools


def fx_text(fx: dict | None) -> str:
    """환율 한 줄 문장. 예: '1 SGD = 1,020.5원 (ECB 기준, 2026-11-04)'. 없으면 빈 문자열."""
    if not fx or not fx.get("krw_per_unit"):
        return ""
    rate = f"{float(fx['krw_per_unit']):,.2f}".rstrip("0").rstrip(".")
    return f"1 {fx['currency']} = {rate}원 (ECB 기준, {fx.get('date', '')})"


def weather_text(weather: dict | None) -> str:
    """날씨 요약 문장(날짜별 한 줄). 예: '11/22 맑음 18~24℃ 강수확률 10%'. 없으면 빈 문자열."""
    if not weather or not weather.get("days"):
        return ""
    lines = []
    for d in weather["days"]:
        month, day = int(d["date"][5:7]), int(d["date"][8:10])
        lines.append(f"{month}/{day} {d['description']} {d['min']}~{d['max']}℃ 강수확률 {d['pop']}%")
    return "\n".join(lines)


RAIN_POP_THRESHOLD = 50  # 강수확률이 이 값(%) 이상인 날을 '비 가능성이 높은 날'로 본다
HOT_MAX_C, COLD_MIN_C = 28, 10  # 이 기온 이상이면 더위, 이하면 추위 준비물을 권한다


def weather_summary(weather: dict | None) -> list[str]:
    """날씨 예보를 준비물·유의 사항 탭에 넣을 요약 문장 목록으로 만든다(코드 계산, LLM 호출 없음).

    기간·기온 범위·대표 날씨·비 가능성이 높은 날과, 그에 맞는 준비 제안을 담는다. 예보가 없으면 빈 목록.
    """
    from collections import Counter

    days = (weather or {}).get("days") or []
    if not days:
        return []

    def md(day: dict) -> str:
        return f"{int(day['date'][5:7])}/{int(day['date'][8:10])}"

    lines = [
        f"예보 기간: {md(days[0])}" + (f" ~ {md(days[-1])}" if len(days) > 1 else ""),
        f"기온: {min(d['min'] for d in days)}~{max(d['max'] for d in days)}℃, "
        f"주된 날씨: {Counter(d['description'] for d in days).most_common(1)[0][0]}",
    ]
    rainy = [d for d in days if d["pop"] >= RAIN_POP_THRESHOLD]
    if rainy:
        lines.append("비 가능성이 높은 날: " + ", ".join(f"{md(d)}({d['pop']}%)" for d in rainy))
    else:
        lines.append(f"강수 가능성은 낮은 편입니다(최대 {max(d['pop'] for d in days)}%).")
    tips = []
    if rainy:
        tips.append("우산·우비, 방수 신발")
    if max(d["max"] for d in days) >= HOT_MAX_C:
        tips.append("자외선 차단제, 통풍이 잘 되는 옷차림")
    if min(d["min"] for d in days) <= COLD_MIN_C:
        tips.append("겉옷·방한 용품")
    if tips:
        lines.append("준비 제안: " + " / ".join(tips))
    return lines


def _weather_html(weather: dict | None) -> str:
    """준비물·유의 사항 탭 맨 위에 넣는 날씨 요약 카드. 예보가 없으면 빈 문자열. 여행 기간이 예보 범위 밖이면 참고용이라고 밝힌다."""
    lines = weather_summary(weather)
    if not lines:
        return ""
    in_range = bool(weather.get("in_range"))
    title = "날씨 요약" if in_range else "날씨 요약 (참고용)"
    items = "".join(f"<li>{escape(line)}</li>" for line in lines)
    note = "" if in_range else (
        "<p class='notice'>여행 기간이 5일 예보 범위 밖이라 오늘 기준 앞으로 5일 예보를 참고용으로 보여 드립니다. "
        "실제 여행 날씨와 다를 수 있으니 출발 전에 다시 확인하세요.</p>"
    )
    return (
        f"<div class='card'><strong>{title} — {escape(str(weather.get('city', '')))} (OpenWeatherMap)</strong>"
        f"<ul>{items}</ul>{note}</div>"
    )


WIKIPEDIA_API = "https://{lang}.wikipedia.org/w/api.php"
WIKI_IMAGE_HOSTS = ("wikimedia.org",)  # 리포트에 넣어도 되는 이미지 호스트(그 밖의 주소는 버린다)
WIKI_THUMB_SIZE = 480
IMAGE_WORKERS = 4
WIKI_USER_AGENT = "TravelPlannerAgent/0.1 (place image lookup)"  # 위키미디어는 식별 가능한 User-Agent 를 요구한다


def _title_core(text: Any) -> str:
    """이름 비교용: 괄호 안 설명을 지우고 소문자·공백·기호를 없앤다. 예: '센소지 (도쿄)' → '센소지'."""
    return _norm_name(re.sub(r"[\(（][^\)）]*[\)）]", "", str(text or "")))


def image_title_matches(title: str, name: str) -> bool:
    """검색된 문서 제목이 장소 이름과 같은 대상인지 본다(한쪽이 다른 쪽을 포함). 엉뚱한 문서의 사진을 쓰지 않으려는 안전장치."""
    a, b = _title_core(title), _title_core(name)
    return bool(a) and bool(b) and (a in b or b in a)


def safe_image_url(url: Any) -> str:
    """https 이고 허용 호스트(위키미디어)인 이미지 주소만 돌려준다. 아니면 빈 문자열."""
    m = re.fullmatch(r"https://([A-Za-z0-9.-]+)/[^\s\"'<>]+", str(url or ""))
    if not m:
        return ""
    host = m.group(1).lower()
    return str(url) if any(host == h or host.endswith("." + h) for h in WIKI_IMAGE_HOSTS) else ""


class WikipediaImages:
    """명소 대표 이미지 조회: 위키백과(키 불필요)의 문서 대표 이미지(썸네일). 이름이 맞는 문서가 없거나 이미지가 없으면 None.

    장소 이름으로 먼저 검색하고, 맞는 문서가 없으면 지역명을 붙여 한 번 더 찾는다. 실패는 예외 대신 None 이다.
    """

    def __init__(self, http: Any | None = None, timeout: float = 8.0, lang: str = "ko"):
        import requests

        self._http = http or requests
        self._timeout = timeout
        self._lang = lang

    def _search(self, query: str) -> list[dict]:
        """검색 결과 문서 목록(순위순)을 돌려준다. 실패하면 빈 목록."""
        try:
            resp = self._http.get(
                WIKIPEDIA_API.format(lang=self._lang),
                params={"action": "query", "generator": "search", "gsrsearch": query, "gsrlimit": 3, "prop": "pageimages",
                        "piprop": "thumbnail", "pithumbsize": WIKI_THUMB_SIZE, "format": "json", "formatversion": 2},
                headers={"User-Agent": WIKI_USER_AGENT}, timeout=self._timeout,
            )
            if not resp.ok:
                return []
            pages = resp.json().get("query", {}).get("pages", [])
            return sorted(pages, key=lambda p: p.get("index", 99))
        except Exception as exc:  # 이미지는 없어도 리포트는 만들어진다
            logger.warning("명소 이미지 조회 실패: %s", type(exc).__name__)
            return []

    def find(self, name: str, region: str = "") -> dict | None:
        """장소 이름의 대표 이미지를 {url, title, page} 로 돌려준다. 못 찾으면 None."""
        queries = [name] + ([f"{region} {name}"] if region else [])
        for query in queries:
            for page in self._search(query):
                url = safe_image_url((page.get("thumbnail") or {}).get("source"))
                if url and image_title_matches(str(page.get("title", "")), name):
                    title = str(page["title"])
                    return {"url": url, "title": title, "page": f"https://{self._lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"}
        return None


def attach_place_images(guide: dict | None, finder: Any, region: str = "") -> tuple[dict | None, int]:
    """가이드의 명소(kind=명소)마다 대표 이미지를 찾아 place['image']={url, title, page} 로 붙인다. (새 가이드, 붙인 개수)를 돌려준다.

    맛집은 대표 이미지를 찾기 어려워 건드리지 않는다. 조회는 병렬로 하며, 하나가 실패해도 나머지는 계속한다. 원본은 바꾸지 않는다.
    """
    from concurrent.futures import ThreadPoolExecutor

    if not guide or not guide.get("places"):
        return guide, 0
    places = [dict(p) for p in guide["places"]]
    targets = [p for p in places if p.get("kind") == "명소" and p.get("name")]

    def lookup(place: dict) -> dict | None:
        try:
            return finder.find(str(place["name"]), region)
        except Exception as exc:
            logger.warning("명소 이미지 조회 실패: %s", type(exc).__name__)
            return None

    with ThreadPoolExecutor(max_workers=IMAGE_WORKERS) as pool:
        found = list(pool.map(lookup, targets))
    count = 0
    for place, image in zip(targets, found):
        if image and safe_image_url(image.get("url")):
            place["image"] = image
            count += 1
    return {**guide, "places": places}, count
