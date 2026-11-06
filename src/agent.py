"""여행 일정 에이전트 그래프: LangGraph StateGraph (DESIGN 2~9절). 예산 판정은 tools.py 코드가 한다."""
import logging
from typing import Annotated, Any, Literal, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import RetryPolicy, Send, interrupt
from pydantic import BaseModel, ValidationError

from src.tools import (
    PLAN_TYPES,
    ItineraryItem,
    PlanState,
    PlanType,
    TripRequest,
    NOTE_MAX_CHARS,
    apply_domestic_routes,
    check_budget,
    clip_note,
    closest_attempt,
    difference_ok,
    fallback_difference,
    final_items,
    finalize_items,
    fit_difference,
    needs_readjust,
    render_plain_answer,
    period_label,
    render_report_html,
    split_sentences,
    target_budget,
    trip_dates,
    trip_origin,
)

MAX_OUTPUT_TOKENS = 16384  # 긴 일정(항목 많음)이 중간에 잘려 필수 필드가 빠지는 것을 막는다
STRUCTURED_ATTEMPTS = 3  # 구조화 출력 파싱 실패 시 같은 프롬프트로 다시 시도하는 횟수
logger = logging.getLogger(__name__)
BEDROCK_READ_TIMEOUT = 120  # 응답이 이 시간(초) 안에 오지 않으면 끊고 다시 시도한다. 정상 일정 생성은 약 45~60초라 여유가 있다
BEDROCK_CONNECT_TIMEOUT = 10
LLM_ATTEMPTS = 3  # LLM 호출 최대 시도 횟수(최초 포함). botocore 자체 재시도는 끄고 이 횟수로만 센다(타임아웃이 곱절로 쌓이지 않게)
RETRY_BACKOFF = (2.0, 6.0)  # 재시도 전 대기(초)
_NON_RETRYABLE = (ValueError, TypeError, ArithmeticError, ImportError, LookupError, NameError, SyntaxError, RuntimeError, ReferenceError, StopIteration)
_RETRYABLE_AWS_CODES = {"ThrottlingException", "ServiceUnavailableException", "InternalServerException", "ModelTimeoutException", "ModelNotReadyException"}


def is_retryable(exc: Exception) -> bool:
    """다시 시도할 만한 오류인지 판단한다. 읽기·연결 시간 초과와 Bedrock 일시 오류(쓰로틀링 등)는 True, 코드·형식 오류는 False."""
    from botocore.exceptions import ClientError, ConnectTimeoutError, EndpointConnectionError, ReadTimeoutError

    if isinstance(exc, (ReadTimeoutError, ConnectTimeoutError, EndpointConnectionError)):
        return True
    if isinstance(exc, ClientError):
        return exc.response.get("Error", {}).get("Code", "") in _RETRYABLE_AWS_CODES
    return not isinstance(exc, _NON_RETRYABLE)


LLM_RETRY = RetryPolicy(max_attempts=LLM_ATTEMPTS, initial_interval=RETRY_BACKOFF[0], backoff_factor=3.0, retry_on=is_retryable)
PLAN_STYLE = {
    "맛집형": "현지 맛집과 식사 경험을 중심으로",
    "명소형": "대표 명소와 관광 코스를 중심으로",
    "가성비형": "비용을 아끼는 가성비 중심으로",
}


class ItemModel(BaseModel):
    """구조화 출력용 일정 항목 (영문 키, time_slot 은 아침/오후/저녁 고정)."""

    day: int
    time_slot: Literal["아침", "오후", "저녁"]
    activity: str
    cost_krw: int
    transport: str
    note: str
    origin: str = ""  # 출발지
    arrival: str = ""  # 도착지
    duration: str = ""  # 이동 소요시간(예: "약 30분")
    depart_time: str = ""  # 출발(시작) 시각 HH:MM (24시간, 예: 09:00)
    arrive_time: str = ""  # 도착(종료) 시각 HH:MM
    category: str = ""  # 식당 / 명소 / 이동 / 숙박 / 기타 중 하나(주요 일정에서 식당을 가려내는 데 쓴다)
    per_person: bool = False  # 입장권·식사·항공권처럼 1인 단위 비용이면 True
    unit_cost_krw: int = 0  # per_person 일 때 1인 가격(원), 숙박 항목이면 객실 1박 요금(원). 전체 비용 cost_krw 는 코드가 곱해 다시 계산한다
    local_unit_price: float = 0  # 현지 통화로 추정한 단가(숫자만). 환율을 곱해 원화로 바꾸는 일은 코드가 한다
    local_currency: str = ""  # local_unit_price 의 통화 코드(예: JPY)


class PlanOutput(BaseModel):
    """구조화 출력 스키마: 일정 항목 목록."""

    items: list[ItemModel]


class PlanDifference(BaseModel):
    """안 하나의 차별점 설명(다른 안과 비교한 글)."""

    plan_type: PlanType
    difference: str


class PlanComparison(BaseModel):
    """구조화 출력 스키마: 안별 차별점 설명 목록."""

    plans: list[PlanDifference]


class PlaceModel(BaseModel):
    """가이드북 장소 하나: 종류(맛집/명소), 이름, 지역, 설명, 팁."""

    kind: Literal["맛집", "명소"]
    name: str
    area: str
    description: str
    tip: str


class NoteModel(BaseModel):
    """제목과 설명으로 된 유의 정보 하나."""

    title: str
    detail: str


class GuideOutput(BaseModel):
    """가이드 구조화 출력: 맛집·명소, 준비물, 여행 중 유의 사항, 목적지 사전 유의 정보."""

    places: list[PlaceModel]
    packing: list[str]
    cautions: list[NoteModel]
    destination_notes: list[NoteModel]


class RegionOutput(BaseModel):
    """목적지가 대한민국 밖(해외)인지 판정하고, 환율·날씨 조회에 쓸 통화 코드와 도시 검색어를 함께 돌려주는 구조화 출력."""

    overseas: bool
    currency: str = ""  # 현지 통화 ISO 4217 코드(예: SGD). 국내면 KRW
    weather_query: str = ""  # OpenWeatherMap 도시 검색어, 영어 'City,국가코드'(예: Singapore,SG)


class PlaceQuery(BaseModel):
    """일정 항목 1개의 지도 검색어."""

    index: int  # get 목록의 항목 번호(0부터)
    query: str  # 지명 또는 상호 + 지역명. 지도에서 찾을 장소가 아니면 빈 문자열


class PlaceQueries(BaseModel):
    """국내 경로 조회용 장소 검색어 구조화 출력."""

    items: list[PlaceQuery]


def _merge_plans(old: dict, new: dict) -> dict:
    """병렬 안 생성 결과를 안 종류 키로 합치는 리듀서."""
    return {**(old or {}), **(new or {})}


def _concat(a: list, b: list) -> list:
    """누적 리듀서: 두 목록을 이어 붙인다."""
    return (a or []) + (b or [])


class GraphState(TypedDict, total=False):
    """그래프 상태. 원본 데이터만 두고 합계·HTML·프롬프트는 두지 않는다."""

    trip: TripRequest
    plans: Annotated[dict[str, PlanState], _merge_plans]
    comparison: dict[str, str]  # 안별 차별점 설명(2~3문장). 3안이 모인 뒤 한 번 만든다
    place_stats: dict[str, dict]  # Google Maps 검색 결과에서 모은 {장소 이름: {rating, reviews, price_level(있을 때만)}} (식당 기준 판정과 식사 가격 확정용)
    selected: PlanType | None
    selected_items: list[ItineraryItem]
    research: str | None  # Google Maps MCP 로 조회한 장소 목록(MOCKDATA_USE=False 일 때만)
    overseas: bool  # 목적지가 해외인지(해외일 때만 경로·소요시간을 조회)
    price_quotes: dict | None  # 마이리얼트립 조회 결과에서 코드가 모은 {flights: [...], stays: [...]} (항공·숙소 가격을 코드가 확정할 때 씀)
    price_research: str | None  # 마이리얼트립 MCP 로 조회한 입장권·투어·항공권 참고 가격(MOCKDATA_USE=False 일 때만)
    fx: dict | None  # 환율 도구 결과 {currency, krw_per_unit, date}. 해외이고 조회에 성공했을 때만
    weather: dict | None  # 날씨 도구 결과 {city, in_range, days}. 조회에 성공했을 때만
    guide: dict | None  # 선택 안 기준 가이드 원본(맛집·명소, 준비물, 유의 사항). 안 선택 직후 1회 생성
    selected_attempts: list[list[ItineraryItem]]  # 수정 1회마다 새로 시작하는 재조정 시도 (원본)
    next_action: str | None  # await_chat 재개 값: "chat" | "done"
    chat_history: Annotated[list, add_messages]
    tool_results: Annotated[list[dict], _concat]
    trace: Annotated[list[str], _concat]
    errors: Annotated[list[str], _concat]


def build_model() -> Any:
    """config 의 리전·모델 ID 로 ChatBedrockConverse 를 만든다 (테스트에서는 가짜 모델로 교체)."""
    from langchain_aws import ChatBedrockConverse

    from src.config import export_aws_credentials, load_settings

    from botocore.config import Config

    export_aws_credentials()
    settings = load_settings()
    client_config = Config(
        read_timeout=BEDROCK_READ_TIMEOUT,
        connect_timeout=BEDROCK_CONNECT_TIMEOUT,
        retries={"max_attempts": 0},  # 재시도는 LangGraph 재시도 정책과 generate_plan 이 맡는다
    )
    return ChatBedrockConverse(model=settings.model_id, region_name=settings.region, max_tokens=MAX_OUTPUT_TOKENS, config=client_config)


COST_RULES = (
    "비용 규칙: 입장권·식당 식사·항공권처럼 1인 단위 비용은 per_person=true 로 두고 unit_cost_krw 에 1인 가격(원)을 쓰세요"
    "(전체 비용은 코드가 인원수를 곱해 계산합니다). 택시·렌터카처럼 인원과 무관한 공용 비용은 "
    "per_person=false 로 두고 cost_krw 에 전체 금액을 쓰세요. 합계 계산과 예산 판정은 코드가 하니 직접 더하거나 곱하지 마세요. "
    "금액은 현실적인 시세로 쓰고, 식사·숙박·항공을 예산에 맞추려고 0원으로 쓰지 마세요. "
)
NOTE_RULE = f"비고(note)는 핵심만 {NOTE_MAX_CHARS}자 이내로 짧게 쓰세요. "
CATEGORY_RULE = (
    "분류 규칙: 모든 항목의 category 는 식당, 명소, 이동, 숙박, 기타 중 하나로 쓰세요"
    "(식사·카페는 식당, 관광·전망대·체험·시장 구경은 명소, 공항·항공·교통은 이동, 체크인은 숙박). "
)
TIME_RULE = (
    "시각 규칙: 모든 항목에 출발시간(depart_time)과 도착시간(arrive_time)을 24시간 HH:MM 형식(예: 09:00)으로 쓰세요. "
    "활동이면 시작·종료 시각이고, 이동·항공편이면 출발·도착 시각입니다. 항공편은 조회된 시각을 그대로 쓰세요. "
)


def _flight_rule(origin: str, destination: str) -> str:
    """프롬프트용 항공권 지시: 출발지 → 목적지 가는 편(1일차)과 목적지 → 출발지 귀국 편(마지막 날, 요금은 왕복에 포함)."""
    return (
        f"항공권: {origin}에서 항공으로 가야 하는 여행지(해외, 제주 등)면 두 항목을 만드세요. "
        f"(1) 1일차에 '{origin}→{destination} 항공' 항목: per_person=true, unit_cost_krw=1인 왕복 요금, "
        "조회된 항공편이 있으면 그 출발·도착 시각을 depart_time·arrive_time 에, 날짜와 항공편명을 note 에 쓰세요. "
        f"(2) 마지막 날에 '{destination}→{origin} 귀국 항공' 항목: per_person=false, cost_krw=0, note 에 '왕복 요금에 포함'과 "
        "오는 편의 날짜·항공편명, depart_time·arrive_time 에 오는 편의 출발·도착 시각을 쓰세요. "
        f"{origin}에서 지상 이동이 자연스러운 곳이면 항공권 항목을 넣지 마세요. "
    )


def _budget_phrase(plan_type: str, budget: int | None, people: int | None = None, days: int | None = None) -> str:
    """프롬프트용 예산 문장. 예산이 없으면 금액 상한 없이 만들라고 알린다.

    인원·일수가 주어지면 코드가 계산한 1인당·1일당 목표와, 1인 단위 항목이 인원수 곱해져 합산된다는 규칙,
    항공·숙박을 먼저 정하고 남은 금액을 배분하라는 순서를 함께 알려 LLM 이 곱해진 합계를 고려해 쓰게 한다.
    """
    target = target_budget(plan_type, budget)
    if target is None:
        return "예산 제한이 없으므로 금액 상한 없이 만드세요. "
    text = f"목표 총액은 {target}원 이하입니다. "
    if not people or not days:
        return text
    text += f"(1인당 약 {target // people}원, 1일당 약 {target // days}원) "
    text += (
        f"합계는 코드가 1인 단위 항목의 (1인 가격 x {people}명) 합과 공용 비용의 합으로 계산하고 예산 판정도 코드가 합니다. "
        "항공권과 숙박비가 있으면 조회된 가격을 그대로 쓰고, 목표 총액 안에 들도록 저렴한 장소·식당·이동수단을 고르되 "
        "가격 자체를 비현실적으로 낮추거나 0원으로 쓰지 마세요. 합계를 직접 계산하지 마세요. "
    )
    return text


def _top_cost_text(items: list[ItineraryItem], limit: int = 3) -> str:
    """재조정 프롬프트용: 비용이 큰 항목 상위 몇 개를 '활동(금액원)' 형태로 보여준다. 줄일 곳을 짚어 주기 위함이다."""
    top = sorted(items, key=lambda i: int(i.get("cost_krw") or 0), reverse=True)[:limit]
    return ", ".join(f"{i['activity']}({int(i.get('cost_krw') or 0)}원)" for i in top)


def _lodging_rule(trip: TripRequest, plan_type: str, has_prices: bool) -> str:
    """프롬프트용 숙박 지시. 당일치기(1일)는 숙박을 만들지 않고, 그 외는 2명당 1실(올림) 기준 총 숙박비를 공용 비용으로 넣는다."""
    if trip["days"] <= 1:
        return "당일치기 여행이므로 숙박 항목과 숙박비는 만들지 마세요. "
    nights, rooms = trip["days"] - 1, (trip["people"] + 1) // 2
    if has_prices:
        kind = "가장 저렴한 곳" if plan_type == "가성비형" else "평점이 좋은 중상급 곳"
        source = f"조회된 숙소 중 {kind}을 고르고, 그 1박 객실 요금"
    else:
        source = "해당 지역의 일반적인 1박 객실 요금을 추정해"
    return (
        f"숙박비: {source}을(를) 객실 1박 요금으로 추정하세요. "
        "식사 항목과 합치지 말고, 1일차 저녁에 activity 가 '숙소 체크인: 숙소명'인 별도 항목을 하나만 만들어 "
        "per_person=false, unit_cost_krw=객실 1박 요금(원)으로 쓰세요. "
        f"객실 수({rooms}실, 2명당 1실 올림)와 박 수({nights}박)를 곱한 총 숙박비는 코드가 계산하니 직접 곱하지 마세요. "
        "숙박 비용은 이 항목에만 쓰고 다른 날의 숙박 관련 항목은 0원으로 두세요. "
    )


def _text(content: Any) -> str:
    """메시지 content(문자열 또는 블록 목록)에서 텍스트만 뽑는다."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content or [])


def _item_tools(current: list[ItineraryItem]) -> tuple[Any, Any, dict]:
    """일정 조회·제출 도구 한 쌍과 제출 결과 보관소를 만든다(chat_edit·lookup_routes 공용)."""
    holder: dict = {"submitted": None}

    @tool
    def get_plan_items() -> list[dict]:
        """현재 선택한 안의 일정 항목 목록을 돌려준다."""
        return current

    @tool
    def submit_plan_items(items: list[ItemModel]) -> str:
        """수정을 반영한 전체 일정 항목 목록을 제출한다. 합계나 예산은 판단하지 않는다."""
        holder["submitted"] = [i.model_dump() if isinstance(i, BaseModel) else dict(i) for i in items]
        return "제출되었습니다"

    return get_plan_items, submit_plan_items, holder


def _load_maps() -> Any | None:
    """MOCKDATA_USE=False 일 때 Google Maps MCP 를 연결한다. 연결에 실패하면 경고를 남기고 None(기존 방식)으로 돌아간다."""
    from src.config import google_maps_api_key
    from src.tools import GoogleMapsMCP

    try:
        return GoogleMapsMCP(google_maps_api_key()).start()
    except Exception as exc:
        logger.warning("Google Maps MCP 연결 실패, LLM 추정 방식으로 진행합니다: %s", type(exc).__name__)
        return None


def _is_overseas(model: Any, destination: str) -> bool:
    """목적지가 대한민국 밖(해외)인지 모델에 구조화 출력으로 묻는다. 판정에 실패하면 국내(False)로 본다."""
    try:
        verdict = model.with_structured_output(RegionOutput).invoke(
            f"여행지 '{destination}' 는 대한민국 밖(해외)입니까? 국내면 overseas=false 입니다."
        )
        return bool(getattr(verdict, "overseas", False))
    except (ValidationError, ValueError, TypeError):
        return False


def _destination_profile(model: Any, destination: str) -> RegionOutput:
    """목적지의 해외 여부·현지 통화 코드·날씨 검색어를 모델에 구조화 출력으로 한 번에 묻는다. 실패하면 국내(해외 아님)로 본다."""
    try:
        verdict = model.with_structured_output(RegionOutput).invoke(
            f"여행지 '{destination}' 는 대한민국 밖(해외)입니까? 국내면 overseas=false 입니다. "
            "currency 는 그 여행지의 현지 통화 ISO 4217 코드 3글자(예: SGD, JPY, USD)이고 국내면 KRW 입니다. "
            "weather_query 는 날씨 조회용 도시 검색어로 영어 'City,국가코드2글자' 형식입니다(예: Singapore,SG / Busan,KR)."
        )
        if isinstance(verdict, dict):
            verdict = RegionOutput.model_validate(verdict)
        return verdict if isinstance(verdict, RegionOutput) else RegionOutput(overseas=False)
    except (ValidationError, ValueError, TypeError):
        return RegionOutput(overseas=False)


def _load_images() -> Any | None:
    """MOCKDATA_USE=False 일 때 명소 대표 이미지 조회(위키백과, 키 불필요)를 연결한다."""
    from src.tools import WikipediaImages

    return WikipediaImages()


def _load_rates() -> Any | None:
    """MOCKDATA_USE=False 일 때 환율(Frankfurter, 키 불필요)을 연결한다."""
    from src.tools import FrankfurterRates

    return FrankfurterRates()


def _load_weather() -> Any | None:
    """MOCKDATA_USE=False 일 때 OpenWeatherMap 5일 예보를 연결한다. 키가 없으면 경고만 남기고 None(날씨 없이 진행)."""
    from src.config import ConfigError, openweather_api_key
    from src.tools import OpenWeatherForecast

    try:
        return OpenWeatherForecast(openweather_api_key())
    except ConfigError as exc:
        logger.warning("OpenWeatherMap 키가 없어 날씨 없이 진행합니다: %s", exc)
        return None


def _conditions_rule(fx: dict | None, weather: dict | None) -> str:
    """프롬프트용 환율·날씨 지시. 환율은 현지 통화 가격을 원화로 환산할 때 쓰고, 날씨는 여행 기간 예보가 있을 때만 일정 배치에 반영한다."""
    from src.tools import fx_text, weather_text

    text = ""
    if fx_text(fx):
        text += (
            f"환율: {fx_text(fx)}. 현지 통화({fx['currency']})로 추정한 가격은 원화로 바꾸지 말고 local_unit_price 에 숫자만"
            f"(1인 단위·숙박 항목은 1인 또는 객실 1박 기준, 그 밖의 공용 항목은 전체 금액), local_currency 에 {fx['currency']} 를 쓰세요. "
            "이때 unit_cost_krw 와 cost_krw 는 0으로 두세요. 환율 곱셈은 코드가 합니다. 원화로 조회된 가격은 원화 칸에 그대로 쓰세요. "
        )
    if weather and weather.get("in_range") and weather_text(weather):
        text += (
            f"여행 기간 날씨 예보({weather.get('city', '')}):\n{weather_text(weather)}\n"
            "비 오는 날이나 강수확률이 높은 날은 실내 명소 위주로, 맑은 날은 야외 일정을 배치하세요. "
        )
    return text


def _load_naver() -> Any | None:
    """MOCKDATA_USE=False 일 때 네이버 지역검색·Directions 15 를 연결한다. 키가 없으면 경고만 남기고 None(기존 방식)으로 돌아간다."""
    from src.config import ConfigError, naver_local_keys, naver_maps_keys
    from src.tools import NaverMaps

    try:
        return NaverMaps(naver_local_keys(), naver_maps_keys())
    except ConfigError as exc:
        logger.warning("네이버 API 키가 없어 국내 경로는 LLM 추정으로 진행합니다: %s", exc)
        return None


def _load_prices() -> Any | None:
    """MOCKDATA_USE=False 일 때 마이리얼트립 MCP 를 연결한다. 연결에 실패하면 경고를 남기고 None(기존 방식)으로 돌아간다."""
    from src.config import myrealtrip_mcp_url
    from src.tools import MyRealTripMCP

    try:
        return MyRealTripMCP(myrealtrip_mcp_url()).start()
    except Exception as exc:
        logger.warning("마이리얼트립 MCP 연결 실패, LLM 추정 방식으로 진행합니다: %s", type(exc).__name__)
        return None


def _items_text(items: list[ItineraryItem], people: int | None = None) -> str:
    """프롬프트용 항목 문자열. 1인 단위 항목은 '1인 X원 x N명 = Y원'으로 보여준다."""

    def cost(i: ItineraryItem) -> str:
        if i.get("rooms") and i.get("nights") and i.get("unit_cost_krw"):
            return f"객실 1박 {i['unit_cost_krw']}원 x {i['rooms']}실 x {i['nights']}박 = {i['cost_krw']}원"
        if i.get("per_person") and i.get("unit_cost_krw") and people:
            return f"1인 {i['unit_cost_krw']}원 x {people}명 = {i['cost_krw']}원"
        return f"{i['cost_krw']}원"

    return "\n".join(
        f"{i['day']}일차 {i['time_slot']} {i.get('depart_time', '')}~{i.get('arrive_time', '')}: {i['activity']} ({cost(i)}, {i['transport']}, {i['note']}, "
        f"{i.get('origin', '')}→{i.get('arrival', '')}, {i.get('duration', '')})"
        for i in items
    )


def _to_items(output: Any) -> list[ItineraryItem]:
    """구조화 출력(PlanOutput 또는 dict)을 항목 목록으로 검증·변환한다. 실패 시 ValueError."""
    if isinstance(output, dict):
        output = PlanOutput.model_validate(output)
    if not isinstance(output, PlanOutput):
        raise ValueError("구조화 출력 형식이 올바르지 않습니다")
    if not output.items:
        raise ValueError("일정 항목이 비어 있습니다")
    return [item.model_dump() for item in output.items]


def _call_structured(model: Any, prompt: str) -> tuple[list[ItineraryItem] | None, list[str]]:
    """구조화 출력으로 항목을 받는다. 파싱·필드 오류는 errors 에 기록하고 1회만 다시 시도한다."""
    errors: list[str] = []
    structured = model.with_structured_output(PlanOutput)
    for _ in range(STRUCTURED_ATTEMPTS):
        try:
            return _to_items(structured.invoke(prompt)), errors
        except (ValidationError, ValueError, TypeError) as exc:
            errors.append(f"구조화 출력 오류: {exc}")
    return None, errors


def _call_with_retry(call: Any, attempts: int = LLM_ATTEMPTS, sleep: Any = None) -> Any:
    """call() 을 실행하고, 다시 시도할 만한 오류(시간 초과·쓰로틀링)면 대기 후 최대 attempts 회까지 반복한다.

    시도를 모두 썼거나 재시도 대상이 아닌 오류면 마지막 예외를 그대로 올린다.
    """
    import time

    wait = sleep or time.sleep
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:
            if attempt + 1 >= attempts or not is_retryable(exc):
                raise
            delay = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            logger.warning("LLM 호출 실패(%s), %.0f초 후 재시도 %d/%d", type(exc).__name__, delay, attempt + 2, attempts)
            wait(delay)


def _call_comparison(model: Any, prompt: str, plan_types: list[str]) -> dict[str, str]:
    """안별 차별점 설명을 받는다. 2~3문장·250자 이내가 아닌 안이 있으면 현재 문장 수·글자 수를 알려 1회만 다시 요청한다.

    호출이 어떤 이유로든 실패하면 빈 사전을 돌려준다(요청 전체를 실패시키지 않고 호출한 쪽이 폴백 문장을 쓴다).
    """

    def ask(text: str) -> dict[str, str]:
        try:
            output = model.with_structured_output(PlanComparison).invoke(text)
            if isinstance(output, dict):
                output = PlanComparison.model_validate(output)
            if isinstance(output, PlanComparison):
                return {d.plan_type: d.difference.strip() for d in output.plans}
        except Exception as exc:  # 시간 초과·한도·형식 오류 모두 폴백으로 대신한다
            logger.warning("안별 차별점 생성 실패: %s", type(exc).__name__)
        return {}

    result = ask(prompt)
    bad = {pt: f"{len(split_sentences(result.get(pt)))}문장 {len(result.get(pt, ''))}자" for pt in plan_types if not difference_ok(result.get(pt))}
    if bad:
        retry = ask(f"{prompt}\n이전 응답의 문장 수·글자 수: {bad}. 각 안의 difference 를 2~3문장, 200자 이내로 다시 쓰세요.")
        for pt in bad:
            if retry.get(pt):
                result[pt] = retry[pt]
    return result


def _call_guide(model: Any, prompt: str) -> dict | None:
    """가이드 구조화 출력을 받는다. 형식 오류나 실패면 None (가이드는 선택 사항이라 일정 흐름을 막지 않는다)."""
    try:
        output = model.with_structured_output(GuideOutput).invoke(prompt)
        if isinstance(output, dict):
            output = GuideOutput.model_validate(output)
        return output.model_dump() if isinstance(output, GuideOutput) else None
    except (ValidationError, ValueError, TypeError):
        return None


def build_graph(model: Any | None = None, checkpointer: Any | None = None, maps: Any | None = None, prices: Any | None = None, naver: Any | None = None,
                rates: Any | None = None, weather: Any | None = None, images: Any | None = None):
    """에이전트 StateGraph 를 조립해 컴파일한다. model 을 주면 그 모델을 쓴다(테스트용).

    maps·prices 를 주면 그 MCP 도구를 쓴다. 모델을 직접 만들 때(실서비스)는 MOCKDATA_USE=False 일 때만
    Google Maps MCP(장소·해외 경로)와 마이리얼트립 MCP(항공권·입장권·투어 가격), 네이버 지역검색·Directions 15(국내 경로)를 연결한다.
    """
    from src.config import mockdata_use

    llm = model if model is not None else build_model()
    if maps is None and model is None and not mockdata_use():
        maps = _load_maps()
        prices = prices if prices is not None else _load_prices()
    if naver is None and model is None and not mockdata_use():
        naver = _load_naver()
    if model is None and not mockdata_use():
        rates = rates if rates is not None else _load_rates()
        weather = weather if weather is not None else _load_weather()
        images = images if images is not None else _load_images()
    from src.tools import make_condition_tools

    condition_tools = {t.name: t for t in make_condition_tools(rates, weather)}  # 환율·날씨 Tool (연결된 것만)

    def validate_request(state: GraphState) -> dict:
        """여행지·기간·인원(필수)과 예산(선택) 형식을 확인한다 (범위 상한 없음)."""
        trip = state.get("trip") or {}
        errors = []
        if not str(trip.get("destination", "")).strip():
            errors.append("여행지가 필요합니다")
        if not isinstance(trip.get("days"), int) or isinstance(trip.get("days"), bool) or trip["days"] <= 0:
            errors.append("여행 기간은 1일 이상의 정수여야 합니다")
        people = trip.get("people")
        if not isinstance(people, int) or isinstance(people, bool) or people <= 0:
            errors.append("여행 인원은 1명 이상의 정수여야 합니다")
        budget = trip.get("budget_krw")  # 예산은 선택: 없으면 금액 제한 없이 진행한다
        if budget is not None and (not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0):
            errors.append("예산은 생략하거나 0원보다 큰 정수여야 합니다")
        return {"errors": errors, "trace": ["validate_request"]}

    def fan_out(state: GraphState) -> list[Send]:
        """안 종류별로 병렬 생성(Send). 조사한 장소 목록이 있으면 함께 넘긴다."""
        return [
            Send("generate_plan", {
                "trip": state["trip"], "plan_type": pt,
                "research": state.get("research"), "price_research": state.get("price_research"),
                "fx": state.get("fx"), "weather": state.get("weather"), "place_stats": state.get("place_stats"),
                "price_quotes": state.get("price_quotes"),
            })
            for pt in PLAN_TYPES
        ]

    def route_after_conditions(state: GraphState) -> list[Send] | Literal["research_places", "research_prices"]:
        """MCP 가 있으면 장소 조사, 아니면 안 종류별로 병렬 생성(Send)."""
        if maps:
            return "research_places"
        return "research_prices" if prices else fan_out(state)

    def route_after_validate(state: GraphState) -> list[Send] | Literal["research_conditions", "research_places", "research_prices", "__end__"]:
        """검증 실패면 종료, 환율·날씨 도구가 있으면 먼저 조회, 아니면 장소 조사 또는 안 생성으로 이어진다."""
        if state.get("errors"):
            return END
        if condition_tools:
            return "research_conditions"
        return route_after_conditions(state)

    def research_conditions(state: GraphState) -> dict:
        """환율·날씨 Tool 을 호출한다. 모델이 목적지의 해외 여부·통화·날씨 검색어를 정하고, 값은 도구가 조회한다. 실패해도 일정 생성은 계속된다."""
        trip = state["trip"]
        profile = _destination_profile(llm, trip["destination"])
        update: dict = {"trace": ["research_conditions"], "overseas": profile.overseas}
        results = []
        rate_tool = condition_tools.get("get_exchange_rate")
        currency = profile.currency.strip().upper()
        if rate_tool and profile.overseas and currency and currency != "KRW":
            found = rate_tool.invoke({"currency": currency})
            results.append({"tool": "get_exchange_rate", "result": found})
            if "error" not in found:
                update["fx"] = found
        weather_tool = condition_tools.get("get_weather_forecast")
        if weather_tool and profile.weather_query.strip():
            start, end, _ = trip_dates(trip)
            found = weather_tool.invoke({"city": profile.weather_query.strip(), "start_date": start.isoformat(), "end_date": end.isoformat()})
            results.append({"tool": "get_weather_forecast", "result": {"error": found["error"]} if "error" in found else {
                "city": found["city"], "in_range": found["in_range"], "days": len(found["days"])}})
            if "error" not in found:
                update["weather"] = found
        update["tool_results"] = results
        return update

    def research_places(state: GraphState) -> dict:
        """Google Maps MCP 도구를 에이전트가 직접 호출해 목적지의 실제 맛집·명소를 조사하고, 해외 여부를 판정한다."""
        trip = state["trip"]
        research = None
        try:
            agent = create_agent(
                llm,
                tools=maps.search_tools,
                system_prompt="당신은 여행지 조사 도우미입니다. 도구로 실제 장소를 검색한 뒤, 한국어로 정리한 목록만 답하세요.",
            )
            result = agent.invoke(
                {"messages": [HumanMessage(content=(
                    f"{trip['destination']} 여행을 위해 도구로 실제 맛집 6곳 이상과 명소 6곳 이상을 검색하세요. "
                    "각 장소의 이름, 지역, 평점, 특징을 한 줄씩 목록으로 정리해 답하세요."
                ))]},
                {"recursion_limit": 40},
            )
            research = _text(result["messages"][-1].content).strip() or None
        except Exception as exc:  # 조사 실패는 일정 생성을 막지 않는다(LLM 추정으로 진행)
            logger.warning("장소 조사 실패: %s", type(exc).__name__)
        overseas = state.get("overseas")  # 환율·날씨 조회 노드가 이미 판정했으면 다시 묻지 않는다
        if overseas is None:
            overseas = _is_overseas(llm, trip["destination"])
        stats_snapshot = getattr(maps, "stats_snapshot", None)
        return {
            "trace": ["research_places"],
            "research": research,
            "place_stats": stats_snapshot() if stats_snapshot else {},
            "overseas": overseas,
            "tool_results": [{"tool": "google_maps_search", "result": {"found": bool(research), "overseas": overseas}}],
        }

    def research_prices(state: GraphState) -> dict:
        """마이리얼트립 MCP 도구를 에이전트가 직접 호출해 입장권·투어 가격, 숙소 요금, (서울 출발) 항공권 참고 가격을 조회한다."""
        trip = state["trip"]
        overseas = state.get("overseas")
        update: dict = {"trace": ["research_prices"]}
        if overseas is None:  # Google Maps 조사 노드가 없을 때는 여기서 해외 여부를 판정한다
            overseas = _is_overseas(llm, trip["destination"])
            update["overseas"] = overseas
        depart, back, _ = trip_dates(trip)  # 사용자가 명시한 날짜가 있으면 그 날짜, 없으면 오늘+30일
        nights = max(trip["days"] - 1, 0)
        note = None
        quotes: dict = {}
        make_quote_tools = getattr(prices, "quote_tools", None)  # 요청별 sink 에 항공·숙소 구조화 값을 모으는 도구(없으면 기존 도구)
        try:
            agent = create_agent(
                llm,
                tools=make_quote_tools(quotes) if make_quote_tools else prices.price_tools,
                system_prompt=(
                    "당신은 여행 가격 조사 도우미입니다. 도구로 실제 가격을 조회한 뒤 한국어 목록만 답하세요. "
                    "도구가 실패하거나 결과가 없으면 '조회 실패'라고 쓰고 가격을 지어내지 마세요. "
                    "항공권·입장권·투어 가격은 1인 기준으로 정리하고(항공은 passengers=1), 가격이 실시간으로 변동된다고 함께 적으세요."
                ),
            )
            origin = trip_origin(trip)[0]
            flight_hint = (
                f"항공권: 출발지는 {origin}입니다. 출발지가 한국이고 목적지가 해외면 searchInternationalFlights"
                f"(tripType=ROUND_TRIP, origin=출발지 공항 IATA 코드, destination=목적지 공항 IATA 코드, departDate={depart}, "
                f"returnDate={back}, 성인 1명, maxResults=3)로, 출발지와 목적지가 모두 한국이고 비행기 이동이 현실적이면 "
                f"searchDomesticFlights(tripType=ROUND_TRIP, origin=출발지 공항 IATA 코드, destination=목적지 공항 IATA 코드, "
                f"departDate={depart}, returnDate={back}, 성인 1명)로 조회하세요. 출발지가 해외이거나 지상 이동이 자연스러우면 "
                "항공권은 조회하지 않고 생략하세요. 가는 편·오는 편의 날짜, 출발·도착 시각, 항공사·편명, 1인 왕복 요금을 그대로 적으세요. "
            )
            result = agent.invoke(
                {"messages": [HumanMessage(content=(
                    f"{trip['destination']} {trip['days']}일 여행의 참고 가격을 조회하세요. "
                    f"입장권·투어: searchTnas 로 '{trip['destination']} 입장권', '{trip['destination']} 투어' 등을 검색해 "
                    "대표 5건 이상의 이름과 가격을 정리하세요. " + flight_hint +
                    (
                        f"숙소: searchStays(keyword={trip['destination']}, checkIn={depart}, checkOut={back}, adultCount={trip['people']}, "
                        f"isDomestic={'false' if overseas else 'true'}, size=5)로 숙소 5곳의 이름·평점·1박 객실 요금을 조회하세요. "
                        if nights >= 1 else ""
                    ) +
                    "답은 '입장권·투어', '숙소', '항공권' 소제목 아래 한 줄씩 쓰세요(입장권·항공권은 1인 가격, 숙소는 1박 객실 요금)."
                ))]},
                {"recursion_limit": 40},
            )
            note = _text(result["messages"][-1].content).strip() or None
        except Exception as exc:  # 가격 조사 실패는 일정 생성을 막지 않는다(LLM 추정으로 진행)
            logger.warning("가격 조사 실패: %s", type(exc).__name__)
        update["price_research"] = note
        update["price_quotes"] = quotes
        update["tool_results"] = [{"tool": "myrealtrip_prices", "result": {"found": bool(note), "overseas": overseas}}]
        return update

    def generate_plan(payload: dict) -> dict:
        """안 1개를 생성한다. 실패하면 이 안은 plans 에 남기지 않는다."""
        trip, plan_type = payload["trip"], payload["plan_type"]
        grounding = (
            f"아래는 Google Maps 로 조회한 실제 장소입니다. 식당과 명소는 이 목록에서 고르세요.\n{payload['research']}\n"
            if payload.get("research") else ""
        )
        if payload.get("price_research"):
            grounding += (
                "아래는 마이리얼트립에서 조회한 참고 가격입니다(항공권·입장권·투어는 1인 기준). 해당 항목의 1인 가격은 "
                f"이 가격을 근거로 하세요.\n{payload['price_research']}\n"
            )
        prompt = (
            f"{grounding}{trip['destination']} {trip['days']}일, 여행 인원 {trip['people']}명 여행 일정을 "
            f"{PLAN_STYLE[plan_type]} 만드세요. 출발지는 {trip_origin(trip)[0]}입니다. "
            f"여행 기간은 {trip_dates(trip)[0]}~{trip_dates(trip)[1]}입니다. "
            f"{_budget_phrase(plan_type, trip.get('budget_krw'), trip['people'], trip['days'])}"
            "각 일차마다 아침/오후/저녁 항목을 두고, 비용은 원화(KRW)로 추정하세요. "
            "각 항목에는 출발지(origin), 도착지(arrival), 이동 소요시간(duration, 예: 약 30분)도 채우세요. "
            f"{COST_RULES}{TIME_RULE}{NOTE_RULE}{CATEGORY_RULE}{_flight_rule(trip_origin(trip)[0], trip['destination'])}"
            f"{_lodging_rule(trip, plan_type, bool(payload.get('price_research')))}"
            f"{_conditions_rule(payload.get('fx'), payload.get('weather'))}"
        )
        try:
            items, errors = _call_with_retry(lambda: _call_structured(llm, prompt))
        except Exception as exc:  # 재시도까지 실패한 안만 빼고 나머지 안으로 계속한다
            if not is_retryable(exc):
                raise
            logger.error("%s 안 생성 실패, 이 안은 제외합니다: %s", plan_type, type(exc).__name__)
            items, errors = None, [f"{plan_type} 안 생성 실패: {type(exc).__name__}"]
        if items is not None:
            items = finalize_items(items, trip["people"], trip["days"], payload.get("fx"), payload.get("place_stats"), payload.get("price_quotes"), plan_type)
        update: dict = {"trace": ["generate_plan"], "errors": errors}
        if items is not None:
            update["plans"] = {plan_type: {"plan_type": plan_type, "attempts": [items]}}
        return update

    def check_budget_node(state: GraphState) -> dict:
        """모든 안의 합계와 판정을 코드로 계산해 contexts 에 남긴다."""
        budget = state["trip"].get("budget_krw")
        results = []
        for pt, plan in state.get("plans", {}).items():
            res = check_budget(pt, plan["attempts"][-1], budget)
            results.append({"tool": "check_budget", "plan_type": pt, "result": res})
        return {"trace": ["check_budget"], "tool_results": results}

    def route_after_check(state: GraphState) -> list[Send] | Literal["pick_closest", "collect_plans"]:
        """초과 + 재조정 3회 미만인 안은 readjust_plan(Send), 없으면 근접 안 선택 또는 수집."""
        budget = state["trip"].get("budget_krw")
        plans = state.get("plans", {})
        sends = [
            Send("readjust_plan", {"trip": state["trip"], "plan_type": pt, "attempts": plan["attempts"], "fx": state.get("fx"), "place_stats": state.get("place_stats"), "price_quotes": state.get("price_quotes")})
            for pt, plan in plans.items()
            if needs_readjust(pt, plan["attempts"], budget)
        ]
        if sends:
            return sends
        exhausted = any(not check_budget(pt, p["attempts"][-1], budget)["within"] for pt, p in plans.items())
        return "pick_closest" if exhausted else "collect_plans"

    def readjust_plan(payload: dict) -> dict:
        """초과 금액을 알려 항목 금액을 줄인 안을 다시 생성한다."""
        trip, plan_type, attempts = payload["trip"], payload["plan_type"], payload["attempts"]
        last = attempts[-1]
        res = check_budget(plan_type, last, trip.get("budget_krw"))
        prompt = (
            f"{trip['destination']} {trip['days']}일, {trip['people']}명 {plan_type} 일정의 합계 {res['total']}원이 목표 "
            f"{target_budget(plan_type, trip.get('budget_krw'))}원을 {res['overage']}원 초과했습니다. "
            "목표 이하가 되도록 더 저렴한 장소·식당·이동수단으로 바꾸고 그 항목의 금액을 다시 추정하세요. per_person 항목은 unit_cost_krw(1인 가격), "
            "숙박 항목은 unit_cost_krw(객실 1박 요금)를 쓰세요(인원수·객실 수·박 수는 코드가 곱합니다). "
            "가격을 비현실적으로 낮추거나 0원으로 쓰지 마세요. 최소 단가보다 낮은 금액은 코드가 올립니다. "
            f"비용이 큰 항목: {_top_cost_text(last)}. 이런 항목부터 검토하세요. "
            f"{NOTE_RULE}{CATEGORY_RULE}\n{_items_text(last, trip['people'])}"
        )
        items, errors = _call_structured(llm, prompt)
        if items is not None:
            items = finalize_items(items, trip["people"], trip["days"], payload.get("fx"), payload.get("place_stats"), payload.get("price_quotes"), plan_type)
        if items is None:
            # 실패한 재조정도 한 번의 시도로 센다(같은 안을 남겨 무한 루프를 막는다)
            items = last
        return {
            "trace": ["readjust_plan"],
            "errors": errors,
            "plans": {plan_type: {"plan_type": plan_type, "attempts": attempts + [items]}},
        }

    def pick_closest(state: GraphState) -> dict:
        """예산을 못 맞춘 안마다 초과 금액이 가장 작은 시도를 고른다(결과는 contexts 로만 남김)."""
        budget = state["trip"].get("budget_krw")
        results = []
        for pt, plan in state.get("plans", {}).items():
            if not check_budget(pt, plan["attempts"][-1], budget)["within"]:
                best = closest_attempt(plan["attempts"], pt, budget)
                results.append({"tool": "closest_attempt", "plan_type": pt, "result": check_budget(pt, best, budget)})
        return {"trace": ["pick_closest"], "tool_results": results}

    def collect_plans(state: GraphState) -> dict:
        """3안 결과 모음 지점 (plans 는 리듀서가 이미 합쳤다)."""
        return {"trace": ["collect_plans"]}

    def route_after_collect(state: GraphState) -> Literal["compare_plans", "__end__"]:
        """생성된 안이 하나도 없으면 종료(오류 응답), 있으면 안별 비교 후 선택 대기."""
        return "compare_plans" if state.get("plans") else END

    def compare_plans(state: GraphState) -> dict:
        """3안을 서로 비교해 안마다 '다른 안 대비 차별점'(300~500자)을 만든다. 안이 2개 미만이면 건너뛰고, 실패하면 수치 기반 문장으로 대신한다."""
        trip, plans = state["trip"], state.get("plans", {})
        finals = {pt: final_items(pt, plans[pt]["attempts"], trip.get("budget_krw")) for pt in PLAN_TYPES if pt in plans}
        update: dict = {"trace": ["compare_plans"]}
        if len(finals) < 2:
            return update
        blocks = "\n\n".join(
            f"[{pt}] 합계 {check_budget(pt, its, trip.get('budget_krw'))['total']}원\n{_items_text(its, trip['people'])}"
            for pt, its in finals.items()
        )
        prompt = (
            f"{trip['destination']} {trip['days']}일, {trip['people']}명 여행의 {len(finals)}가지 안(" + ", ".join(finals) + ")입니다. "
            "안별 차별점 비교: 각 안이 다른 안과 비교해 무엇이 다른지(명소·식사·이동 비중, 대표 일정, 숙소·항공)를 "
            "안마다 2~3문장(200자 이내) 한국어로 간결하게 설명하세요. 금액·합계·비율 같은 숫자는 쓰지 마세요"
            "(합계와 예산 대비는 코드가 계산해 화면에 따로 보여 줍니다). 대소 비교는 '더 저렴하다', '가장 비싸다'처럼 말로만 쓰세요. "
            "아래에 주어진 사실만 쓰고 없는 장소를 지어내지 마세요. "
            f"plan_type 은 {', '.join(finals)} 중 하나입니다.\n\n{blocks}"
        )
        raw = _call_comparison(llm, prompt, list(finals))
        comparison = {pt: fit_difference(raw.get(pt), fallback_difference(pt, finals)) for pt in finals}
        update["comparison"] = comparison
        update["tool_results"] = [{"tool": "compare_plans", "result": {pt: len(text) for pt, text in comparison.items()}}]
        return update

    def await_selection(state: GraphState) -> dict:
        """3안을 보여주고 사용자 선택을 interrupt 로 기다린다."""
        payload = {
            "stage": "select",
            "plainAnswer": render_plain_answer(
                state["trip"], state["plans"], comparison=state.get("comparison"), place_stats=state.get("place_stats")
            ),
        }
        value = interrupt(payload)
        plan = value.get("plan") if isinstance(value, dict) and value.get("type") == "select" else None
        if plan not in state["plans"]:
            return {"trace": ["await_selection"], "errors": [f"알 수 없는 안: {plan}"], "selected": None}
        items = final_items(plan, state["plans"][plan]["attempts"], state["trip"].get("budget_krw"))
        return {"trace": ["await_selection"], "selected": plan, "selected_items": items, "selected_attempts": [items]}

    def route_after_selection(state: GraphState) -> list[str] | Literal["await_selection"]:
        """선택이 유효하면 경로 조회(해외는 Google, 국내는 네이버)와 가이드 생성을 병렬로 시작하고, 아니면 다시 선택 대기."""
        if not state.get("selected"):
            return "await_selection"
        if state.get("overseas"):
            return ["lookup_routes", "generate_guide"] if maps else ["generate_guide"]
        return ["lookup_domestic_routes", "generate_guide"] if naver else ["generate_guide"]

    def lookup_domestic_routes(state: GraphState) -> dict:
        """국내 여행지만: 모델이 항목별 장소 검색어를 정하고, 코드가 네이버 지역검색·Directions 15 로 이동 경로·소요시간·비용을 채운다."""
        current = state["selected_items"]
        update: dict = {"trace": ["lookup_domestic_routes"]}
        lines = "\n".join(
            f"{i}: {it['day']}일차 {it['time_slot']} {it['activity']} (분류 {it.get('category') or '-'})" for i, it in enumerate(current)
        )
        queries: dict[int, str] = {}
        try:
            output = llm.with_structured_output(PlaceQueries).invoke(
                f"여행지 {state['trip']['destination']} 일정 항목입니다. 항목마다 네이버 지도에서 찾을 장소의 검색어를 정하세요. "
                "검색어는 지명 또는 상호에 지역명을 붙여 쓰세요(예: '해운대해수욕장 부산'). 항공·기차·공항, 이동 자체, 숙소 체크인, "
                "특정 장소가 없는 항목은 query 를 빈 문자열로 쓰세요. index 는 아래 번호를 그대로 쓰세요.\n" + lines
            )
            if isinstance(output, dict):
                output = PlaceQueries.model_validate(output)
            queries = {q.index: q.query.strip() for q in output.items if 0 <= q.index < len(current) and q.query.strip()}
        except Exception as exc:  # 검색어를 못 얻으면 기존(LLM 추정) 값을 그대로 쓴다
            logger.warning("국내 경로용 검색어 생성 실패: %s", type(exc).__name__)
        if queries:
            filled_items, filled = apply_domestic_routes(current, queries, naver, state["trip"]["destination"])
            if filled:
                update["selected_items"] = filled_items
                update["selected_attempts"] = [filled_items]
            update["tool_results"] = [{"tool": "naver_routes", "result": {"count": filled}}]
        return update

    def lookup_routes(state: GraphState) -> dict:
        """해외 여행지만: 에이전트가 경로 도구(maps_directions 등)로 이동 구간의 출발지·도착지·소요시간을 실제 조회값으로 채운다."""
        current = state["selected_items"]
        get_items, submit_items, holder = _item_tools(current)
        update: dict = {"trace": ["lookup_routes"]}
        try:
            agent = create_agent(
                llm,
                tools=[get_items, submit_items, *maps.route_tools],
                system_prompt=(
                    "당신은 이동 경로 조회 도우미입니다. get_plan_items 로 일정을 확인하고, 같은 날 연속한 항목 사이의 이동을 "
                    "경로 도구로 조회해 각 항목의 origin(출발지)·arrival(도착지)·duration(소요시간)과 transport 를 조회 결과로 채워 "
                    "submit_plan_items 로 전체 항목을 제출하세요. 활동과 비용(cost_krw)은 바꾸지 마세요.\n"
                    "경로 도구 결과 읽는 법: 경로는 routes[0].legs[0] 이고 duration(초)이 전체 소요시간, steps 가 구간별 안내입니다. "
                    "'Head northwest' 같은 안내는 걷기 시작할 때의 나침반 방향일 뿐이며 버스·지하철의 종점역이 아닙니다. "
                    "대중교통의 방향(종점)은 'Tram towards VivoCity Station'·'Subway towards ...' 처럼 transitDetails 가 붙은 "
                    "구간의 'towards' 뒤에 있습니다. transport 에는 걷기를 빼고 탄 수단(예: 트램+지하철)을 쓰고, "
                    "duration 에는 전체 소요시간을 '약 28분' 형태로 쓰세요. 도구가 실패하거나 결과가 없으면 값을 지어내지 말고 "
                    "기존 값을 그대로 두세요."
                ),
            )
            agent.invoke(
                {"messages": [HumanMessage(content=f"{state['trip']['destination']} 일정의 이동 경로와 소요시간을 조회해 채워 주세요.")]},
                {"recursion_limit": 60},
            )
        except Exception as exc:  # 조회 실패는 기존(LLM 추정) 값을 그대로 쓴다
            logger.warning("경로 조회 실패: %s", type(exc).__name__)
        submitted = holder["submitted"]
        if submitted and len(submitted) == len(current):
            for old, new in zip(current, submitted):
                new["cost_krw"] = old["cost_krw"]  # 비용은 경로 조회로 바뀌지 않는다
                new["per_person"] = old.get("per_person", False)
                new["unit_cost_krw"] = old.get("unit_cost_krw", 0)
                for key in ("rooms", "nights"):  # 코드가 계산한 숙박 객실·박 수도 유지한다
                    if key in old:
                        new[key] = old[key]
                new["depart_time"] = old.get("depart_time", "")  # 시각은 계획값을 유지한다
                new["arrive_time"] = old.get("arrive_time", "")
                new["category"] = new.get("category") or old.get("category", "")
                new["note"] = clip_note(new.get("note"))  # 경로 조회가 비고를 길게 다시 써도 상한을 지킨다
            update["selected_items"] = submitted
            update["selected_attempts"] = [submitted]
            update["tool_results"] = [{"tool": "google_maps_routes", "result": {"count": len(submitted)}}]
        return update

    def generate_guide(state: GraphState) -> dict:
        """선택한 안의 일정을 바탕으로 맛집·명소 가이드와 준비물·목적지 유의 정보를 1회 생성한다."""
        trip = state["trip"]
        grounding = (
            f"아래는 Google Maps 로 조회한 실제 장소입니다. places 는 이 목록과 일정에서 고르세요.\n{state['research']}\n"
            if state.get("research") else ""
        )
        if state.get("price_research"):
            grounding += f"아래는 마이리얼트립 참고 가격입니다. 준비물·유의 사항에 필요하면 참고하세요.\n{state['price_research']}\n"
        prompt = (
            f"{grounding}{trip['destination']} {trip['days']}일 {state['selected']} 일정에 맞는 여행 가이드를 만드세요. "
            "places: 일정에 나온 곳과 근처 추천 맛집(kind=맛집)·명소(kind=명소), 각각 지역·설명·팁. "
            "packing: 준비물. cautions: 여행 중 유의 사항. destination_notes: 목적지에 대한 사전 유의 정보"
            "(입국·비자, 전압·플러그, 교통·결제 수단, 치안·날씨, 문화·에티켓, 팁 문화 등 이 목적지에 해당하는 것). "
            "분량: places 는 맛집 4개·명소 4개, packing 8개 이내, cautions 4개, destination_notes 5개이며 "
            "각 설명은 한두 문장으로 짧게 쓰세요. "
            "확인되지 않은 최신 정보는 단정하지 말고 출발 전 확인을 권하세요.\n"
            f"{_items_text(state['selected_items'])}"
        )
        guide = _call_guide(llm, prompt)
        update: dict = {"trace": ["generate_guide"], "guide": guide}
        if images and guide:  # 명소마다 대표 이미지를 붙인다(실패해도 이미지만 빠지고 가이드는 그대로)
            from src.tools import attach_place_images

            try:
                guide, found = attach_place_images(guide, images, trip["destination"])
                update["guide"] = guide
                total = sum(1 for p in guide.get("places", []) if p.get("kind") == "명소")
                update["tool_results"] = [{"tool": "wikipedia_images", "result": {"found": found, "total": total}}]
            except Exception as exc:
                logger.warning("명소 이미지 붙이기 실패: %s", type(exc).__name__)
        return update

    def await_chat(state: GraphState) -> dict:
        """현재 선택 안에 대한 대화 입력(또는 종료)을 interrupt 로 기다린다."""
        text = render_plain_answer(state["trip"], state["plans"], state["selected"], state["selected_items"])
        value = interrupt({"stage": "chat", "plainAnswer": text})
        kind = value.get("type") if isinstance(value, dict) else None
        if kind == "chat" and str(value.get("text", "")).strip():
            return {"trace": ["await_chat"], "next_action": "chat", "chat_history": [HumanMessage(content=str(value["text"]))]}
        return {"trace": ["await_chat"], "next_action": "done"}

    def route_after_chat_wait(state: GraphState) -> Literal["chat_edit", "__end__"]:
        """chat 이면 수정, done 이면 종료."""
        return "chat_edit" if state.get("next_action") == "chat" else END

    def chat_edit(state: GraphState) -> dict:
        """create_agent 에이전트가 get_plan_items/submit_plan_items 로 선택 안을 수정한다."""
        current = state["selected_items"]
        get_plan_items, submit_plan_items, holder = _item_tools(current)

        agent = create_agent(
            llm,
            tools=[get_plan_items, submit_plan_items],
            system_prompt=(
                "당신은 여행 일정 수정 도우미입니다. get_plan_items 로 현재 일정을 확인하고, 사용자가 요청한 부분만 고쳐 "
                "submit_plan_items 로 전체 항목(출발지·도착지·소요시간·출발시간·도착시간·per_person·unit_cost_krw 포함)을 제출하세요. 비용은 원화로 추정하고, "
                "입장권·식사·항공권 같은 1인 단위 비용은 per_person=true 와 unit_cost_krw(1인 가격)로, 숙박 체크인 항목은 unit_cost_krw(객실 1박 요금)로 쓰세요"
                "(인원수·객실 수·박 수는 코드가 곱합니다). 0원으로 낮추지 마세요. "
                "각 항목의 category(식당/명소/이동/숙박/기타)도 유지하세요. "
                f"비고(note)는 {NOTE_MAX_CHARS}자 이내로 짧게 쓰세요. 합계나 예산 판단은 말하지 마세요."
            ),
        )
        request = state["chat_history"][-1].content if state.get("chat_history") else ""
        agent.invoke({"messages": [HumanMessage(content=str(request))]})
        submitted = holder["submitted"]
        if submitted:
            submitted = finalize_items(submitted, state["trip"].get("people"), state["trip"].get("days"), state.get("fx"), state.get("place_stats"), state.get("price_quotes"), state.get("selected"))
        update: dict = {"trace": ["chat_edit"]}
        if submitted:
            update["selected_items"] = submitted
            update["selected_attempts"] = [submitted]  # 수정 1회마다 재조정 횟수를 새로 센다
            update["tool_results"] = [{"tool": "submit_plan_items", "result": {"count": len(submitted)}}]
        else:
            update["selected_attempts"] = [current]
            update["errors"] = ["수정 결과가 제출되지 않았습니다"]
        return update

    def check_budget_selected(state: GraphState) -> dict:
        """수정된 선택 안의 합계·판정을 계산한다. 재조정이 끝났는데 초과면 가장 근접한 시도를 택한다."""
        budget = state["trip"].get("budget_krw")
        pt = state["selected"]
        attempts = state["selected_attempts"]
        res = check_budget(pt, attempts[-1], budget)
        update: dict = {
            "trace": ["check_budget_selected"],
            "tool_results": [{"tool": "check_budget", "plan_type": pt, "result": res}],
        }
        if res["within"]:
            update["selected_items"] = attempts[-1]
        elif not needs_readjust(pt, attempts, budget):
            update["selected_items"] = closest_attempt(attempts, pt, budget)
        return update

    def route_after_check_selected(state: GraphState) -> Literal["readjust_selected", "render_answer"]:
        """초과이고 재조정 3회 미만이면 재조정, 아니면 렌더."""
        if needs_readjust(state["selected"], state["selected_attempts"], state["trip"].get("budget_krw")):
            return "readjust_selected"
        return "render_answer"

    def readjust_selected(state: GraphState) -> dict:
        """선택 안이 초과하면 초과 금액을 알려 항목 금액을 줄여 다시 생성한다."""
        trip, pt, attempts = state["trip"], state["selected"], state["selected_attempts"]
        last = attempts[-1]
        res = check_budget(pt, last, trip.get("budget_krw"))
        prompt = (
            f"{trip['destination']} {trip['people']}명 {pt} 일정의 합계 {res['total']}원이 목표 "
            f"{target_budget(pt, trip.get('budget_krw'))}원을 {res['overage']}원 초과했습니다. "
            "사용자의 수정 요청 내용은 유지하며 다른 항목을 더 저렴한 선택으로 바꾸고 그 금액을 다시 추정하세요. per_person 항목은 unit_cost_krw(1인 가격), "
            "숙박 항목은 unit_cost_krw(객실 1박 요금)를 쓰세요(인원수·객실 수·박 수는 코드가 곱합니다). "
            "가격을 비현실적으로 낮추거나 0원으로 쓰지 마세요. 최소 단가보다 낮은 금액은 코드가 올립니다. "
            f"비용이 큰 항목: {_top_cost_text(last)}. 이런 항목부터 검토하세요. "
            f"{NOTE_RULE}{CATEGORY_RULE}\n{_items_text(last, trip['people'])}"
        )
        items, errors = _call_structured(llm, prompt)
        if items is not None:
            items = finalize_items(items, trip["people"], trip["days"], state.get("fx"), state.get("place_stats"), state.get("price_quotes"), state.get("selected"))
        return {
            "trace": ["readjust_selected"],
            "errors": errors,
            "selected_attempts": attempts + [items if items is not None else last],
        }

    def render_answer(state: GraphState) -> dict:
        """plainAnswer 와 HTML 리포트를 만들어 대화 이력에 남긴다 (HTML 은 상태에 저장하지 않음)."""
        answer = build_answer(state)
        return {"trace": ["render_answer"], "chat_history": [AIMessage(content=answer["plainAnswer"])]}

    g = StateGraph(GraphState)
    g.add_node("validate_request", validate_request)
    g.add_node("generate_plan", generate_plan)  # 재시도는 노드 안(_call_with_retry)에서 하고, 끝내 실패한 안은 제외한다
    g.add_node("check_budget", check_budget_node)
    g.add_node("readjust_plan", readjust_plan, retry_policy=LLM_RETRY)
    g.add_node("pick_closest", pick_closest)
    g.add_node("collect_plans", collect_plans)
    g.add_node("compare_plans", compare_plans)
    g.add_node("await_selection", await_selection)
    g.add_node("await_chat", await_chat)
    g.add_node("chat_edit", chat_edit, retry_policy=LLM_RETRY)
    g.add_node("check_budget_selected", check_budget_selected)
    g.add_node("readjust_selected", readjust_selected, retry_policy=LLM_RETRY)
    g.add_node("generate_guide", generate_guide, retry_policy=LLM_RETRY)
    if maps:
        g.add_node("research_places", research_places)
        g.add_node("lookup_routes", lookup_routes)
    if naver:
        g.add_node("lookup_domestic_routes", lookup_domestic_routes)
        g.add_edge("lookup_domestic_routes", "render_answer")  # 가이드와 병렬 실행 후 같은 단계에서 합류
    if prices:
        g.add_node("research_prices", research_prices)
    g.add_node("render_answer", render_answer)

    g.add_edge(START, "validate_request")
    entry_targets = ["generate_plan"] + (["research_places"] if maps else []) + (["research_prices"] if prices else [])
    g.add_conditional_edges("validate_request", route_after_validate, entry_targets + [END] + (["research_conditions"] if condition_tools else []))
    if condition_tools:
        g.add_node("research_conditions", research_conditions)
        g.add_conditional_edges("research_conditions", route_after_conditions, entry_targets)
    if maps:
        if prices:
            g.add_edge("research_places", "research_prices")
        else:
            g.add_conditional_edges("research_places", fan_out, ["generate_plan"])
        g.add_edge("lookup_routes", "render_answer")  # 가이드와 병렬 실행 후 같은 단계에서 합류
    if prices:
        g.add_conditional_edges("research_prices", fan_out, ["generate_plan"])
    g.add_edge("generate_plan", "check_budget")
    g.add_conditional_edges("check_budget", route_after_check, ["readjust_plan", "pick_closest", "collect_plans"])
    g.add_edge("readjust_plan", "check_budget")
    g.add_edge("pick_closest", "collect_plans")
    g.add_conditional_edges("collect_plans", route_after_collect, ["compare_plans", END])
    g.add_edge("compare_plans", "await_selection")
    g.add_conditional_edges(
        "await_selection", route_after_selection,
        ["await_selection", "generate_guide"] + (["lookup_routes"] if maps else []) + (["lookup_domestic_routes"] if naver else []),
    )
    g.add_edge("generate_guide", "render_answer")
    g.add_edge("render_answer", "await_chat")
    g.add_conditional_edges("await_chat", route_after_chat_wait, ["chat_edit", END])
    g.add_edge("chat_edit", "check_budget_selected")
    g.add_conditional_edges("check_budget_selected", route_after_check_selected, ["readjust_selected", "render_answer"])
    g.add_edge("readjust_selected", "check_budget_selected")
    return g.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())


def build_answer(state: dict) -> dict:
    """상태에서 plainAnswer 와 html 을 계산한다. 선택 전이면 3안 요약과 빈 html."""
    trip = state["trip"]
    selected = state.get("selected")
    if not selected:
        plain = render_plain_answer(trip, state.get("plans", {}), comparison=state.get("comparison"), place_stats=state.get("place_stats"))
        missing = [pt for pt in PLAN_TYPES if pt not in state.get("plans", {})]
        if missing:  # 시간 초과 등으로 빠진 안은 조용히 숨기지 않고 알린다
            plain = f"안내: {'·'.join(missing)} 안은 생성하지 못해 제외했습니다. 다시 요청하면 전체 안을 받을 수 있습니다.\n{plain}"
        return {"plainAnswer": plain, "html": ""}
    items = state["selected_items"]
    origin, is_default = trip_origin(trip)
    origin_text = f"{origin} (입력이 없어 기본값)" if is_default else origin
    return {
        "plainAnswer": render_plain_answer(trip, state["plans"], selected, items),
        "html": render_report_html(
            selected, items, trip.get("budget_krw"), state.get("guide"), state.get("price_research"),
            trip.get("people"), origin_text, period_label(trip), trip_dates(trip)[0],
            state.get("fx"), state.get("weather"),
        ),
    }
