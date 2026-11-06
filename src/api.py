"""FastAPI 앱: POST /query 로 LangGraph 에이전트를 구동하고 interrupt 를 재개한다."""
import json
import logging
import os
import queue
import re
import threading
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel

from src.agent import PLAN_TYPES, build_answer, build_graph, build_model, extract_trip
from src.tools import CONFLICT_NOTICE, classify_request, has_conflict, merge_trip, restates_trip

logger = logging.getLogger("uvicorn.error")

WEB_INDEX = Path(__file__).resolve().parent.parent / "web" / "index.html"
DONE_WORDS = {"종료", "끝", "그만", "완료", "done", "끝내기", "종료할게", "끝낼게"}
_STOPWORDS = {"예산", "여행", "계획", "일정", "총", "약", "으로", "로", "정확한", "정확", "실시간", "최저가", "대략"}


class QueryRequest(BaseModel):
    """요청 본문: 자연어 한 줄과 선택적 session_id."""

    question: str | None = None
    session_id: str | None = None


class ApiError(Exception):
    """HTTP 오류로 바꿀 API 오류 (code, message, status)."""

    def __init__(self, status: int, code: str, message: str, extra: dict | None = None):
        self.status, self.code, self.message, self.extra = status, code, message, extra or {}


def error_response(status: int, code: str, message: str, extra: dict | None = None) -> JSONResponse:
    """{"error": {"code", "message"}} 형식의 오류 응답을 만든다. extra 는 error 안에 덧붙인다(예: session_id, missing)."""
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, **(extra or {})}})


_PEOPLE_WORDS = {
    "혼자": 1, "혼행": 1, "한 명": 1, "두 명": 2, "세 명": 3, "네 명": 4,
    "부부": 2, "커플": 2, "둘이": 2, "셋이": 3, "넷이": 4,
}


def parse_people(question: str) -> int | None:
    """여행 인원(성인 기준)을 규칙으로 뽑는다. '2명', '3인', '혼자', '부부', '둘이' 등. '1인분'·'2인실'은 인원으로 읽지 않는다."""
    m = re.search(r"(\d+)\s*(?:명|인)(?![분실])", question)
    if m and int(m.group(1)) > 0:
        return int(m.group(1))
    for word, count in _PEOPLE_WORDS.items():
        if word in question:
            return count
    return None


_PLACE = r"[가-힣A-Za-z]+"
_COUNTRIES = (
    "중국", "일본", "미국", "영국", "프랑스", "이탈리아", "스페인", "독일", "태국", "베트남", "대만", "홍콩", "싱가포르", "필리핀",
    "인도네시아", "말레이시아", "호주", "캐나다", "튀르키예", "터키", "스위스", "체코", "오스트리아", "포르투갈", "그리스", "이집트",
    "인도", "몽골", "라오스", "캄보디아", "뉴질랜드", "멕시코", "러시아", "네덜란드", "헝가리", "크로아티아", "노르웨이", "핀란드",
)  # '중국 장가계'처럼 나라 이름이 앞에 붙으면 도시를 여행지로 본다
_FILLED_TRIP = {"destination": "-", "days": 1, "people": 1}  # classify_request 의 누락 검사를 통과시키는 자리표시 값(원문 규칙 검사만 먼저 하려고 쓴다)
_COUNTRY_RE = "|".join(_COUNTRIES)
_FILLER = {"여행", "가족여행", "해외여행", "국내여행", "일정", "계획", "여행지"}  # 여행지와 날짜 사이에 끼어도 되는 말
_NOT_PLACE = _STOPWORDS | {"가족여행", "해외여행", "국내여행", "여행지", "가족", "친구", "예정", "예정이야", "예정이에요", "이번", "다음", "내일", "올해"}


_DATE_A = re.compile(r"(\d{4})\s*[-./년]\s*(\d{1,2})\s*[-./월]\s*(\d{1,2})\s*일?")  # 2026-12-20, 2026.12.20, 2026년 12월 20일
_DATE_B = re.compile(r"(?:(\d{4})\s*년\s*)?(\d{1,2})\s*[월/]\s*(\d{1,2})\s*일?")  # 12월 20일, 12/20 (연도 없는 표기는 월·/ 만 허용)
_RANGE_SEP = re.compile(r"\s*(?:~|–|-|부터)\s*")
_DAY_ONLY = re.compile(r"(\d{1,2})\s*일?(?![\d월/])")
_DATE_MARK = "@@"


def _match_date(text: str, pos: int) -> tuple[int | None, int, int, int] | None:
    """text[pos:] 가 날짜 표기로 시작하면 (연도 또는 None, 월, 일, 끝 위치)를 돌려준다."""
    m = _DATE_A.match(text, pos) or _DATE_B.match(text, pos)
    if not m:
        return None
    year = int(m.group(1)) if m.group(1) else None
    return year, int(m.group(2)), int(m.group(3)), m.end()


def _resolve_date(year: int | None, month: int, day: int, today: date) -> date | None:
    """연도가 없으면 오늘 이후 가장 가까운 날짜로 본다. 존재하지 않는 날짜면 None."""
    try:
        if year is not None:
            return date(year, month, day)
        candidate = date(today.year, month, day)
        return candidate if candidate >= today else date(today.year + 1, month, day)
    except ValueError:
        return None


def parse_dates(question: str, today: date | None = None) -> tuple[date | None, date | None, str]:
    """문장에서 여행 시작일·종료일을 뽑고, 날짜 부분을 '@@' 로 바꾼 문장을 함께 돌려준다.

    단일 날짜와 범위('12월 20일부터 23일까지', '12/20~12/23', '2026-12-20~2026-12-22')를 읽는다.
    이미 지난 날짜나 존재하지 않는 날짜는 날짜로 쓰지 않는다(시작일 None). 날짜 표기는 그래도 문장에서 빼서 일수로 오인하지 않게 한다.
    """
    today = today or date.today()
    found = [m for m in (_DATE_A.search(question), _DATE_B.search(question)) if m]
    if not found:
        return None, None, question
    first = min(found, key=lambda m: m.start())
    year, month, day, end_pos = _match_date(question, first.start())
    start = _resolve_date(year, month, day, today)
    end = None
    sep = _RANGE_SEP.match(question, end_pos)
    if sep:
        full = _match_date(question, sep.end())
        if full:
            e_year, e_month, e_day, stop = full
            if start is not None:
                end = _resolve_date(e_year or start.year, e_month, e_day, today)
                if end is not None and end < start and e_year is None:
                    end = _resolve_date(start.year + 1, e_month, e_day, today)
            end_pos = stop
        else:
            only = _DAY_ONLY.match(question, sep.end())
            if only:
                if start is not None:
                    try:
                        end = date(start.year, start.month, int(only.group(1)))
                        if end < start:
                            end = (date(start.year + (start.month == 12), start.month % 12 + 1, 1) + timedelta(days=int(only.group(1)) - 1))
                    except ValueError:
                        end = None
                end_pos = only.end()
    tail = re.match(r"\s*까지", question[end_pos:])
    if tail:
        end_pos += tail.end()
    cleaned = f"{question[:first.start()]} {_DATE_MARK} {question[end_pos:]}"
    if start is not None and end is not None and end >= start and start < today:
        # 지난 날짜 범위는 날짜로 쓰지 않지만(기본값 D+30), 범위가 뜻한 여행 일수는 살린다
        cleaned = f"{question[:first.start()]} {_DATE_MARK} {(end - start).days + 1}일 {question[end_pos:]}"
    if start is None or start < today or (end is not None and end < start):
        return None, None, cleaned
    return start, end, cleaned


def _parse_labeled_route(question: str) -> tuple[str | None, str | None] | None:
    """'서울 출발 도쿄 도착', '서울발 도쿄행', '출발지는 서울, 도착지는 도쿄', '출발: 부산 도착: 오사카'처럼 출발·도착을 말로 밝힌 문장을 읽는다. 없으면 None."""
    m = re.search(rf"({_PLACE})\s*출발\s+(?:(?:{_COUNTRY_RE})\s+)?({_PLACE})\s*도착", question) or re.search(rf"({_PLACE}?)발\s+({_PLACE}?)행(?![가-힣])", question)
    if m:
        origin, dest = _place_token(m.group(1)), _place_token(m.group(2))
        return (origin or None, dest or None) if (origin or dest) else None
    o = re.search(rf"출발지?(?:은|는|이|가)?\s*[:：]\s*(?:(?:{_COUNTRY_RE})\s+)?({_PLACE})", question) or re.search(rf"출발지(?:은|는|이|가)?\s*(?:(?:{_COUNTRY_RE})\s+)?({_PLACE})", question)
    d = re.search(rf"도착지?(?:은|는|이|가)?\s*[:：]\s*(?:(?:{_COUNTRY_RE})\s+)?({_PLACE})", question) or re.search(rf"도착지(?:은|는|이|가)?\s*(?:(?:{_COUNTRY_RE})\s+)?({_PLACE})", question)
    origin = _place_token(o.group(1)) if o else ""
    dest = _place_token(d.group(1)) if d else ""
    return (origin or None, dest or None) if (origin or dest) else None


def parse_route(question: str) -> tuple[str | None, str | None]:
    """출발지와(명시된 경우) 목적지를 뽑는다. 'A에서 B로', 'A → B', 'A 출발' 형태만 출발지로 읽고, '제주에서 3박'처럼 에서만 있으면 읽지 않는다."""
    labeled = _parse_labeled_route(question)
    if labeled:
        return labeled
    m = re.search(rf"({_PLACE})\s*(?:->|→|➡|➜)\s*({_PLACE})", question)
    if not m:
        m = re.search(rf"({_PLACE})에서\s*({_PLACE}?)(?:으로|로|까지|행)(?![가-힣])", question)
    if not m:  # 'A에서 중국 장가계로' 처럼 나라 이름이 끼면 나라 뒤의 도시를 목적지로 본다
        m = re.search(rf"({_PLACE})에서\s+(?:{_COUNTRY_RE})\s+({_PLACE}?)(?:으로|로|까지|행)(?![가-힣])", question)
    if m and m.group(1) not in _STOPWORDS:
        return m.group(1), m.group(2)
    m = re.search(rf"({_PLACE}?)(?:에서)?\s*출발", question)
    if m and m.group(1) and m.group(1) not in _STOPWORDS:
        return m.group(1), None
    return None, None


def _place_token(raw: str) -> str:
    """토큰에서 구두점과 목적지 뒤 조사('로', '에서')를 떼고, 목적지로 쓸 수 있는 이름이면 돌려준다. 아니면 빈 문자열."""
    token = raw.strip(".,!?")
    if re.search(r"[가-힣]{2,}로$", token) and not token.endswith("으로"):
        token = token[:-1]  # '도쿄로' 처럼 목적지 뒤에 붙은 조사 제거
    elif re.search(r"[가-힣]{2,}에서$", token):
        token = token[:-2]  # '제주에서 3박 4일' 은 제주에서 머무는 여행(목적지)으로 본다
    if re.fullmatch(r"[A-Za-z가-힣]+", token) and token not in _NOT_PLACE and not re.search(r"(으로|해서)$", token):
        return token
    return ""


def _pick_destination(question: str, anchor: int | None) -> str:
    """기간·날짜 표기(anchor) 근처에서 여행지를 고른다. 앞쪽은 가까운 토큰부터 3개, 앞에서 못 찾으면 날짜 표기 뒤 2개를 본다.

    '중국 장가계'처럼 나라 이름이 먼저 잡히면 옆의 도시 이름을 여행지로 쓴다(도시가 없으면 나라).
    """
    if anchor is None:
        return ""
    tokens_before = [t for t in re.split(r"[\s,]+", question[:anchor].strip(" ,")) if t and t != _DATE_MARK]
    mark = question.find(_DATE_MARK)
    tokens_after = []
    if mark >= 0:  # '12월 20일부터 23일까지 도쿄 2명' 처럼 날짜 뒤에 목적지가 오는 경우
        tokens_after = [t for t in re.split(r"[\s,]+", question[mark + len(_DATE_MARK):].strip(" ,")) if t][:2]
    ordered = [[(t, tokens_before, k) for k, t in reversed(list(enumerate(tokens_before)))][:3], [(t, tokens_after, k) for k, t in enumerate(tokens_after)]]
    for candidates in ordered:
        for raw, group, k in candidates:
            if raw.strip(".,!?") in _FILLER:  # '장가계 여행 11/8~' 처럼 사이에 낀 군더더기 단어는 건너뛴다
                continue
            token = _place_token(raw)
            if not token:  # '정확한' 같은 수식어를 만나면 그 쪽에서는 더 찾지 않는다
                break
            if token in _COUNTRIES:
                for j in (k + 1, k - 1):  # 나라 이름 옆의 도시
                    if 0 <= j < len(group):
                        city = _place_token(group[j])
                        if city and city not in _COUNTRIES:
                            return city
            return token
    return ""


def parse_trip(question: str) -> dict:
    """자연어 한 줄에서 여행지·기간·인원·예산(원)·출발지·시작일을 규칙으로 뽑는다(LLM 미사용). 못 찾은 값은 None(여행지는 빈 문자열)."""
    start_date, end_date, question = parse_dates(question)
    budget = None
    m = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*만\s*원", question)
    if m:
        budget = int(round(float(m.group(1).replace(",", "")) * 10_000))
    else:
        m = re.search(r"(\d[\d,]*)\s*원", question)
        if m:
            budget = int(m.group(1).replace(",", ""))
    days, anchor = None, None
    m = re.search(r"(\d+)\s*박\s*(?:(\d+)\s*일)?", question)
    if m:
        days = int(m.group(2)) if m.group(2) else int(m.group(1)) + 1
        anchor = m.start()
    else:
        m = re.search(r"(\d+)\s*일", question)
        if m:
            days, anchor = int(m.group(1)), m.start()
        else:
            m = re.search(r"당일(?:치기)?", question)  # 당일치기는 1일(숙박 없음)
            if m:
                days, anchor = 1, m.start()
    if start_date and end_date:  # 날짜 범위가 있으면 기간은 종료일 - 시작일 + 1
        days = (end_date - start_date).days + 1
    if anchor is None and _DATE_MARK in question:
        anchor = question.find(_DATE_MARK)
    origin, route_dest = parse_route(question)
    destination = route_dest or ""
    if not destination:
        destination = _pick_destination(question, anchor)
    return {
        "destination": destination, "days": days, "people": parse_people(question),
        "budget_krw": budget, "origin": origin, "start_date": start_date.isoformat() if start_date else None,
    }


EDIT_RE = re.compile(r"변경|바꿔|바꾸|수정|교체|추가|빼|제외|줄여|늘려|낮춰|올려|대신|말고")  # 선택과 함께 온 수정 요청 표현


def parse_selection(question: str) -> str | None:
    """'2번 안', '3안에서', '가성비형' 같은 문장에서 안 종류를 고른다. 모르면 None.

    숫자는 '번'·'안'이 붙었거나 숫자만 입력한 경우에만 선택으로 본다('숙소를 2박으로'의 2는 선택이 아니다).
    """
    m = re.search(r"(?<!\d)([1-3])\s*(?:번|안)(?!\d)", question) or re.fullmatch(r"\s*([1-3])\s*", question)
    if m:
        return PLAN_TYPES[int(m.group(1)) - 1]
    for pt in PLAN_TYPES:
        if pt[:-1] in question:
            return pt
    return None


def has_edit_request(question: str) -> bool:
    """문장에 일정 수정 요청 표현('3안에서 숙소를 변경해줘')이 있으면 True. 안을 고르기 전에는 수정 요청을 받지 않는 데 쓴다."""
    return bool(EDIT_RE.search(question))


def _pending_stage(snapshot: Any) -> str | None:
    """체크포인트에서 대기 중인 interrupt 종류(select/chat)를 읽는다."""
    for task in snapshot.tasks:
        for it in task.interrupts:
            if isinstance(it.value, dict):
                return it.value.get("stage")
    return None


def _run_config(session_id: str) -> dict:
    """그래프 실행 설정을 만든다. thread_id 는 session_id, LANGFUSE_* 환경변수가 있으면 Langfuse 추적 콜백을 붙인다."""
    from src.config import export_aws_credentials

    export_aws_credentials()
    config: dict = {"configurable": {"thread_id": session_id}}
    if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
        from langfuse.langchain import CallbackHandler

        config["callbacks"] = [CallbackHandler()]
        config["metadata"] = {"langfuse_session_id": session_id}
    return config


def resolve_trip(question: str, model: Any | None) -> dict:
    """문장에서 여행 정보를 뽑는다. 모델이 판단한 값을 쓰고, 코드는 빈 값만 확인해 비어 있는 항목은 규칙 파서(parse_trip) 값으로 채운다.

    model 이 없거나 호출이 실패하면 규칙 파서 결과를 그대로 쓴다(키와 오류 내용은 로그에 남기지 않는다).
    """
    rules = parse_trip(question)
    if model is None:
        return rules
    try:
        found = extract_trip(model, question)
    except Exception as exc:  # 모델 호출·형식 오류는 규칙 파서로 대신한다
        logger.warning("여행 정보 추출 실패, 규칙 파서로 대신합니다: %s", type(exc).__name__)
        return rules
    return {key: found.get(key) or rules.get(key) for key in rules}


def create_app(graph: Any | None = None, extractor_model: Any | None = None) -> FastAPI:
    """FastAPI 앱을 만든다. graph 를 주면 그 그래프를 쓰고, 없으면 첫 요청에서 만든다.

    새 요청의 여행 정보는 모델이 뽑는다. extractor_model 을 주면 그 모델을 쓰고, graph 도 extractor_model 도 없으면(실서비스) 첫 요청에서 모델을 만든다.
    graph 만 주면(테스트) 모델을 부르지 않고 규칙 파서만 쓴다.
    """
    app = FastAPI(title="여행 일정 에이전트")
    holder: dict = {"graph": graph, "extractor": extractor_model}
    known_trips: dict[str, dict] = {}  # 정보 부족으로 이어가는 세션의 파싱값(서버 메모리, 그래프 시작 전)

    def get_graph():
        """그래프(체크포인터 포함)를 한 번만 만들어 재사용한다."""
        if holder["graph"] is None:
            holder["graph"] = build_graph()
        return holder["graph"]

    def get_extractor():
        """여행 정보 추출용 모델. 실서비스에서는 첫 요청에서 한 번 만들고, 테스트(그래프만 주입)에서는 None 이다."""
        if holder["extractor"] is None and graph is None:
            holder["extractor"] = build_model()
        return holder["extractor"]

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        """본문 형식 오류는 422."""
        return error_response(422, "invalid_request", "요청 형식이 올바르지 않습니다")

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        """API 오류를 지정한 상태 코드로 돌려준다."""
        return error_response(exc.status, exc.code, exc.message, exc.extra)

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception):
        """예상 못 한 오류는 500. 원인 파악을 위해 스택 트레이스를 로그에 남긴다."""
        logger.error("처리되지 않은 예외: %s %s", request.method, request.url.path, exc_info=exc)
        return error_response(500, "internal_error", "서버 내부 오류가 발생했습니다")

    def run(g: Any, graph_input: Any, config: dict, on_step: Any) -> None:
        """그래프를 실행한다. on_step 이 있으면 노드가 끝날 때마다 노드 이름을 알려 준다(진행 상태 표시용)."""
        if on_step is None:
            g.invoke(graph_input, config)
            return
        for chunk in g.stream(graph_input, config, stream_mode="updates"):
            for node in chunk:
                if not node.startswith("__"):
                    on_step(node)

    def process(body: QueryRequest, on_step: Any = None) -> dict:
        """첫 요청은 새 세션으로 그래프를 실행하고, session_id 가 있으면 대기 중 interrupt 를 재개한다."""
        question = (body.question or "").strip()
        if not question:
            raise ApiError(422, "empty_question", "question 이 비어 있습니다")
        g = get_graph()
        new_trip = body.session_id is None or body.session_id in known_trips
        trip = None
        if new_trip:
            early = classify_request(question, _FILLED_TRIP, True)  # 비밀·개인정보·실시간 조회 요청은 모델을 부르기 전에 규칙으로 거절한다(누락 검사는 제외)
            if early:
                raise ApiError(422, early["code"], early["message"])
            parsed = resolve_trip(question, get_extractor())
            known = known_trips.get(body.session_id)
            if restates_trip(known, parsed):  # 이미 알던 기간·날짜를 다시 말하면 새 요청이다(이전 여행지를 물려받지 않는다)
                known = None
            trip = merge_trip(known, parsed)
        verdict = classify_request(question, trip, new_trip)
        if verdict:
            extra = {}
            if verdict["code"] == "missing_fields":
                session_id = body.session_id or f"sess_{uuid.uuid4().hex}"
                known_trips[session_id] = trip
                extra = {"session_id": session_id, "missing": verdict["missing"]}
            raise ApiError(422, verdict["code"], verdict["message"], extra)
        notice = CONFLICT_NOTICE if new_trip and has_conflict(question) else ""
        if new_trip:
            session_id = body.session_id or f"sess_{uuid.uuid4().hex}"
            known_trips.pop(session_id, None)
            config = _run_config(session_id)
            run(g, {"trip": trip}, config, on_step)
        else:
            session_id = body.session_id
            config = _run_config(session_id)
            snapshot = g.get_state(config)
            if not snapshot.values:
                raise ApiError(404, "session_not_found", "알 수 없는 session_id 입니다")
            stage = _pending_stage(snapshot)
            if stage == "select":
                if has_edit_request(question):  # 안을 고르기 전에는 수정 요청을 받지 않는다(안 선택 후 대화로만 수정)
                    raise ApiError(
                        422, "edit_before_selection",
                        "수정 요청은 안을 선택한 뒤에 할 수 있습니다. 먼저 맛집형·명소형·가성비형 중 하나(1~3번)를 골라 주세요",
                    )
                plan = parse_selection(question)
                if plan is None:
                    raise ApiError(422, "invalid_selection", "맛집형·명소형·가성비형 중 하나(1~3번)를 골라 주세요")
                resume = {"type": "select", "plan": plan}
            elif stage == "chat":
                if question.lower() in DONE_WORDS:
                    resume = {"type": "done"}
                else:
                    resume = {"type": "chat", "text": question}
            else:
                raise ApiError(404, "session_closed", "종료되었거나 대기 중이 아닌 세션입니다")
            run(g, Command(resume=resume), config, on_step)
        response = _build_response(g, config, session_id)
        if notice:
            response["answer"]["plainAnswer"] = f"{notice}\n{response['answer']['plainAnswer']}"
        return response

    @app.post("/query")
    def query(body: QueryRequest):
        """POST /query: 처리가 끝난 뒤 answer·contexts·trace 를 한 번에 돌려준다."""
        return process(body)

    @app.post("/query/stream")
    def query_stream(body: QueryRequest):
        """진행 상태 표시용: 노드가 끝날 때마다 {"step": 노드} 줄을, 마지막에 {"result": ...} 또는 {"error": ...} 줄을 보낸다(NDJSON)."""
        events: queue.Queue = queue.Queue()

        def worker() -> None:
            try:
                events.put({"result": process(body, lambda node: events.put({"step": node}))})
            except ApiError as exc:
                events.put({"error": {"code": exc.code, "message": exc.message, **exc.extra}, "status": exc.status})
            except Exception:
                logger.exception("스트림 처리 중 예외: POST /query/stream")
                events.put({"error": {"code": "internal_error", "message": "서버 내부 오류가 발생했습니다"}, "status": 500})
            events.put(None)

        def lines():
            threading.Thread(target=worker, daemon=True).start()
            while (event := events.get()) is not None:
                yield json.dumps(event, ensure_ascii=False) + "\n"

        return StreamingResponse(lines(), media_type="application/x-ndjson")

    @app.get("/", include_in_schema=False)
    def index():
        """검증용 웹 UI(web/index.html)를 돌려준다."""
        return FileResponse(WEB_INDEX)

    return app


def _build_response(g: Any, config: dict, session_id: str) -> dict:
    """그래프 상태에서 answer·contexts·trace 응답을 조립한다. 생성 불가면 ApiError."""
    state = g.get_state(config).values
    if not state.get("plans"):
        errors = state.get("errors") or []
        if state.get("trip") and any("필요" in e or "이상" in e or "보다 큰" in e for e in errors):
            raise ApiError(422, "invalid_trip", "; ".join(errors))
        raise ApiError(500, "generation_failed", "일정을 생성하지 못했습니다")
    answer = build_answer(state)
    return {
        "answer": {"session_id": session_id, "plainAnswer": answer["plainAnswer"], "html": answer["html"]},
        "contexts": state.get("tool_results", []),
        "trace": state.get("trace", []),
    }


app = create_app()
