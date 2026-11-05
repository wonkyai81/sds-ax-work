"""FastAPI 앱: POST /query 로 LangGraph 에이전트를 구동하고 interrupt 를 재개한다."""
import json
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

from src.agent import PLAN_TYPES, build_answer, build_graph
from src.tools import CONFLICT_NOTICE, classify_request, has_conflict, merge_trip

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


def parse_route(question: str) -> tuple[str | None, str | None]:
    """출발지와(명시된 경우) 목적지를 뽑는다. 'A에서 B로', 'A → B', 'A 출발' 형태만 출발지로 읽고, '제주에서 3박'처럼 에서만 있으면 읽지 않는다."""
    m = re.search(rf"({_PLACE})\s*(?:->|→|➡|➜)\s*({_PLACE})", question)
    if not m:
        m = re.search(rf"({_PLACE})에서\s*({_PLACE}?)(?:으로|로|까지|행)(?![가-힣])", question)
    if m and m.group(1) not in _STOPWORDS:
        return m.group(1), m.group(2)
    m = re.search(rf"({_PLACE}?)(?:에서)?\s*출발", question)
    if m and m.group(1) and m.group(1) not in _STOPWORDS:
        return m.group(1), None
    return None, None


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
    if not destination and anchor is not None:
        before = [t for t in re.split(r"[\s,]+", question[:anchor].strip(" ,")) if t != _DATE_MARK]
        token = before[-1].strip(".,!?") if before else ""
        if re.search(r"[가-힣]{2,}로$", token) and not token.endswith("으로"):
            token = token[:-1]  # '도쿄로' 처럼 목적지 뒤에 붙은 조사 제거
        elif re.search(r"[가-힣]{2,}에서$", token):
            token = token[:-2]  # '제주에서 3박 4일' 은 제주에서 머무는 여행(목적지)으로 본다
        if re.fullmatch(r"[A-Za-z가-힣]+", token or "") and token not in _STOPWORDS and not re.search(r"(으로|해서)$", token):
            destination = token
    if not destination and _DATE_MARK in question:  # '12월 20일부터 23일까지 도쿄 2명' 처럼 날짜 뒤에 목적지가 오는 경우
        after = [t for t in re.split(r"[\s,]+", question[question.find(_DATE_MARK) + len(_DATE_MARK):].strip(" ,")) if t]
        token = after[0].strip(".,!?") if after else ""
        if re.search(r"[가-힣]{2,}로$", token) and not token.endswith("으로"):
            token = token[:-1]
        if re.fullmatch(r"[A-Za-z가-힣]+", token or "") and token not in _STOPWORDS and not re.search(r"(으로|해서)$", token):
            destination = token
    return {
        "destination": destination, "days": days, "people": parse_people(question),
        "budget_krw": budget, "origin": origin, "start_date": start_date.isoformat() if start_date else None,
    }


def parse_selection(question: str) -> str | None:
    """'2번 안', '가성비형' 같은 문장에서 안 종류를 고른다. 모르면 None."""
    m = re.search(r"([1-3])\s*(?:번|안)?", question)
    if m and not re.search(r"\d{2,}", question):
        return PLAN_TYPES[int(m.group(1)) - 1]
    for pt in PLAN_TYPES:
        if pt[:-1] in question:
            return pt
    return None


def _pending_stage(snapshot: Any) -> str | None:
    """체크포인트에서 대기 중인 interrupt 종류(select/chat)를 읽는다."""
    for task in snapshot.tasks:
        for it in task.interrupts:
            if isinstance(it.value, dict):
                return it.value.get("stage")
    return None


def _run_config(session_id: str) -> dict:
    """그래프 실행 설정을 만든다. thread_id 는 session_id, LANGFUSE_* 환경변수가 있으면 Langfuse 추적 콜백을 붙인다."""
    config: dict = {"configurable": {"thread_id": session_id}}
    if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
        from langfuse.langchain import CallbackHandler

        config["callbacks"] = [CallbackHandler()]
        config["metadata"] = {"langfuse_session_id": session_id}
    return config


def create_app(graph: Any | None = None) -> FastAPI:
    """FastAPI 앱을 만든다. graph 를 주면 그 그래프를 쓰고, 없으면 첫 요청에서 만든다."""
    app = FastAPI(title="여행 일정 에이전트")
    holder: dict = {"graph": graph}
    known_trips: dict[str, dict] = {}  # 정보 부족으로 이어가는 세션의 파싱값(서버 메모리, 그래프 시작 전)

    def get_graph():
        """그래프(체크포인터 포함)를 한 번만 만들어 재사용한다."""
        if holder["graph"] is None:
            holder["graph"] = build_graph()
        return holder["graph"]

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
        """예상 못 한 오류는 500."""
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
            trip = merge_trip(known_trips.get(body.session_id), parse_trip(question))
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
