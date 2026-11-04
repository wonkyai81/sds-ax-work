"""FastAPI 앱: POST /query 로 LangGraph 에이전트를 구동하고 interrupt 를 재개한다."""
import re
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from langgraph.types import Command
from pydantic import BaseModel

from src.agent import PLAN_TYPES, build_answer, build_graph
from src.tools import CONFLICT_NOTICE, classify_request, has_conflict, merge_trip

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


def parse_trip(question: str) -> dict:
    """자연어 한 줄에서 여행지·기간·예산(원)을 규칙으로 뽑는다(LLM 미사용). 못 찾은 값은 None."""
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
    destination = ""
    if anchor is not None:
        before = re.split(r"[\s,]+", question[:anchor].strip(" ,"))
        token = before[-1].strip(".,!?") if before else ""
        if re.fullmatch(r"[A-Za-z가-힣]+", token or "") and token not in _STOPWORDS and not re.search(r"(으로|해서|에서)$", token):
            destination = token
    return {"destination": destination, "days": days, "budget_krw": budget}


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

    @app.post("/query")
    def query(body: QueryRequest):
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
            config = {"configurable": {"thread_id": session_id}}
            g.invoke({"trip": trip}, config)
        else:
            session_id = body.session_id
            config = {"configurable": {"thread_id": session_id}}
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
            g.invoke(Command(resume=resume), config)
        response = _build_response(g, config, session_id)
        if notice:
            response["answer"]["plainAnswer"] = f"{notice}\n{response['answer']['plainAnswer']}"
        return response

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
