"""여행 일정 에이전트 그래프: LangGraph StateGraph (DESIGN 2~9절). 예산 판정은 tools.py 코드가 한다."""
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
    check_budget,
    closest_attempt,
    final_items,
    needs_readjust,
    render_plain_answer,
    render_report_html,
    target_budget,
)

LLM_RETRY = RetryPolicy(max_attempts=3, initial_interval=1.0)
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


class PlanOutput(BaseModel):
    """구조화 출력 스키마: 일정 항목 목록."""

    items: list[ItemModel]


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
    selected: PlanType | None
    selected_items: list[ItineraryItem]
    selected_attempts: list[list[ItineraryItem]]  # 수정 1회마다 새로 시작하는 재조정 시도 (원본)
    next_action: str | None  # await_chat 재개 값: "chat" | "done"
    chat_history: Annotated[list, add_messages]
    tool_results: Annotated[list[dict], _concat]
    trace: Annotated[list[str], _concat]
    errors: Annotated[list[str], _concat]


def build_model() -> Any:
    """config 의 리전·모델 ID 로 ChatBedrockConverse 를 만든다 (테스트에서는 가짜 모델로 교체)."""
    from langchain_aws import ChatBedrockConverse

    from src.config import load_settings

    settings = load_settings()
    return ChatBedrockConverse(model=settings.model_id, region_name=settings.region)


def _items_text(items: list[ItineraryItem]) -> str:
    """프롬프트용 항목 문자열."""
    return "\n".join(
        f"{i['day']}일차 {i['time_slot']}: {i['activity']} ({i['cost_krw']}원, {i['transport']}, {i['note']})"
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
    for _ in range(2):
        try:
            return _to_items(structured.invoke(prompt)), errors
        except (ValidationError, ValueError, TypeError) as exc:
            errors.append(f"구조화 출력 오류: {exc}")
    return None, errors


def build_graph(model: Any | None = None, checkpointer: Any | None = None):
    """에이전트 StateGraph 를 조립해 컴파일한다. model 을 주면 그 모델을 쓴다(테스트용)."""
    llm = model if model is not None else build_model()

    def validate_request(state: GraphState) -> dict:
        """여행지·기간·예산 형식을 확인한다 (범위 상한 없음)."""
        trip = state.get("trip") or {}
        errors = []
        if not str(trip.get("destination", "")).strip():
            errors.append("여행지가 필요합니다")
        if not isinstance(trip.get("days"), int) or isinstance(trip.get("days"), bool) or trip["days"] <= 0:
            errors.append("여행 기간은 1일 이상의 정수여야 합니다")
        if not isinstance(trip.get("budget_krw"), int) or isinstance(trip.get("budget_krw"), bool) or trip["budget_krw"] <= 0:
            errors.append("예산은 0원보다 큰 정수여야 합니다")
        return {"errors": errors, "trace": ["validate_request"]}

    def route_after_validate(state: GraphState) -> list[Send] | Literal["__end__"]:
        """검증 실패면 종료, 성공이면 안 종류별로 병렬 생성(Send)."""
        if state.get("errors"):
            return END
        return [Send("generate_plan", {"trip": state["trip"], "plan_type": pt}) for pt in PLAN_TYPES]

    def generate_plan(payload: dict) -> dict:
        """안 1개를 생성한다. 실패하면 이 안은 plans 에 남기지 않는다."""
        trip, plan_type = payload["trip"], payload["plan_type"]
        prompt = (
            f"{trip['destination']} {trip['days']}일 여행 일정을 {PLAN_STYLE[plan_type]} 만드세요. "
            f"목표 총액은 {target_budget(plan_type, trip['budget_krw'])}원 이하입니다. "
            "각 일차마다 아침/오후/저녁 항목을 두고, 비용은 원화(KRW)로 추정하세요."
        )
        items, errors = _call_structured(llm, prompt)
        update: dict = {"trace": ["generate_plan"], "errors": errors}
        if items is not None:
            update["plans"] = {plan_type: {"plan_type": plan_type, "attempts": [items]}}
        return update

    def check_budget_node(state: GraphState) -> dict:
        """모든 안의 합계와 판정을 코드로 계산해 contexts 에 남긴다."""
        budget = state["trip"]["budget_krw"]
        results = []
        for pt, plan in state.get("plans", {}).items():
            res = check_budget(pt, plan["attempts"][-1], budget)
            results.append({"tool": "check_budget", "plan_type": pt, "result": res})
        return {"trace": ["check_budget"], "tool_results": results}

    def route_after_check(state: GraphState) -> list[Send] | Literal["pick_closest", "collect_plans"]:
        """초과 + 재조정 3회 미만인 안은 readjust_plan(Send), 없으면 근접 안 선택 또는 수집."""
        budget = state["trip"]["budget_krw"]
        plans = state.get("plans", {})
        sends = [
            Send("readjust_plan", {"trip": state["trip"], "plan_type": pt, "attempts": plan["attempts"]})
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
        res = check_budget(plan_type, last, trip["budget_krw"])
        prompt = (
            f"{trip['destination']} {trip['days']}일 {plan_type} 일정의 합계 {res['total']}원이 목표 "
            f"{target_budget(plan_type, trip['budget_krw'])}원을 {res['overage']}원 초과했습니다. "
            f"항목 금액을 줄여 목표 이하로 다시 만드세요.\n{_items_text(last)}"
        )
        items, errors = _call_structured(llm, prompt)
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
        budget = state["trip"]["budget_krw"]
        results = []
        for pt, plan in state.get("plans", {}).items():
            if not check_budget(pt, plan["attempts"][-1], budget)["within"]:
                best = closest_attempt(plan["attempts"], pt, budget)
                results.append({"tool": "closest_attempt", "plan_type": pt, "result": check_budget(pt, best, budget)})
        return {"trace": ["pick_closest"], "tool_results": results}

    def collect_plans(state: GraphState) -> dict:
        """3안 결과 모음 지점 (plans 는 리듀서가 이미 합쳤다)."""
        return {"trace": ["collect_plans"]}

    def route_after_collect(state: GraphState) -> Literal["await_selection", "__end__"]:
        """생성된 안이 하나도 없으면 종료(오류 응답), 있으면 선택 대기."""
        return "await_selection" if state.get("plans") else END

    def await_selection(state: GraphState) -> dict:
        """3안을 보여주고 사용자 선택을 interrupt 로 기다린다."""
        payload = {"stage": "select", "plainAnswer": render_plain_answer(state["trip"], state["plans"])}
        value = interrupt(payload)
        plan = value.get("plan") if isinstance(value, dict) and value.get("type") == "select" else None
        if plan not in state["plans"]:
            return {"trace": ["await_selection"], "errors": [f"알 수 없는 안: {plan}"], "selected": None}
        items = final_items(plan, state["plans"][plan]["attempts"], state["trip"]["budget_krw"])
        return {"trace": ["await_selection"], "selected": plan, "selected_items": items, "selected_attempts": [items]}

    def route_after_selection(state: GraphState) -> Literal["await_selection", "render_answer"]:
        """선택이 유효하면 리포트 렌더, 아니면 다시 선택 대기."""
        return "render_answer" if state.get("selected") else "await_selection"

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

        agent = create_agent(
            llm,
            tools=[get_plan_items, submit_plan_items],
            system_prompt=(
                "당신은 여행 일정 수정 도우미입니다. get_plan_items 로 현재 일정을 확인하고, 사용자가 요청한 부분만 고쳐 "
                "submit_plan_items 로 전체 항목을 제출하세요. 비용은 원화로 추정하고, 합계나 예산 판단은 말하지 마세요."
            ),
        )
        request = state["chat_history"][-1].content if state.get("chat_history") else ""
        agent.invoke({"messages": [HumanMessage(content=str(request))]})
        submitted = holder["submitted"]
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
        budget = state["trip"]["budget_krw"]
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
        if needs_readjust(state["selected"], state["selected_attempts"], state["trip"]["budget_krw"]):
            return "readjust_selected"
        return "render_answer"

    def readjust_selected(state: GraphState) -> dict:
        """선택 안이 초과하면 초과 금액을 알려 항목 금액을 줄여 다시 생성한다."""
        trip, pt, attempts = state["trip"], state["selected"], state["selected_attempts"]
        last = attempts[-1]
        res = check_budget(pt, last, trip["budget_krw"])
        prompt = (
            f"{trip['destination']} {pt} 일정의 합계 {res['total']}원이 목표 {target_budget(pt, trip['budget_krw'])}원을 "
            f"{res['overage']}원 초과했습니다. 사용자의 수정 요청 내용은 유지하며 항목 금액을 줄여 다시 만드세요.\n"
            f"{_items_text(last)}"
        )
        items, errors = _call_structured(llm, prompt)
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
    g.add_node("generate_plan", generate_plan, retry_policy=LLM_RETRY)
    g.add_node("check_budget", check_budget_node)
    g.add_node("readjust_plan", readjust_plan, retry_policy=LLM_RETRY)
    g.add_node("pick_closest", pick_closest)
    g.add_node("collect_plans", collect_plans)
    g.add_node("await_selection", await_selection)
    g.add_node("await_chat", await_chat)
    g.add_node("chat_edit", chat_edit, retry_policy=LLM_RETRY)
    g.add_node("check_budget_selected", check_budget_selected)
    g.add_node("readjust_selected", readjust_selected, retry_policy=LLM_RETRY)
    g.add_node("render_answer", render_answer)

    g.add_edge(START, "validate_request")
    g.add_conditional_edges("validate_request", route_after_validate, ["generate_plan", END])
    g.add_edge("generate_plan", "check_budget")
    g.add_conditional_edges("check_budget", route_after_check, ["readjust_plan", "pick_closest", "collect_plans"])
    g.add_edge("readjust_plan", "check_budget")
    g.add_edge("pick_closest", "collect_plans")
    g.add_conditional_edges("collect_plans", route_after_collect, ["await_selection", END])
    g.add_conditional_edges("await_selection", route_after_selection, ["await_selection", "render_answer"])
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
        return {"plainAnswer": render_plain_answer(trip, state.get("plans", {})), "html": ""}
    items = state["selected_items"]
    return {
        "plainAnswer": render_plain_answer(trip, state["plans"], selected, items),
        "html": render_report_html(selected, items, trip["budget_krw"]),
    }
