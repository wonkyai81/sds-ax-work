"""AC1: 3안 병렬 생성·합계·판정 테스트 (가짜 모델, 네트워크 없음)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402
from langgraph.types import Command  # noqa: E402

from src import tools  # noqa: E402
from src.agent import PlanOutput, build_graph  # noqa: E402

TRIP = {"destination": "도쿄", "days": 2, "budget_krw": 1_000_000}


def make_output(per_item: int, days: int = 2) -> PlanOutput:
    """일수 x 3시간대 항목, 항목당 per_item 원인 구조화 출력을 만든다."""
    slots = ["아침", "오후", "저녁"]
    return PlanOutput(items=[
        {"day": d, "time_slot": s, "activity": f"{d}-{s}", "cost_krw": per_item, "transport": "지하철", "note": "추정"}
        for d in range(1, days + 1) for s in slots
    ])


class FakeModel(BaseChatModel):
    """구조화 출력은 prompt 기반 함수로, 일반 호출은 도구 호출 스크립트로 응답하는 가짜 모델."""

    planner: object = None
    script: list = []

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self.script.pop(0) if self.script else AIMessage(content="완료")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(lambda prompt: self.planner(prompt))


def run(planner, script=None):
    """그래프를 만들고 첫 요청을 실행해 (graph, config, 결과)를 돌려준다."""
    graph = build_graph(FakeModel(planner=planner, script=script or []))
    config = {"configurable": {"thread_id": "s1"}}
    result = graph.invoke({"trip": TRIP}, config)
    return graph, config, result


def test_three_plans_generated_and_wait_for_selection():
    graph, config, result = run(lambda p: make_output(50_000))
    assert set(result["plans"]) == {"맛집형", "명소형", "가성비형"}
    assert "__interrupt__" in result
    assert result["trace"].count("generate_plan") == 3
    assert result["trace"][-1] == "collect_plans"  # await_selection 은 interrupt 로 멈춰 있음
    assert graph.get_state(config).next == ("await_selection",)
    for plan in result["plans"].values():
        assert len(plan["attempts"]) == 1
        assert plan["attempts"][0][0].keys() == {"day", "time_slot", "activity", "cost_krw", "transport", "note"}


def test_totals_match_tool_and_targets():
    graph, config, result = run(lambda p: make_output(100_000))  # 합계 600,000
    checks = [r for r in result["tool_results"] if r["tool"] == "check_budget"]
    assert len(checks) == 3
    for plan_type, plan in result["plans"].items():
        verdict = tools.check_budget(plan_type, plan["attempts"][-1], TRIP["budget_krw"])
        assert verdict == {"within": True, "total": 600_000, "overage": 0}
    assert tools.target_budget("가성비형", 1_000_000) == 800_000
    assert tools.target_budget("맛집형", 1_000_000) == 1_000_000


def test_value_plan_uses_80_percent_target():
    # 합계 900,000: 맛집형·명소형은 충족, 가성비형(목표 800,000)은 초과 → 재조정
    graph, config, result = run(lambda p: make_output(150_000))
    assert len(result["plans"]["맛집형"]["attempts"]) == 1
    assert len(result["plans"]["명소형"]["attempts"]) == 1
    assert len(result["plans"]["가성비형"]["attempts"]) == 4  # 1 + 재조정 3회
    assert "pick_closest" in result["trace"]


def test_state_holds_only_raw_data():
    graph, config, result = run(lambda p: make_output(50_000))
    state = graph.get_state(config).values
    assert not ({"total", "html", "prompt", "overage"} & set(state))


def test_partial_failure_keeps_other_plans():
    def planner(prompt):
        if "가성비" in prompt:
            return {"items": []}  # 항상 실패
        return make_output(50_000)

    graph, config, result = run(planner)
    assert set(result["plans"]) == {"맛집형", "명소형"}
    assert result["errors"]


def test_all_fail_ends_without_selection():
    graph, config, result = run(lambda p: {"items": []})
    assert not result.get("plans")
    assert "__interrupt__" not in result


@pytest.mark.parametrize("trip", [
    {"destination": "", "days": 2, "budget_krw": 1000},
    {"destination": "파리", "days": 0, "budget_krw": 1000},
    {"destination": "파리", "days": 2, "budget_krw": 0},
])
def test_invalid_request_stops(trip):
    graph = build_graph(FakeModel(planner=lambda p: make_output(1)))
    result = graph.invoke({"trip": trip}, {"configurable": {"thread_id": "bad"}})
    assert result["errors"] and not result.get("plans")


def test_select_chat_and_done_flow():
    new_items = [
        {"day": 1, "time_slot": "아침", "activity": "수정", "cost_krw": 10_000, "transport": "도보", "note": "n"}
    ]
    call = AIMessage(content="", tool_calls=[{"name": "submit_plan_items", "args": {"items": new_items}, "id": "c1"}])
    graph, config, result = run(lambda p: make_output(50_000), script=[call, AIMessage(content="끝")])
    result = graph.invoke(Command(resume={"type": "select", "plan": "맛집형"}), config)
    assert result["selected"] == "맛집형" and "__interrupt__" in result
    result = graph.invoke(Command(resume={"type": "chat", "text": "첫 아침을 바꿔줘"}), config)
    assert result["selected_items"] == new_items
    assert "chat_edit" in result["trace"] and "check_budget_selected" in result["trace"]
    graph.invoke(Command(resume={"type": "done"}), config)
    assert not graph.get_state(config).next
