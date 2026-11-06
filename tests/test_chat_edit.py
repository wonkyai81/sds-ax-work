"""AC3: 선택 안 대화 수정(chat_edit)과 수정 후 재조정 테스트 (가짜 모델, 네트워크 없음)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.types import Command  # noqa: E402

from src import tools  # noqa: E402
from src.agent import build_graph  # noqa: E402
from tests.test_plans import TRIP, FakeModel, make_output  # noqa: E402


def submit(items, call_id="c1"):
    """submit_plan_items 도구 호출을 하는 가짜 모델 응답."""
    return AIMessage(content="", tool_calls=[{"name": "submit_plan_items", "args": {"items": items}, "id": call_id}])


def to_selected(planner, script, plan="맛집형", thread="e1"):
    """3안 생성 → 안 선택까지 진행한 그래프와 설정을 돌려준다."""
    graph = build_graph(FakeModel(planner=planner, script=script))
    cfg = {"configurable": {"thread_id": thread}}
    graph.invoke({"trip": TRIP}, cfg)
    graph.invoke(Command(resume={"type": "select", "plan": plan}), cfg)
    return graph, cfg


TIME_KEYS = ("depart_time", "arrive_time")


def without_times(item):
    """시각은 비어 있으면 코드가 채우므로 비교에서 뺀다."""
    return {k: v for k, v in item.items() if k not in TIME_KEYS}


def base_items(per_item=50_000):
    """선택 안의 원본 항목 목록(딕셔너리)."""
    return [i.model_dump() for i in make_output(per_item).items]


def test_chat_edit_changes_only_requested_part():
    items = base_items()
    edited = [dict(i) for i in items]
    edited[2] = {**edited[2], "activity": "스시 맛집", "cost_krw": 80_000}
    graph, cfg = to_selected(lambda p: make_output(50_000), [submit(edited), AIMessage(content="끝")])
    result = graph.invoke(Command(resume={"type": "chat", "text": "1일차 저녁을 스시로"}), cfg)
    state = graph.get_state(cfg).values
    assert state["selected_items"][2]["activity"] == "스시 맛집"
    for idx in (0, 1, 3, 4, 5):
        got = {k: v for k, v in state["selected_items"][idx].items() if k not in TIME_KEYS}  # 시각은 비어 있으면 코드가 채운다
        assert got == {k: v for k, v in items[idx].items() if k not in TIME_KEYS}  # 요청하지 않은 부분은 그대로
    assert "chat_edit" in result["trace"]
    assert "readjust_selected" not in result["trace"]  # 예산 이내면 재조정 없음
    assert "330,000" in result["__interrupt__"][0].value["plainAnswer"]  # 수정 후 합계
    assert state["selected"] == "맛집형"


def test_chat_edit_over_budget_runs_readjust_and_stops_when_met():
    items = base_items()
    over = [dict(i) for i in items]
    over[0] = {**over[0], "cost_krw": 2_000_000}

    def planner(prompt):
        if "초과했습니다" in prompt:
            return make_output(100_000)  # 합계 600,000 으로 충족
        return make_output(50_000)

    graph, cfg = to_selected(planner, [submit(over), AIMessage(content="끝")])
    result = graph.invoke(Command(resume={"type": "chat", "text": "첫 활동을 고급으로"}), cfg)
    state = graph.get_state(cfg).values
    assert result["trace"].count("readjust_selected") == 1
    assert len(state["selected_attempts"]) == 2
    assert tools.check_budget("맛집형", state["selected_items"], TRIP["budget_krw"])["within"]


def test_each_chat_edit_gets_fresh_three_readjustments():
    over = [{"day": 1, "time_slot": "아침", "activity": "고가", "cost_krw": 5_000_000, "transport": "택시", "note": "n"}]
    script = [submit(over, "c1"), AIMessage(content="끝"), submit(over, "c2"), AIMessage(content="끝")]
    graph, cfg = to_selected(lambda p: make_output(50_000) if "초과했습니다" not in p else {"items": over}, script)
    first = graph.invoke(Command(resume={"type": "chat", "text": "비싸게"}), cfg)
    assert first["trace"].count("readjust_selected") == 3
    assert len(graph.get_state(cfg).values["selected_attempts"]) == 4
    second = graph.invoke(Command(resume={"type": "chat", "text": "더 비싸게"}), cfg)
    assert second["trace"].count("readjust_selected") == 6  # 두 번째 수정에서 새로 3회
    assert len(graph.get_state(cfg).values["selected_attempts"]) == 4
    assert "경고" in second["__interrupt__"][0].value["plainAnswer"]


def test_chat_edit_without_submission_keeps_items():
    items = base_items()
    graph, cfg = to_selected(lambda p: make_output(50_000), [AIMessage(content="수정하지 못했습니다")])
    result = graph.invoke(Command(resume={"type": "chat", "text": "아무거나"}), cfg)
    state = graph.get_state(cfg).values
    assert [without_times(i) for i in state["selected_items"]] == [without_times(i) for i in items]
    assert state["errors"]
    assert "readjust_selected" not in result["trace"]
