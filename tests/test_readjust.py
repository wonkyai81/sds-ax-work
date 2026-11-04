"""AC2: 안별 최대 3회 자동 재조정과 가장 근접한 안·초과 경고 테스트 (가짜 모델, 네트워크 없음)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.types import Command  # noqa: E402

from src import tools  # noqa: E402
from src.agent import build_graph  # noqa: E402
from tests.test_plans import TRIP, FakeModel, make_output, run  # noqa: E402


def test_over_budget_plan_readjusted_at_most_three_times():
    calls = {"가성비": 0}

    def planner(prompt):
        if "가성비" in prompt:
            calls["가성비"] += 1
        return make_output(150_000)  # 합계 900,000 > 가성비 목표 800,000, 다른 안은 충족

    graph, config, result = run(planner)
    assert len(result["plans"]["가성비형"]["attempts"]) == 4  # 원본 + 재조정 3회
    assert calls["가성비"] == 4  # 4번째 이후 재호출 없음
    assert result["trace"].count("readjust_plan") == 3
    assert len(result["plans"]["맛집형"]["attempts"]) == 1
    assert len(result["plans"]["명소형"]["attempts"]) == 1


def test_readjust_stops_as_soon_as_budget_met():
    # 가성비형: 생성 900k → 재조정 1회째 700k 로 충족
    def planner(prompt):
        if "초과했습니다" in prompt:
            return make_output(700_000 // 6)
        return make_output(150_000)

    graph, config, result = run(planner)
    attempts = result["plans"]["가성비형"]["attempts"]
    assert len(attempts) == 2
    assert tools.check_budget("가성비형", attempts[-1], TRIP["budget_krw"])["within"]
    assert "pick_closest" not in result["trace"]


def test_each_plan_counted_separately_with_own_target():
    # 모든 안 합계 1,200,000: 세 안 모두 초과 → 안마다 3회씩
    graph, config, result = run(lambda p: make_output(200_000))
    for pt in ("맛집형", "명소형", "가성비형"):
        assert len(result["plans"][pt]["attempts"]) == 4
    assert result["trace"].count("readjust_plan") == 9


def test_closest_attempt_and_warning_after_three_failures():
    per_item = {"맛집형": [1_000_000, 700_000, 900_000, 800_000]}

    def planner(prompt):
        if "맛집" in prompt:
            planner.count += 1
            return make_output(per_item["맛집형"][planner.count - 1])
        return make_output(10_000)

    planner.count = 0
    graph, config, result = run(planner)
    attempts = result["plans"]["맛집형"]["attempts"]
    assert len(attempts) == 4
    best = tools.closest_attempt(attempts, "맛집형", TRIP["budget_krw"])
    assert tools.total_cost(best) == 6 * 700_000  # 초과 금액이 가장 작은 시도
    assert "pick_closest" in result["trace"]
    closest = [r for r in result["tool_results"] if r["tool"] == "closest_attempt"]
    assert closest and closest[0]["result"]["overage"] == 6 * 700_000 - 1_000_000

    result = graph.invoke(Command(resume={"type": "select", "plan": "맛집형"}), config)
    assert tools.total_cost(graph.get_state(config).values["selected_items"]) == 6 * 700_000
    assert "경고" in result["__interrupt__"][0].value["plainAnswer"]
    html = tools.render_report_html("맛집형", best, TRIP["budget_krw"])
    assert "warning" in html and "초과" in html


def test_selected_plan_chat_edit_gets_fresh_three_readjustments():
    over = [
        {"day": 1, "time_slot": "아침", "activity": "고가", "cost_krw": 5_000_000, "transport": "택시", "note": "n"}
    ]
    call = AIMessage(content="", tool_calls=[{"name": "submit_plan_items", "args": {"items": over}, "id": "c1"}])
    # 모든 생성·재조정이 초과 안을 돌려줘 재조정 3회를 모두 소진시킨다
    model = FakeModel(planner=lambda p: {"items": over}, script=[call, AIMessage(content="끝")])
    graph2 = build_graph(model)
    cfg = {"configurable": {"thread_id": "s2"}}
    graph2.invoke({"trip": TRIP}, cfg)
    graph2.invoke(Command(resume={"type": "select", "plan": "명소형"}), cfg)
    result = graph2.invoke(Command(resume={"type": "chat", "text": "비싼 곳으로"}), cfg)
    state = graph2.get_state(cfg).values
    assert len(state["selected_attempts"]) == 4  # 수정 1회마다 새로 3회
    assert result["trace"].count("readjust_selected") == 3
    assert "경고" in result["__interrupt__"][0].value["plainAnswer"]
