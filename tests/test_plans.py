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

TRIP = {"destination": "도쿄", "days": 2, "people": 2, "budget_krw": 1_000_000}


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
        assert plan["attempts"][0][0].keys() == {"day", "time_slot", "activity", "cost_krw", "transport", "note", "origin", "arrival", "duration", "depart_time", "arrive_time", "per_person", "unit_cost_krw"}


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
        {"day": 1, "time_slot": "아침", "activity": "수정", "cost_krw": 10_000, "transport": "도보", "note": "n",
         "origin": "", "arrival": "", "duration": "", "depart_time": "", "arrive_time": "",
         "per_person": False, "unit_cost_krw": 0}
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


class FakeMaps:
    """MCP 도구 대신 쓰는 가짜(도구 없음): 연동 경로(장소 조사·경로 조회 노드)만 검증한다."""

    search_tools: list = []
    route_tools: list = []


def test_mcp_mode_researches_places_and_grounds_prompts():
    """maps 가 있으면 research_places 노드가 먼저 돌고, 조사 결과가 일정 생성 프롬프트에 들어간다."""
    from src.agent import RegionOutput

    prompts = []

    def planner(prompt):
        prompts.append(prompt)
        return RegionOutput(overseas=False) if "대한민국 밖" in prompt else make_output(50_000)

    graph = build_graph(FakeModel(planner=planner, script=[AIMessage(content="맛집A, 명소B")]), maps=FakeMaps())
    config = {"configurable": {"thread_id": "m1"}}
    result = graph.invoke({"trip": TRIP}, config)
    assert result["trace"][:2] == ["validate_request", "research_places"]
    assert result["overseas"] is False and result["research"] == "맛집A, 명소B"
    plan_prompts = [p for p in prompts if "일정을" in p]
    assert len(plan_prompts) == 3 and all("Google Maps" in p and "맛집A" in p for p in plan_prompts)
    selected = graph.invoke(Command(resume={"type": "select", "plan": "맛집형"}), config)
    assert "lookup_routes" not in selected["trace"] and "generate_guide" in selected["trace"]  # 국내는 경로 조회 없음


def test_mcp_mode_looks_up_routes_only_for_overseas():
    """해외로 판정되면 선택 후 lookup_routes 가 돌아 경로 조회 결과로 항목을 채우고 비용은 유지한다."""
    from src.agent import RegionOutput

    def planner(prompt):
        return RegionOutput(overseas=True) if "대한민국 밖" in prompt else make_output(50_000)

    routed = [{"day": 1, "time_slot": "아침", "activity": "1-아침", "cost_krw": 1, "transport": "지하철",
               "note": "추정", "origin": "호텔", "arrival": "센소지", "duration": "약 20분"}] + [
        {"day": d, "time_slot": s, "activity": f"{d}-{s}", "cost_krw": 1, "transport": "도보", "note": "x"}
        for d in (1, 2) for s in ("아침", "오후", "저녁") if (d, s) != (1, "아침")]
    call = AIMessage(content="", tool_calls=[{"name": "submit_plan_items", "args": {"items": routed}, "id": "r1"}])
    graph = build_graph(FakeModel(planner=planner, script=[AIMessage(content="목록"), call, AIMessage(content="끝")]), maps=FakeMaps())
    config = {"configurable": {"thread_id": "m2"}}
    graph.invoke({"trip": TRIP}, config)
    result = graph.invoke(Command(resume={"type": "select", "plan": "맛집형"}), config)
    assert "lookup_routes" in result["trace"]
    first = result["selected_items"][0]
    assert first["origin"] == "호텔" and first["arrival"] == "센소지" and first["duration"] == "약 20분"
    assert all(i["cost_krw"] == 50_000 for i in result["selected_items"])  # 비용은 조회로 바뀌지 않는다


class FakePrices:
    """마이리얼트립 MCP 대신 쓰는 가짜(도구 없음)."""

    price_tools: list = []


def test_prices_mode_adds_research_prices_and_grounds_prompts():
    """prices 가 있으면 research_prices 노드가 돌고, 조회 가격이 일정 생성 프롬프트와 리포트에 들어간다."""
    from src.agent import RegionOutput

    prompts = []

    def planner(prompt):
        prompts.append(prompt)
        return RegionOutput(overseas=True) if "대한민국 밖" in prompt else make_output(50_000)

    graph = build_graph(FakeModel(planner=planner, script=[AIMessage(content="도쿄 디즈니 48,401원")]), prices=FakePrices())
    config = {"configurable": {"thread_id": "p1"}}
    result = graph.invoke({"trip": TRIP}, config)
    assert result["trace"][:2] == ["validate_request", "research_prices"]
    assert result["overseas"] is True and "48,401원" in result["price_research"]
    plan_prompts = [p for p in prompts if "일정을" in p]
    assert len(plan_prompts) == 3 and all("마이리얼트립" in p and "48,401원" in p and "1인 기준" in p for p in plan_prompts)
    selected = graph.invoke(Command(resume={"type": "select", "plan": "맛집형"}), config)
    from src.agent import build_answer

    assert "참고 가격" in build_answer(graph.get_state(config).values)["html"]
    assert "lookup_routes" not in selected["trace"]  # Google Maps 가 없으면 경로 조회 노드 없음


def test_both_mcp_modes_chain_places_then_prices():
    """maps 와 prices 가 모두 있으면 장소 조사 다음에 가격 조사가 이어진 뒤 3안이 생성된다."""
    from src.agent import RegionOutput

    def planner(prompt):
        return RegionOutput(overseas=False) if "대한민국 밖" in prompt else make_output(50_000)

    graph = build_graph(
        FakeModel(planner=planner, script=[AIMessage(content="맛집A"), AIMessage(content="입장권 가격")]),
        maps=FakeMaps(), prices=FakePrices(),
    )
    result = graph.invoke({"trip": TRIP}, {"configurable": {"thread_id": "p2"}})
    assert result["trace"][:3] == ["validate_request", "research_places", "research_prices"]
    assert result["research"] == "맛집A" and result["price_research"] == "입장권 가격"


def test_prices_mode_tells_plans_to_include_lodging_cost():
    """가격 조사 결과가 있으면 안 종류별로 숙소 선택 기준과 총 숙박비 반영 지시가 프롬프트에 들어간다."""
    from src.agent import RegionOutput

    prompts = []

    def planner(prompt):
        prompts.append(prompt)
        return RegionOutput(overseas=False) if "대한민국 밖" in prompt else make_output(50_000)

    graph = build_graph(FakeModel(planner=planner, script=[AIMessage(content="숙소 A 100,000원/박")]), prices=FakePrices())
    graph.invoke({"trip": TRIP}, {"configurable": {"thread_id": "p3"}})  # TRIP 은 2일 = 1박
    plan = {("가성비" in p): p for p in prompts if "일정을" in p}
    assert "객실 1실(2명당 1실, 홀수는 올림) x 1박" in plan[False] and "평점이 좋은 중상급" in plan[False]
    assert "가장 저렴한 곳" in plan[True] and "숙소 체크인" in plan[True]


def _one_item(unit: int, per_person: bool = True, cost: int = 1) -> PlanOutput:
    """1인 가격(unit)만 맞게 쓰고 전체 비용(cost)은 일부러 틀리게 쓴 항목 1개짜리 출력."""
    return PlanOutput(items=[{"day": 1, "time_slot": "아침", "activity": "입장권", "cost_krw": cost, "transport": "도보",
                              "note": "", "per_person": per_person, "unit_cost_krw": unit}])


def test_code_multiplies_unit_price_by_people_over_llm_total():
    """3안 생성 후 per_person 항목의 전체 비용은 LLM 값이 아니라 코드의 1인 가격 x 인원이다."""
    graph, config, result = run(lambda p: _one_item(13_000))
    for plan in result["plans"].values():
        assert plan["attempts"][0][0]["cost_krw"] == 26_000  # TRIP 은 2명
    assert "2명" in result["__interrupt__"][0].value["plainAnswer"]


def test_readjust_lowers_unit_price_and_code_multiplies_again():
    """예산 초과 재조정 후에도 1인 가격 x 인원 계산이 적용되고 예산 판정은 곱셈 결과로 한다."""
    trip = {**TRIP, "budget_krw": 100_000}
    calls = []

    def planner(prompt):
        calls.append(prompt)
        return _one_item(30_000) if "초과했습니다" in prompt and "여행 가이드" not in prompt else _one_item(60_000)

    graph = build_graph(FakeModel(planner=planner, script=[]))
    result = graph.invoke({"trip": trip}, {"configurable": {"thread_id": "h1"}})
    assert any("1인 60000원 x 2명 = 120000원" in p for p in calls)  # 재조정 프롬프트가 공식을 보여준다
    for plan in result["plans"].values():
        assert plan["attempts"][0][0]["cost_krw"] == 120_000 and plan["attempts"][-1][0]["cost_krw"] == 60_000
        assert tools.check_budget(plan["plan_type"], plan["attempts"][0], 100_000)["within"] is False  # 곱셈 후 120,000원은 초과
        assert tools.check_budget(plan["plan_type"], plan["attempts"][-1], 100_000)["within"] is True  # 재조정 후 60,000원은 충족


def test_chat_edit_submission_is_multiplied_by_code():
    """대화 수정으로 제출된 per_person 항목도 코드가 인원을 곱해 전체 비용을 계산한다."""
    edited = [{"day": 1, "time_slot": "아침", "activity": "스시", "cost_krw": 1, "transport": "도보", "note": "",
               "per_person": True, "unit_cost_krw": 10_000}]
    call = AIMessage(content="", tool_calls=[{"name": "submit_plan_items", "args": {"items": edited}, "id": "e1"}])
    graph, config, _ = run(lambda p: make_output(50_000), script=[call, AIMessage(content="끝")])
    graph.invoke(Command(resume={"type": "select", "plan": "맛집형"}), config)
    result = graph.invoke(Command(resume={"type": "chat", "text": "스시로 바꿔줘"}), config)
    assert result["selected_items"][0]["cost_krw"] == 20_000


def test_no_budget_generates_three_plans_without_readjust_or_limit():
    """예산이 없으면 금액 상한 없이 3안을 만들고 재조정 노드가 한 번도 돌지 않는다."""
    trip = {"destination": "도쿄", "days": 2, "people": 2, "budget_krw": None}
    prompts = []

    def planner(prompt):
        prompts.append(prompt)
        return make_output(900_000)  # 합계 540만원: 예산이 있었다면 초과

    graph = build_graph(FakeModel(planner=planner, script=[]))
    config = {"configurable": {"thread_id": "nb"}}
    result = graph.invoke({"trip": trip}, config)
    assert set(result["plans"]) == {"맛집형", "명소형", "가성비형"}
    assert not any(t.startswith("readjust") for t in result["trace"]) and not result.get("errors")
    assert all("예산 제한이 없으므로" in p for p in prompts) and not any("목표 총액" in p for p in prompts)
    text = result["__interrupt__"][0].value["plainAnswer"]
    assert "예산 제한 없음" in text and "미충족" not in text
    selected = graph.invoke(Command(resume={"type": "select", "plan": "맛집형"}), config)
    assert "readjust_selected" not in selected["trace"]


@pytest.mark.parametrize("trip", [
    {"destination": "도쿄", "days": 2, "budget_krw": 1000},
    {"destination": "도쿄", "days": 2, "people": 0, "budget_krw": 1000},
    {"destination": "도쿄", "days": 2, "people": 2, "budget_krw": -5},
])
def test_people_required_and_budget_must_be_positive_if_given(trip):
    """인원은 필수(1 이상), 예산은 생략하거나 0 초과여야 한다."""
    graph = build_graph(FakeModel(planner=lambda p: make_output(1)))
    result = graph.invoke({"trip": trip}, {"configurable": {"thread_id": "pp"}})
    assert result["errors"] and not result.get("plans")


def test_prompts_have_cost_flight_and_lodging_rules_even_without_mcp():
    """MCP 없이(MOCKDATA_USE=True 경로)도 1인 단위 표시, 항공권 항목, 숙박비(2명당 1실) 지시가 들어간다."""
    prompts = []

    def planner(prompt):
        prompts.append(prompt)
        return make_output(50_000)

    graph = build_graph(FakeModel(planner=planner, script=[]))
    graph.invoke({"trip": {**TRIP, "days": 4, "people": 3}}, {"configurable": {"thread_id": "r1"}})
    for p in prompts:
        assert "per_person=true" in p and "서울→도쿄 항공" in p and "여행 인원 3명" in p
        assert "객실 2실(2명당 1실, 홀수는 올림) x 3박" in p  # 3명 -> 2실, 4일 -> 3박


def test_day_trip_has_no_lodging():
    """여행 기간이 1일이면 숙박 항목·숙박비 지시가 없다."""
    prompts = []

    def planner(prompt):
        prompts.append(prompt)
        return make_output(10_000, days=1)

    graph = build_graph(FakeModel(planner=planner, script=[]))
    graph.invoke({"trip": {**TRIP, "days": 1}}, {"configurable": {"thread_id": "d1"}})
    assert prompts and all("당일치기 여행이므로 숙박" in p and "숙박비:" not in p for p in prompts)


def test_lodging_is_a_separate_shared_cost_item_in_prompt():
    """숙박비는 식사와 합치지 않고 별도의 공용 비용(per_person=false) 항목으로 만들라고 지시한다."""
    prompts = []

    def planner(prompt):
        prompts.append(prompt)
        return make_output(50_000)

    graph = build_graph(FakeModel(planner=planner, script=[]))
    graph.invoke({"trip": TRIP}, {"configurable": {"thread_id": "lg"}})
    for p in prompts:
        assert "식사 항목과 합치지 말고" in p and "per_person=false, cost_krw=총 숙박비" in p


def test_prompts_use_origin_and_ask_for_return_flight_and_times():
    """출발지가 있으면 출발지→목적지 가는 편과 목적지→출발지 귀국 편(날짜·시각)을, 없으면 서울을 쓰고 모든 항목에 시각을 요구한다."""
    seen = {}

    for origin in (None, "부산"):
        prompts = []

        def planner(prompt, prompts=prompts):
            prompts.append(prompt)
            return make_output(50_000)

        graph = build_graph(FakeModel(planner=planner, script=[]))
        graph.invoke({"trip": {**TRIP, "origin": origin}}, {"configurable": {"thread_id": f"o-{origin}"}})
        seen[origin] = prompts
    for p in seen[None]:
        assert "출발지는 서울입니다" in p and "'서울→도쿄 항공'" in p and "'도쿄→서울 귀국 항공'" in p
    for p in seen["부산"]:
        assert "출발지는 부산입니다" in p and "'부산→도쿄 항공'" in p and "'도쿄→부산 귀국 항공'" in p
        assert "왕복 요금에 포함" in p and "오는 편의 날짜" in p
        assert "depart_time" in p and "arrive_time" in p and "HH:MM" in p


def test_route_lookup_keeps_planned_times():
    """경로 조회로 항목을 다시 제출해도 계획한 출발·도착 시각은 유지된다."""
    from src.agent import RegionOutput

    def planner(prompt):
        return RegionOutput(overseas=True) if "대한민국 밖" in prompt else PlanOutput(items=[
            {"day": 1, "time_slot": "아침", "activity": "항공", "cost_krw": 1, "transport": "비행기", "note": "",
             "depart_time": "07:10", "arrive_time": "09:15"}])

    submitted = [{"day": 1, "time_slot": "아침", "activity": "항공", "cost_krw": 9, "transport": "비행기", "note": "",
                  "depart_time": "", "arrive_time": "", "duration": "약 2시간"}]
    call = AIMessage(content="", tool_calls=[{"name": "submit_plan_items", "args": {"items": submitted}, "id": "r2"}])
    graph = build_graph(FakeModel(planner=planner, script=[AIMessage(content="목록"), call, AIMessage(content="끝")]), maps=FakeMaps())
    config = {"configurable": {"thread_id": "tm"}}
    graph.invoke({"trip": TRIP}, config)
    result = graph.invoke(Command(resume={"type": "select", "plan": "맛집형"}), config)
    first = result["selected_items"][0]
    assert first["depart_time"] == "07:10" and first["arrive_time"] == "09:15" and first["duration"] == "약 2시간"


class RecordingModel(FakeModel):
    """에이전트에 전달된 메시지(사용자 요청 문장)를 기록하는 가짜 모델."""

    seen: list = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(" ".join(str(getattr(m, "content", "")) for m in messages))
        return super()._generate(messages, stop, run_manager, **kwargs)


def test_mcp_price_research_uses_input_dates_or_d_plus_30():
    """날짜를 명시하면 그 날짜로(가는 날·오는 날, 체크인·체크아웃), 없으면 오늘+30일로 항공·숙소를 조회하라고 지시한다."""
    from datetime import date, timedelta

    from src.agent import RegionOutput

    def planner(prompt):
        return RegionOutput(overseas=True) if "대한민국 밖" in prompt else make_output(50_000)

    def research_message(trip, thread):
        model = RecordingModel(planner=planner, script=[AIMessage(content="조회 결과")], seen=[])
        graph = build_graph(model, prices=FakePrices())
        graph.invoke({"trip": trip}, {"configurable": {"thread_id": thread}})
        return next(m for m in model.seen if "searchStays" in m)

    msg = research_message({**TRIP, "days": 3, "start_date": "2099-01-10"}, "dt1")
    assert "departDate=2099-01-10" in msg and "returnDate=2099-01-12" in msg
    assert "checkIn=2099-01-10" in msg and "checkOut=2099-01-12" in msg
    start = date.today() + timedelta(days=30)
    msg = research_message({**TRIP, "days": 3}, "dt2")
    assert f"departDate={start}" in msg and f"checkIn={start}" in msg and f"checkOut={start + timedelta(days=2)}" in msg
