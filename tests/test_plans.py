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
    comparer: object = None  # PlanComparison(안별 차별점) 호출에 응답하는 함수. 없으면 응답 없음(폴백 문장 사용)

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self.script.pop(0) if self.script else AIMessage(content="완료")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        if getattr(schema, "__name__", "") == "PlanComparison":  # 안별 차별점 비교 호출은 일정 생성용 planner 와 분리한다
            return RunnableLambda(lambda prompt: self.comparer(prompt) if self.comparer else None)
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
    assert result["trace"][-2:] == ["collect_plans", "compare_plans"]  # await_selection 은 interrupt 로 멈춰 있음
    assert graph.get_state(config).next == ("await_selection",)
    for plan in result["plans"].values():
        assert len(plan["attempts"]) == 1
        assert plan["attempts"][0][0].keys() == {"day", "time_slot", "activity", "cost_krw", "transport", "note", "origin", "arrival", "duration", "depart_time", "arrive_time", "category", "per_person", "unit_cost_krw", "local_unit_price", "local_currency"}


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
         "category": "", "per_person": False, "unit_cost_krw": 0, "local_unit_price": 0, "local_currency": ""}
    ]
    call = AIMessage(content="", tool_calls=[{"name": "submit_plan_items", "args": {"items": new_items}, "id": "c1"}])
    graph, config, result = run(lambda p: make_output(50_000), script=[call, AIMessage(content="끝")])
    result = graph.invoke(Command(resume={"type": "select", "plan": "맛집형"}), config)
    assert result["selected"] == "맛집형" and "__interrupt__" in result
    result = graph.invoke(Command(resume={"type": "chat", "text": "첫 아침을 바꿔줘"}), config)
    strip = lambda i: {k: v for k, v in i.items() if k not in ("depart_time", "arrive_time")}  # 시각은 비어 있으면 코드가 채운다
    assert [strip(i) for i in result["selected_items"]] == [strip(i) for i in new_items]
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
    assert "객실 수(1실, 2명당 1실 올림)와 박 수(1박)" in plan[False] and "평점이 좋은 중상급" in plan[False]
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
        assert "객실 수(2실, 2명당 1실 올림)와 박 수(3박)" in p  # 3명 -> 2실, 4일 -> 3박


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
        assert "식사 항목과 합치지 말고" in p and "per_person=false, unit_cost_krw=객실 1박 요금" in p and "직접 곱하지 마세요" in p


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


def test_note_limit_in_prompts_and_applied_to_generated_plans():
    """3안 생성·재조정 프롬프트가 비고 상한을 알리고, 모델이 길게 써도 저장되는 비고는 상한 이내다."""
    from src.tools import NOTE_MAX_CHARS

    long_item = PlanOutput(items=[{"day": 1, "time_slot": "아침", "activity": "a", "cost_krw": 50_000, "transport": "도보",
                                   "note": "길" * 120}])
    prompts = []

    def planner(prompt):
        prompts.append(prompt)
        return long_item if "초과했습니다" not in prompt else make_output(1)

    trip = {**TRIP, "budget_krw": 10_000}  # 항상 초과 -> 재조정 프롬프트도 만들어진다
    graph = build_graph(FakeModel(planner=planner, script=[]))
    result = graph.invoke({"trip": trip}, {"configurable": {"thread_id": "nc"}})
    assert all(f"{NOTE_MAX_CHARS}자 이내" in p for p in prompts)
    assert any("초과했습니다" in p for p in prompts)
    for plan in result["plans"].values():
        assert len(plan["attempts"][0][0]["note"]) == NOTE_MAX_CHARS


def _comparison(counts):
    """안별 차별점 문장 수를 정한 PlanComparison 을 만든다. counts: {plan_type: 문장 수}"""
    from src.agent import PlanComparison, PlanDifference

    return PlanComparison(plans=[
        PlanDifference(plan_type=pt, difference=" ".join(f"{pt}의 {i + 1}번째 설명입니다." for i in range(n)))
        for pt, n in counts.items()])


def test_compare_plans_runs_between_collect_and_selection_and_feeds_plain_answer():
    """compare_plans 는 collect_plans 뒤 await_selection 앞에서 돌고, 2~3문장 차별점이 plainAnswer 와 상태에 들어간다."""
    from src.tools import difference_ok

    calls = []

    def comparer(prompt):
        calls.append(prompt)
        return _comparison({"맛집형": 2, "명소형": 3, "가성비형": 2})

    graph = build_graph(FakeModel(planner=lambda p: make_output(50_000), script=[], comparer=comparer))
    config = {"configurable": {"thread_id": "cp1"}}
    result = graph.invoke({"trip": TRIP}, config)
    assert result["trace"][-2:] == ["collect_plans", "compare_plans"] and graph.get_state(config).next == ("await_selection",)
    assert len(calls) == 1 and "안별 차별점 비교" in calls[0] and "2~3문장" in calls[0] and "가성비형" in calls[0]
    assert "숫자는 쓰지 마세요" in calls[0]  # 금액·합계는 LLM 이 아니라 코드가 계산해 보여 준다
    assert all(difference_ok(t) for t in result["comparison"].values())
    text = result["__interrupt__"][0].value["plainAnswer"]
    assert "  차별점: 명소형의 1번째 설명입니다. 명소형의 2번째 설명입니다. 명소형의 3번째 설명입니다." in text
    assert text.count("  차별점: ") == 3 and text.count("  주요 일정:") == 3
    assert any(c["tool"] == "compare_plans" for c in result["tool_results"])


def test_compare_plans_retries_once_when_sentence_count_is_wrong():
    """문장 수가 2~3이 아닌 안이 있으면 현재 문장 수·글자 수를 알려 1회만 다시 요청하고, 다시 받은 글을 쓴다."""
    prompts = []

    def comparer(prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return _comparison({"맛집형": 2, "명소형": 5, "가성비형": 3})  # 명소형만 너무 길다
        return _comparison({"맛집형": 1, "명소형": 3, "가성비형": 1})

    graph = build_graph(FakeModel(planner=lambda p: make_output(50_000), script=[], comparer=comparer))
    result = graph.invoke({"trip": TRIP}, {"configurable": {"thread_id": "cp2"}})
    assert len(prompts) == 2 and "이전 응답의 문장 수·글자 수" in prompts[1] and "'명소형': '5문장" in prompts[1]
    assert "맛집형" not in prompts[1].split("이전 응답의")[1].split("각 안의")[0]  # 범위 안인 안은 다시 요청하지 않는다
    assert result["comparison"]["명소형"].count("명소형의") == 3  # 재요청 결과(3문장)를 썼다
    assert result["comparison"]["맛집형"].count("맛집형의") == 2  # 첫 응답을 유지했다


def test_compare_plans_falls_back_when_call_fails_or_stays_out_of_range():
    """호출이 실패하면 폴백으로 대체하고, 재요청 후에도 어긋나면 앞 3문장만 남기거나 폴백으로 2문장이 되게 보강한다."""
    from src.tools import difference_ok

    def boom(prompt):
        raise RuntimeError("bedrock down")

    graph = build_graph(FakeModel(planner=lambda p: make_output(50_000), script=[], comparer=boom))
    result = graph.invoke({"trip": TRIP}, {"configurable": {"thread_id": "cp3"}})
    assert "__interrupt__" in result and not result.get("errors") and set(result["plans"]) == {"맛집형", "명소형", "가성비형"}
    for plan_type, text in result["comparison"].items():
        assert text.startswith(f"{plan_type}의 합계는 ") and difference_ok(text)  # 폴백 문장(2~3문장)

    graph = build_graph(FakeModel(planner=lambda p: make_output(50_000), script=[], comparer=lambda p: _comparison(
        {"맛집형": 1, "명소형": 6, "가성비형": 0})))
    result = graph.invoke({"trip": TRIP}, {"configurable": {"thread_id": "cp4"}})
    comparison = result["comparison"]
    assert comparison["맛집형"].startswith("맛집형의 1번째 설명입니다. 맛집형의 합계는")  # 1문장 + 폴백 1문장
    assert comparison["명소형"].count("명소형의") == 3  # 6문장은 앞 3문장만 남는다
    assert comparison["가성비형"].startswith("가성비형의 합계는") and all(difference_ok(t) for t in comparison.values())


def test_compare_plans_is_skipped_when_only_one_plan_exists():
    """안이 1개만 만들어지면 비교 호출을 하지 않고 comparison 도 만들지 않는다."""
    calls = []

    def planner(prompt):
        if "대표 명소" in prompt or "가성비 중심" in prompt:
            raise ValueError("생성 실패")
        return make_output(50_000)

    graph = build_graph(FakeModel(planner=planner, script=[], comparer=lambda p: calls.append(p)))
    result = graph.invoke({"trip": TRIP}, {"configurable": {"thread_id": "cp5"}})
    assert set(result["plans"]) == {"맛집형"} and calls == [] and "comparison" not in result
    assert result["__interrupt__"][0].value["plainAnswer"].count("  차별점: ") == 1  # 카드는 폴백 문장으로 표시된다


class FakeMapsWithStats:
    """장소 평점·리뷰 수를 모아 둔 가짜 Google Maps(도구 없음)."""

    search_tools: list = []
    route_tools: list = []

    def stats_snapshot(self):
        return {"Momo Paradise Shinjuku": {"rating": 4.7, "reviews": 8200}, "Halal Wagyu Yakiniku": {"rating": 4.4, "reviews": 9000}}


def test_google_place_stats_reach_the_cards_and_filter_restaurants():
    """조사 노드가 모은 평점·리뷰 수가 상태에 담겨, 카드의 주요 일정에 기준을 만족하는 식당만(★ 표시와 함께) 나온다."""
    from src.agent import RegionOutput

    day = lambda d, act, cat, cost: {"day": d, "time_slot": "저녁", "activity": act, "cost_krw": cost, "transport": "도보",  # noqa: E731
                                     "note": "", "category": cat}
    items = PlanOutput(items=[
        day(1, "센소지 관광", "명소", 1000), day(1, "Momo Paradise Shinjuku 저녁", "식당", 40_000),
        day(2, "Halal Wagyu Yakiniku 저녁", "식당", 48_000), day(2, "도쿄타워 전망대", "명소", 15_000),
        day(3, "메이지 신궁 참배", "명소", 0), day(3, "우에노 공원 산책", "명소", 0), day(3, "스카이트리 전망대", "명소", 20_000),
    ])

    def planner(prompt):
        return RegionOutput(overseas=False) if "대한민국 밖" in prompt else items

    graph = build_graph(FakeModel(planner=planner, script=[AIMessage(content="조사 결과")]), maps=FakeMapsWithStats())
    result = graph.invoke({"trip": TRIP}, {"configurable": {"thread_id": "ps"}})
    assert result["place_stats"]["Momo Paradise Shinjuku"] == {"rating": 4.7, "reviews": 8200}
    text = result["__interrupt__"][0].value["plainAnswer"]
    schedule = [l for l in text.splitlines() if l.startswith("  · ")]
    assert any("Momo Paradise Shinjuku 저녁" in l and "★4.7 리뷰 8,200개" in l for l in schedule)  # 기준 충족 식당
    assert not any("Halal" in l for l in schedule)  # 평점 4.4 미달은 제외
    assert sum("★" in l for l in schedule) == 3  # 3안 모두 같은 식당 1개씩


def test_prompts_ask_for_category_on_every_item():
    """일정 생성·재조정 프롬프트가 category(식당/명소/이동/숙박/기타) 표시를 요구한다."""
    prompts = []

    def planner(prompt):
        prompts.append(prompt)
        return make_output(900_000 if "초과했습니다" not in prompt else 1)

    graph = build_graph(FakeModel(planner=planner, script=[]))
    graph.invoke({"trip": {**TRIP, "budget_krw": 10_000}}, {"configurable": {"thread_id": "cat"}})  # 재조정 프롬프트까지 만든다
    assert any("초과했습니다" in p for p in prompts)
    assert all("category" in p and "식당, 명소, 이동, 숙박, 기타" in p for p in prompts)


# ---- 국내 이동: 네이버 지역검색 + Directions 15 (16절) ----

class FakeResp:
    """requests 응답 대용: ok·json 만 쓴다."""

    def __init__(self, data, ok=True):
        self._data, self.ok = data, ok

    def json(self):
        return self._data


class FakeHttp:
    """네이버 API 호출 대용: 주소로 지역검색/길찾기를 구분하고 호출 기록을 남긴다."""

    def __init__(self, places=None, route=None, boom=False):
        self.places, self.route, self.boom, self.calls = places or {}, route, boom, []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append((url, params))
        if self.boom:
            raise RuntimeError("down")
        if "search/v1/local" in url:
            return FakeResp({"items": self.places.get(params["query"], [])})
        return FakeResp(self.route if self.route is not None else {"code": 1, "message": "fail"})


def _naver(http):
    return tools.NaverMaps(("id", "key"), ("id2", "key2"), http=http)


def _place(title, mapx, mapy, address="부산광역시 해운대구 1"):
    return {"title": title, "category": "여행,명소", "address": address, "roadAddress": address, "mapx": str(mapx), "mapy": str(mapy)}


def _route(duration_ms=1_800_000, distance=12_300, toll=1_000, fuel=1_500, taxi=15_000):
    return {"code": 0, "route": {"traoptimal": [{"summary": {
        "distance": distance, "duration": duration_ms, "tollFare": toll, "fuelPrice": fuel, "taxiFare": taxi}}]}}


def test_naver_search_place_converts_coords_and_strips_markup():
    """좌표는 x10^7 정수를 도 단위로 바꾸고, 제목의 태그·엔티티를 지우며, 지역이 다르면 None 이다."""
    http = FakeHttp(places={"해운대 부산": [_place("<b>해운대</b>해수욕장&amp;광장", 1291583542, 351581445)],
                            "경복궁 부산": [_place("경복궁", 1269770162, 375788408, "서울특별시 종로구 사직로 161")]})
    naver = _naver(http)
    found = naver.search_place("해운대 부산", "부산")
    assert found["name"] == "해운대해수욕장&광장"
    assert (round(found["lng"], 4), round(found["lat"], 4)) == (129.1584, 35.1581)
    assert naver.search_place("경복궁 부산", "부산") is None  # 주소에 지역명(부산)이 없다
    assert naver.search_place("없는곳", "부산") is None


def test_naver_drive_route_parses_summary_and_fails_to_none():
    """소요시간(ms)→분, 거리·통행료·연료비·택시요금을 돌려주고, 오류 응답·예외는 None 이다."""
    got = _naver(FakeHttp(route=_route())).drive_route((129.0, 35.0), (129.1, 35.1))
    assert got == {"distance_m": 12_300, "minutes": 30, "toll": 1_000, "fuel": 1_500, "taxi": 15_000}
    assert _naver(FakeHttp(route={"code": 1})).drive_route((1, 2), (3, 4)) is None
    assert _naver(FakeHttp(boom=True)).drive_route((1, 2), (3, 4)) is None


def test_format_minutes():
    assert tools.format_minutes(28) == "약 28분"
    assert tools.format_minutes(125) == "약 2시간 5분"
    assert tools.format_minutes(180) == "약 3시간"


def _kr_items():
    return [
        {"day": 1, "time_slot": "오전", "activity": "해운대 산책", "cost_krw": 0, "transport": "지하철", "note": "a", "category": "명소"},
        {"day": 1, "time_slot": "점심", "activity": "자갈치 이동", "cost_krw": 50_000, "transport": "버스", "note": "b", "category": "이동"},
        {"day": 1, "time_slot": "오후", "activity": "KTX 서울행", "cost_krw": 60_000, "transport": "KTX", "note": "c", "category": "이동"},
        {"day": 2, "time_slot": "오전", "activity": "감천 관람", "cost_krw": 0, "transport": "도보", "note": "d", "category": "명소"},
    ]


def test_apply_domestic_routes_fills_and_prices_only_move_items():
    """같은 날 연속 장소 사이 경로로 출발·도착·소요시간·자동차를 채우고, 이동 항목만 통행료+연료비로 바꾼다."""
    http = FakeHttp(places={
        "A": [_place("해운대", 1291583542, 351581445)], "B": [_place("자갈치", 1290311000, 350967000)],
        "C": [_place("기차역", 1290391000, 351161000)], "D": [_place("감천", 1290104000, 350986000)]}, route=_route())
    out, filled = tools.apply_domestic_routes(_kr_items(), {0: "A", 1: "B", 2: "C", 3: "D"}, _naver(http), "부산")
    assert filled == 1  # 둘째 항목만(첫 항목은 앞 장소 없음, KTX 는 건너뜀, 둘째 날 첫 항목은 앞 장소 없음)
    move = out[1]
    assert (move["origin"], move["arrival"], move["duration"], move["transport"]) == ("해운대", "자갈치", "약 30분", "자동차")
    assert move["cost_krw"] == 2_500 and move["per_person"] is False and "12.3km" in move["note"] and "15,000" in move["note"]
    assert out[2]["transport"] == "KTX" and out[2]["cost_krw"] == 60_000  # 기차는 그대로
    assert out[0]["cost_krw"] == 0 and out[3]["note"] == "d"


def test_apply_domestic_routes_keeps_items_when_everything_fails():
    """장소 검색이나 경로 조회가 실패하면 항목은 원래 값 그대로다."""
    items = _kr_items()
    out, filled = tools.apply_domestic_routes(items, {0: "A", 1: "B"}, _naver(FakeHttp(boom=True)), "부산")
    assert filled == 0 and out == items


class FakeNaver:
    """그래프에 넣는 가짜 네이버: 이름마다 좌표를 주고, 경로는 고정값."""

    def __init__(self):
        self.searches, self.routes = [], []

    def search_place(self, query, region=""):
        self.searches.append(query)
        return {"name": query, "category": "", "address": "", "lng": 129.0 + len(self.searches) / 100, "lat": 35.0}

    def drive_route(self, start, goal):
        self.routes.append((start, goal))
        return {"distance_m": 5_000, "minutes": 15, "toll": 0, "fuel": 700, "taxi": 8_000}


def _naver_planner(overseas, fail_queries=False):
    from src.agent import PlaceQueries, RegionOutput

    def planner(prompt):
        if "대한민국 밖" in prompt:
            return RegionOutput(overseas=overseas)
        if "네이버 지도" in prompt:
            if fail_queries:
                raise RuntimeError("boom")
            return PlaceQueries(items=[{"index": i, "query": f"장소{i}"} for i in range(6)])
        return make_output(50_000)

    return planner


def test_graph_domestic_selection_runs_naver_lookup():
    """국내로 판정되면 선택 후 lookup_domestic_routes 가 돌고 경로가 항목에 채워진다."""
    naver = FakeNaver()
    graph = build_graph(FakeModel(planner=_naver_planner(False), script=[AIMessage(content="목록")]), maps=FakeMaps(), naver=naver)
    config = {"configurable": {"thread_id": "n1"}}
    graph.invoke({"trip": TRIP}, config)
    result = graph.invoke(Command(resume={"type": "select", "plan": "맛집형"}), config)
    assert "lookup_domestic_routes" in result["trace"] and "lookup_routes" not in result["trace"]
    assert len(naver.routes) == 4  # 2일 x 3칸: 날마다 둘째·셋째 칸에서 경로 1번씩
    assert result["selected_items"][1]["transport"] == "자동차" and result["selected_items"][1]["duration"] == "약 15분"


def test_graph_overseas_selection_does_not_call_naver():
    """해외면 기존 lookup_routes(Google)만 돌고 네이버는 부르지 않는다."""
    naver = FakeNaver()
    graph = build_graph(FakeModel(planner=_naver_planner(True), script=[AIMessage(content="목록"), AIMessage(content="끝")]), maps=FakeMaps(), naver=naver)
    config = {"configurable": {"thread_id": "n2"}}
    graph.invoke({"trip": TRIP}, config)
    result = graph.invoke(Command(resume={"type": "select", "plan": "맛집형"}), config)
    assert "lookup_routes" in result["trace"] and "lookup_domestic_routes" not in result["trace"]
    assert naver.searches == [] and naver.routes == []


def test_graph_domestic_lookup_survives_query_failure():
    """검색어 생성이 실패해도 요청은 정상 완료되고 항목은 그대로다."""
    naver = FakeNaver()
    graph = build_graph(FakeModel(planner=_naver_planner(False, fail_queries=True), script=[AIMessage(content="목록")]), maps=FakeMaps(), naver=naver)
    config = {"configurable": {"thread_id": "n3"}}
    graph.invoke({"trip": TRIP}, config)
    result = graph.invoke(Command(resume={"type": "select", "plan": "맛집형"}), config)
    assert "lookup_domestic_routes" in result["trace"] and "generate_guide" in result["trace"]
    assert naver.routes == [] and result["selected_items"][1]["transport"] == "지하철"


def test_generate_prompt_carries_budget_breakdown():
    """예산이 있으면 안 생성 프롬프트에 안별 목표·1인당·1일당 금액이 들어간다."""
    prompts = []

    def planner(prompt):
        prompts.append(prompt)
        return make_output(50_000)

    graph = build_graph(FakeModel(planner=planner, script=[]))
    graph.invoke({"trip": TRIP}, {"configurable": {"thread_id": "bd"}})
    plan_prompts = [p for p in prompts if "일정을" in p]
    assert len(plan_prompts) == 3
    assert all("1인당 약" in p and "1일당 약" in p and "x 2명" in p for p in plan_prompts)
    assert any("1인당 약 400000원" in p for p in plan_prompts if "가성비" in p)  # 가성비형 목표 800,000 / 2명


def test_guide_runs_in_parallel_with_routes_and_report_renders_once():
    """해외 선택 후 경로 조회와 가이드가 같은 단계에서 병렬로 돌고, render_answer 는 한 번만 실행된다."""
    from src.agent import RegionOutput

    def planner(prompt):
        return RegionOutput(overseas=True) if "대한민국 밖" in prompt else make_output(50_000)

    graph = build_graph(FakeModel(planner=planner, script=[AIMessage(content="목록"), AIMessage(content="끝")]), maps=FakeMaps())
    config = {"configurable": {"thread_id": "par1"}}
    graph.invoke({"trip": TRIP}, config)
    result = graph.invoke(Command(resume={"type": "select", "plan": "맛집형"}), config)
    trace = result["trace"]
    assert "lookup_routes" in trace and "generate_guide" in trace
    assert trace.count("render_answer") == 1
    assert trace.index("render_answer") > max(trace.index("lookup_routes"), trace.index("generate_guide"))


class _Resp:
    """requests.Response 흉내: ok 와 json() 만 가진다."""

    def __init__(self, data, ok=True):
        self._data, self.ok = data, ok

    def json(self):
        return self._data


class _Http:
    """요청 URL·params 를 기록하고 정해진 응답(또는 예외)을 돌려주는 가짜 HTTP 클라이언트."""

    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.calls = resp, exc, []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.exc:
            raise self.exc
        return self.resp


def test_frankfurter_rate_to_krw_parses_and_rejects_bad_codes():
    """환율 도구: 통화 1단위의 원화 값을 읽고, 잘못된 코드·실패·미지원 통화는 None, KRW 는 1 로 돌려준다."""
    from src.tools import FrankfurterRates

    http = _Http(_Resp({"base": "SGD", "date": "2026-11-04", "rates": {"KRW": 1020.5}}))
    assert FrankfurterRates(http).rate_to_krw("sgd") == {"currency": "SGD", "krw_per_unit": 1020.5, "date": "2026-11-04"}
    assert http.calls[0][1]["params"] == {"base": "SGD", "symbols": "KRW"}
    assert FrankfurterRates(_Http(_Resp({}, ok=False))).rate_to_krw("VND") is None
    assert FrankfurterRates(_Http(exc=RuntimeError("boom"))).rate_to_krw("USD") is None
    assert FrankfurterRates(_Http()).rate_to_krw("싱가포르달러") is None and FrankfurterRates(_Http()).rate_to_krw("") is None
    assert FrankfurterRates(_Http()).rate_to_krw("KRW")["krw_per_unit"] == 1.0


def test_openweather_groups_3h_entries_into_local_days_without_leaking_key(caplog):
    """날씨 도구: 3시간 예보를 도시 현지 날짜별 최저·최고·강수확률·대표 날씨로 묶는다. 실패 로그에 키가 남지 않는다."""
    from datetime import datetime, timezone

    from src.tools import OpenWeatherForecast

    def ts(day, hour):
        return int(datetime(2026, 11, day, hour, tzinfo=timezone.utc).timestamp())

    def entry(day, hour, tmin, tmax, desc, pop):
        return {"dt": ts(day, hour), "main": {"temp_min": tmin, "temp_max": tmax}, "weather": [{"description": desc}], "pop": pop}

    # 시간대 +8h(싱가포르): UTC 11-22 17:00 은 현지 11-23 01:00 이다
    data = {"city": {"name": "Singapore", "timezone": 28800}, "list": [
        entry(22, 3, 26.2, 29.6, "구름조금", 0.1), entry(22, 6, 27.0, 31.4, "구름조금", 0.3), entry(22, 17, 25.0, 26.0, "비", 0.9)]}
    http = _Http(_Resp(data))
    result = OpenWeatherForecast("SECRET-KEY", http).daily_forecast("Singapore,SG")
    assert result["city"] == "Singapore"
    assert result["days"] == [
        {"date": "2026-11-22", "min": 26, "max": 31, "description": "구름조금", "pop": 30},
        {"date": "2026-11-23", "min": 25, "max": 26, "description": "비", "pop": 90},
    ]
    assert http.calls[0][1]["params"]["q"] == "Singapore,SG" and http.calls[0][1]["params"]["units"] == "metric"
    with caplog.at_level("WARNING"):
        assert OpenWeatherForecast("SECRET-KEY", _Http(exc=RuntimeError("url?appid=SECRET-KEY"))).daily_forecast("x") is None
    assert "SECRET-KEY" not in caplog.text


def test_trip_weather_picks_trip_days_or_falls_back_to_reference():
    """여행 기간과 겹치는 날만 고르고, 겹치는 날이 없으면 in_range=False 로 전체 예보를 참고용으로 둔다."""
    from datetime import date

    from src.tools import trip_weather, weather_text

    forecast = {"city": "Tokyo", "days": [{"date": f"2026-11-0{d}", "min": 10, "max": 18, "description": "맑음", "pop": 0} for d in (5, 6, 7)]}
    inside = trip_weather(forecast, date(2026, 11, 6), date(2026, 11, 7))
    assert inside["in_range"] is True and [d["date"] for d in inside["days"]] == ["2026-11-06", "2026-11-07"]
    outside = trip_weather(forecast, date(2026, 12, 20), date(2026, 12, 22))
    assert outside["in_range"] is False and len(outside["days"]) == 3
    assert trip_weather(None, date(2026, 11, 6), date(2026, 11, 7)) is None
    assert weather_text(inside).splitlines()[0] == "11/6 맑음 10~18℃ 강수확률 0%"


def test_condition_tools_exist_only_for_connected_clients_and_report_errors():
    """연결된 클라이언트의 Tool 만 만들어지고, 조회 실패나 잘못된 날짜는 예외 대신 error 를 돌려준다."""
    from src.tools import make_condition_tools

    class Rates:
        def rate_to_krw(self, currency):
            return {"currency": currency, "krw_per_unit": 9.5, "date": "2026-11-04"} if currency == "JPY" else None

    class Weather:
        def daily_forecast(self, query):
            return None

    assert make_condition_tools() == []
    tools = {t.name: t for t in make_condition_tools(Rates(), Weather())}
    assert set(tools) == {"get_exchange_rate", "get_weather_forecast"}
    assert tools["get_exchange_rate"].invoke({"currency": "JPY"})["krw_per_unit"] == 9.5
    assert "error" in tools["get_exchange_rate"].invoke({"currency": "XXX"})
    assert "error" in tools["get_weather_forecast"].invoke({"city": "Tokyo,JP", "start_date": "2026-11-06", "end_date": "2026-11-07"})
    assert "error" in tools["get_weather_forecast"].invoke({"city": "Tokyo,JP", "start_date": "bad", "end_date": "2026-11-07"})


def _conditions_graph(rates, weather, prompts):
    from src.agent import RegionOutput

    def planner(prompt):
        prompts.append(prompt)
        if "대한민국 밖" in prompt:
            return RegionOutput(overseas=True, currency="SGD", weather_query="Singapore,SG")
        return make_output(50_000)

    return build_graph(FakeModel(planner=planner, script=[]), rates=rates, weather=weather)


class _FakeRates:
    def rate_to_krw(self, currency):
        return {"currency": currency, "krw_per_unit": 1020.5, "date": "2026-11-04"}


class _FakeWeather:
    """여행 시작일(오늘+30일)부터 하루씩 맑음/비 예보를 만든다."""

    def daily_forecast(self, query):
        from datetime import date, timedelta

        start = date.today() + timedelta(days=30)
        return {"city": "Singapore", "days": [
            {"date": (start + timedelta(days=i)).isoformat(), "min": 25, "max": 31, "description": "비" if i else "맑음", "pop": 80 if i else 10}
            for i in range(2)]}


def test_graph_conditions_node_feeds_fx_and_weather_into_prompts_and_report():
    """환율·날씨 Tool 이 있으면 research_conditions 가 먼저 돌고, 값이 생성 프롬프트와 리포트에 들어간다."""
    from src.agent import build_answer

    prompts = []
    graph = _conditions_graph(_FakeRates(), _FakeWeather(), prompts)
    config = {"configurable": {"thread_id": "cond1"}}
    result = graph.invoke({"trip": TRIP}, config)
    assert result["trace"][:2] == ["validate_request", "research_conditions"]
    assert result["overseas"] is True and result["fx"]["currency"] == "SGD" and result["weather"]["in_range"] is True
    assert sum("대한민국 밖" in p for p in prompts) == 1  # 해외 판정 호출은 한 번뿐이다(추가 LLM 호출 없음)
    plan_prompts = [p for p in prompts if "일정을" in p]
    assert len(plan_prompts) == 3
    assert all("1 SGD = 1,020.5원" in p and "강수확률 80%" in p and "실내 명소" in p for p in plan_prompts)
    assert {r["tool"] for r in result["tool_results"]} >= {"get_exchange_rate", "get_weather_forecast"}
    graph.invoke(Command(resume={"type": "select", "plan": "맛집형"}), config)
    html = build_answer(graph.get_state(config).values)["html"]
    assert "환율 (Frankfurter, ECB 기준)" in html and "1 SGD = 1,020.5원" in html
    prep = html.split("class='panel p3'>")[1]  # 날씨는 일정표가 아니라 준비물·유의 사항 탭에 요약으로 들어간다
    assert "날씨 요약 — Singapore (OpenWeatherMap)" in prep and "비 가능성이 높은 날: " in prep and "(80%)" in prep and "우산·우비" in prep
    assert "날씨 요약" not in html.split("class='panel p3'>")[0]


def test_graph_conditions_failure_does_not_stop_plan_generation():
    """환율·날씨 조회가 모두 실패해도 일정 생성은 정상 진행되고 프롬프트에 환율·날씨 문구가 없다."""

    class Fail:
        def rate_to_krw(self, currency):
            return None

        def daily_forecast(self, query):
            return None

    prompts = []
    graph = _conditions_graph(Fail(), Fail(), prompts)
    result = graph.invoke({"trip": TRIP}, {"configurable": {"thread_id": "cond2"}})
    assert set(result["plans"]) == {"맛집형", "명소형", "가성비형"} and not result.get("fx") and not result.get("weather")
    assert not any("환율:" in p or "날씨 예보" in p for p in prompts)


def test_graph_domestic_destination_skips_exchange_rate():
    """국내 여행지는 환율을 조회하지 않고 날씨만 조회한다."""
    from src.agent import RegionOutput

    class CountingRates(_FakeRates):
        calls = 0

        def rate_to_krw(self, currency):
            CountingRates.calls += 1
            return super().rate_to_krw(currency)

    def planner(prompt):
        return RegionOutput(overseas=False, currency="KRW", weather_query="Busan,KR") if "대한민국 밖" in prompt else make_output(50_000)

    graph = build_graph(FakeModel(planner=planner, script=[]), rates=CountingRates(), weather=_FakeWeather())
    result = graph.invoke({"trip": TRIP}, {"configurable": {"thread_id": "cond3"}})
    assert CountingRates.calls == 0 and not result.get("fx") and result["weather"]["city"] == "Singapore"


# ---- 일정 생성 시간 초과 대응: 재시도·부분 성공 ----
def _read_timeout():
    from botocore.exceptions import ReadTimeoutError

    return ReadTimeoutError(endpoint_url="https://bedrock-runtime.test/converse")


def test_is_retryable_classifies_errors():
    from botocore.exceptions import ClientError

    from src.agent import is_retryable

    throttle = ClientError({"Error": {"Code": "ThrottlingException", "Message": "x"}}, "Converse")
    denied = ClientError({"Error": {"Code": "AccessDeniedException", "Message": "x"}}, "Converse")
    assert is_retryable(_read_timeout()) and is_retryable(throttle)
    assert not is_retryable(denied) and not is_retryable(ValueError("형식 오류"))


def test_call_with_retry_recovers_then_gives_up():
    from src.agent import _call_with_retry

    calls, waits = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise _read_timeout()
        return "ok"

    assert _call_with_retry(flaky, sleep=waits.append) == "ok" and len(calls) == 3 and waits == [2.0, 6.0]
    with pytest.raises(type(_read_timeout())):
        _call_with_retry(lambda: (_ for _ in ()).throw(_read_timeout()), sleep=lambda s: None)
    with pytest.raises(ValueError):  # 재시도 대상이 아니면 바로 올린다
        _call_with_retry(lambda: (_ for _ in ()).throw(ValueError("x")), sleep=waits.append)


def test_plan_that_keeps_timing_out_is_dropped_and_others_continue(monkeypatch):
    from src.agent import build_answer

    monkeypatch.setattr("time.sleep", lambda s: None)

    def planner(prompt):
        if "맛집과 식사 경험" in prompt:  # 맛집형만 계속 시간 초과
            raise _read_timeout()
        return make_output(50_000)

    graph, config, result = run(planner)
    assert set(result["plans"]) == {"명소형", "가성비형"}
    assert "__interrupt__" in result  # 남은 안으로 선택 단계까지 진행
    plain = build_answer(graph.get_state(config).values)["plainAnswer"]
    assert plain.startswith("안내: 맛집형 안은 생성하지 못해 제외했습니다")


def _wiki_page(title, thumb="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/x.jpg/480px-x.jpg", index=1):
    page = {"title": title, "index": index}
    if thumb:
        page["thumbnail"] = {"source": thumb}
    return page


def _wiki_http(*pages_by_call):
    """호출 순서대로 검색 결과(문서 목록)를 돌려주는 가짜 HTTP. 마지막 결과를 반복한다."""
    class Http:
        def __init__(self):
            self.calls = []

        def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            pages = pages_by_call[min(len(self.calls) - 1, len(pages_by_call) - 1)]
            return _Resp({"query": {"pages": pages}})

    return Http()


def test_image_title_match_and_safe_url_guards():
    """문서 제목이 장소 이름과 같은 대상일 때만 이미지를 쓰고, 위키미디어 https 주소만 허용한다."""
    from src.tools import image_title_matches, safe_image_url

    assert image_title_matches("센소지", "센소지(아사쿠사)") and image_title_matches("도쿄 스카이트리", "도쿄 스카이트리 전망대")
    assert image_title_matches("센소지 (도쿄)", "센소지")
    assert not image_title_matches("도쿄", "오사카 성") and not image_title_matches("", "센소지") and not image_title_matches("센소지", "")
    ok = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/x.jpg/480px-x.jpg"
    assert safe_image_url(ok) == ok and safe_image_url("https://thumb.wikimedia.org/x.png") != ""
    assert safe_image_url("http://upload.wikimedia.org/x.jpg") == "" and safe_image_url("https://evil.example.com/x.jpg") == ""
    assert safe_image_url("https://upload.wikimedia.org.evil.com/x.jpg") == "" and safe_image_url("javascript:alert(1)") == ""
    assert safe_image_url('https://upload.wikimedia.org/x.jpg"onerror="x') == "" and safe_image_url(None) == ""


def test_wikipedia_images_find_picks_matching_page_and_skips_wrong_or_imageless():
    """맨 위 결과가 다른 대상이면 건너뛰고, 이미지가 없거나 실패하면 None 이다. 지역명을 붙여 한 번 더 찾는다."""
    from src.tools import WikipediaImages

    http = _wiki_http([_wiki_page("도쿄", index=1), _wiki_page("센소지", index=2)])
    found = WikipediaImages(http).find("센소지")
    assert found["title"] == "센소지" and found["url"].startswith("https://upload.wikimedia.org/")
    assert found["page"] == "https://ko.wikipedia.org/wiki/센소지" and len(http.calls) == 1
    assert http.calls[0][1]["headers"]["User-Agent"] and http.calls[0][1]["params"]["gsrsearch"] == "센소지"
    assert WikipediaImages(_wiki_http([_wiki_page("도쿄")])).find("오사카 성") is None  # 엉뚱한 문서의 사진은 쓰지 않는다
    assert WikipediaImages(_wiki_http([_wiki_page("센소지", thumb=None)])).find("센소지") is None
    assert WikipediaImages(_wiki_http([_wiki_page("센소지", thumb="https://evil.example.com/x.jpg")])).find("센소지") is None
    retry = _wiki_http([], [_wiki_page("마리나 베이 샌즈")])
    assert WikipediaImages(retry).find("마리나 베이 샌즈", "싱가포르")["title"] == "마리나 베이 샌즈"
    assert retry.calls[1][1]["params"]["gsrsearch"] == "싱가포르 마리나 베이 샌즈"
    assert WikipediaImages(_Http(_Resp({}, ok=False))).find("센소지") is None
    assert WikipediaImages(_Http(exc=RuntimeError("boom"))).find("센소지") is None


def test_attach_place_images_only_for_sights_and_survives_failures():
    """명소에만 이미지를 붙이고(맛집 제외), 하나가 실패해도 나머지는 붙으며 원본 가이드는 바뀌지 않는다."""
    from src.tools import attach_place_images

    class Finder:
        def find(self, name, region=""):
            if name == "실패":
                raise RuntimeError("boom")
            if name == "없음":
                return None
            return {"url": "https://upload.wikimedia.org/a.jpg", "title": name, "page": "p"}

    guide = {"places": [
        {"kind": "명소", "name": "센소지", "area": "a", "description": "d", "tip": "t"},
        {"kind": "맛집", "name": "이치란", "area": "a", "description": "d", "tip": "t"},
        {"kind": "명소", "name": "실패", "area": "a", "description": "d", "tip": "t"},
        {"kind": "명소", "name": "없음", "area": "a", "description": "d", "tip": "t"},
        {"kind": "명소", "name": "스카이트리", "area": "a", "description": "d", "tip": "t"},
    ], "packing": ["x"]}
    new, count = attach_place_images(guide, Finder(), "도쿄")
    images = {p["name"]: "image" in p for p in new["places"]}
    assert images == {"센소지": True, "이치란": False, "실패": False, "없음": False, "스카이트리": True} and count == 2
    assert "image" not in guide["places"][0] and new["packing"] == ["x"]  # 원본은 그대로
    assert attach_place_images(None, Finder()) == (None, 0) and attach_place_images({"places": []}, Finder())[1] == 0


def test_graph_guide_gets_sight_images_and_image_failure_does_not_break_report():
    """이미지 조회기가 연결되면 가이드의 명소에 이미지가 붙어 리포트 2번째 탭에 나오고, 조회가 모두 실패해도 리포트는 만들어진다."""
    from src.agent import GuideOutput, build_answer

    def planner(prompt):
        if "여행 가이드를 만드세요" in prompt:
            return GuideOutput(
                places=[{"kind": "명소", "name": "센소지", "area": "아사쿠사", "description": "사찰", "tip": "t"},
                        {"kind": "맛집", "name": "이치란", "area": "신주쿠", "description": "라멘", "tip": "t"}],
                packing=["우산"], cautions=[], destination_notes=[])
        return make_output(50_000)

    class Images:
        def __init__(self, fail=False):
            self.fail, self.names = fail, []

        def find(self, name, region=""):
            self.names.append((name, region))
            if self.fail:
                raise RuntimeError("boom")
            return {"url": "https://upload.wikimedia.org/a.jpg", "title": name, "page": "p"}

    images = Images()
    graph = build_graph(FakeModel(planner=planner, script=[]), images=images)
    config = {"configurable": {"thread_id": "img1"}}
    graph.invoke({"trip": TRIP}, config)
    result = graph.invoke(Command(resume={"type": "select", "plan": "맛집형"}), config)
    assert images.names == [("센소지", "도쿄")]  # 맛집은 조회하지 않고, 지역은 여행지를 쓴다
    assert {"tool": "wikipedia_images", "result": {"found": 1, "total": 1}} in result["tool_results"]
    assert build_answer(graph.get_state(config).values)["html"].count("<img ") == 1

    bad = build_graph(FakeModel(planner=planner, script=[]), images=Images(fail=True))
    config2 = {"configurable": {"thread_id": "img2"}}
    bad.invoke({"trip": TRIP}, config2)
    bad.invoke(Command(resume={"type": "select", "plan": "맛집형"}), config2)
    html = build_answer(bad.get_state(config2).values)["html"]
    assert "upload.wikimedia.org" not in html and "센소지" in html  # 조회가 모두 실패해도 리포트는 만들어지고
    assert html.count("<img ") == 1 and "대표 이미지 없음" in html  # 명소는 기본 이미지로 표시된다


def test_zero_cost_plan_cannot_meet_tiny_budget_and_ends_with_closest_attempt():
    """LLM 이 예산을 맞추려고 숙박·식사를 0원으로 써도 코드가 최소 단가를 적용해 '예산 충족'이 되지 않는다(3라운드 TQ-09 재현)."""
    from src.tools import check_budget

    def zero_plan(prompt):
        base = {"transport": "도보", "note": ""}
        return PlanOutput(items=[
            {**base, "day": 1, "time_slot": "아침", "activity": "점심 식사", "cost_krw": 0, "category": "식당"},
            {**base, "day": 1, "time_slot": "저녁", "activity": "숙소 체크인: A호텔", "cost_krw": 0, "category": "숙박"},
            {**base, "day": 2, "time_slot": "아침", "activity": "명소 관람", "cost_krw": 0, "category": "명소"},
        ])

    trip = {"destination": "제주", "days": 2, "people": 2, "budget_krw": 10_000}
    graph = build_graph(FakeModel(planner=zero_plan))
    result = graph.invoke({"trip": trip}, {"configurable": {"thread_id": "zero"}})
    # 최소 단가로 확정된 비용만으로 이미 목표를 넘으므로 재조정을 반복하지 않고 바로 근접 안을 고른다
    assert "pick_closest" in result["trace"] and result["trace"].count("readjust_plan") == 0
    for plan_type, plan in result["plans"].items():
        last = plan["attempts"][-1]
        assert sum(i["cost_krw"] for i in last) == 30_000 + 3_000 * 2  # 숙박 최소 1실 x 1박 + 식사 최소 1인 x 2명
        assert not check_budget(plan_type, last, 10_000)["within"]


class FakeQuotePrices:
    """마이리얼트립 MCP 대신 쓰는 가짜: 조회 도구는 없고, quote_tools 가 항공·숙소 구조화 값을 요청별 sink 에 채운다."""

    price_tools: list = []

    def quote_tools(self, sink):
        sink["flights"] = [{"airline": "제주항공", "direct": True, "price_krw": 337_800,
                            "out": {"date": "2026-12-06", "depart": "09:50", "arrive": "12:15", "no": "7C1183"},
                            "back": {"date": "2026-12-08", "depart": "14:00", "arrive": "17:05", "no": "7C1104"}}]
        sink["stays"] = [{"name": "토부 호텔 레밴트 도쿄", "price_krw": 181_584}]
        return []


def test_quoted_flight_and_lodging_prices_reach_every_plan_and_chat_edit():
    """조회된 항공·숙소 가격은 LLM 이 다른 금액을 써도 3안 모두와 대화 수정 뒤에도 코드가 조회값으로 확정한다."""
    from src.agent import RegionOutput

    def base():
        out = make_output(10_000)
        out.items[0] = out.items[0].model_copy(update={"activity": "서울→도쿄 항공", "category": "이동", "per_person": True, "unit_cost_krw": 900_000})
        out.items[2] = out.items[2].model_copy(update={"activity": "숙소 체크인: 토부 호텔 레밴트 도쿄", "category": "숙박", "unit_cost_krw": 400_000})
        out.items[5] = out.items[5].model_copy(update={"activity": "도쿄→서울 귀국 항공", "category": "이동"})
        return out

    trip = {"destination": "도쿄", "days": 2, "people": 2, "budget_krw": None}
    planner = lambda p: RegionOutput(overseas=True) if "대한민국 밖" in p else base()
    graph = build_graph(FakeModel(planner=planner, script=[AIMessage(content="목록")]), prices=FakeQuotePrices())
    config = {"configurable": {"thread_id": "q1"}}
    result = graph.invoke({"trip": trip}, config)
    assert result["price_quotes"]["flights"][0]["price_krw"] == 337_800
    for plan in result["plans"].values():
        flight, hotel = plan["attempts"][-1][0], plan["attempts"][-1][2]
        assert (flight["unit_cost_krw"], flight["cost_krw"]) == (337_800, 675_600)
        assert (hotel["unit_cost_krw"], hotel["cost_krw"]) == (181_584, 181_584)  # 2일 여행 = 1박, 2명 = 1실
        assert flight["depart_time"] == "09:50" and "7C1183" in flight["note"]
