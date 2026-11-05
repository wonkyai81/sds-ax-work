"""예산 계산 도구 테스트: 목표 예산, 합계, 판정, 가장 근접한 시도 선택."""
from src.tools import (
    check_budget,
    closest_attempt,
    final_items,
    needs_readjust,
    readjust_count,
    target_budget,
    total_cost,
)


def _items(*costs):
    """비용 목록으로 테스트용 항목을 만든다."""
    return [
        {"day": 1, "time_slot": "아침", "activity": f"a{i}", "cost_krw": c, "transport": "도보", "note": ""}
        for i, c in enumerate(costs)
    ]


def test_target_budget():
    """맛집형·명소형은 예산 그대로, 가성비형은 80%."""
    assert target_budget("맛집형", 100000) == 100000
    assert target_budget("명소형", 100000) == 100000
    assert target_budget("가성비형", 100000) == 80000


def test_total_cost():
    """합계 계산과 빈 목록."""
    assert total_cost(_items(1000, 2500, 300)) == 3800
    assert total_cost([]) == 0


def test_check_budget_within_over_boundary():
    """충족, 초과, 경계값(합계 = 목표는 충족)."""
    assert check_budget("맛집형", _items(60000, 30000), 100000) == {"within": True, "total": 90000, "overage": 0}
    assert check_budget("맛집형", _items(100000), 100000)["within"] is True
    over = check_budget("맛집형", _items(70000, 40000), 100000)
    assert over == {"within": False, "total": 110000, "overage": 10000}
    assert check_budget("가성비형", _items(90000), 100000) == {"within": False, "total": 90000, "overage": 10000}
    assert check_budget("가성비형", _items(80000), 100000)["within"] is True


def test_closest_attempt():
    """초과가 가장 작은 시도, 동률이면 나중 시도, 빈 목록은 빈 목록."""
    a, b, c = _items(150000), _items(120000), _items(130000)
    assert closest_attempt([a, b, c], "맛집형", 100000) == b
    d, e = _items(120000), _items(120000, 0)
    assert closest_attempt([d, e], "맛집형", 100000) is e
    assert closest_attempt([], "맛집형", 100000) == []


def test_readjust_count_and_needs():
    """재조정 횟수는 시도 수 - 1, 최대 3회."""
    over = _items(200000)
    assert readjust_count([over]) == 0
    assert needs_readjust("맛집형", [over], 100000) is True
    assert needs_readjust("맛집형", [over] * 3, 100000) is True
    assert needs_readjust("맛집형", [over] * 4, 100000) is False
    assert needs_readjust("맛집형", [_items(50000)], 100000) is False


def test_final_items():
    """충족한 마지막 시도 우선, 없으면 가장 근접한 시도."""
    ok, over1, over2 = _items(50000), _items(200000), _items(150000)
    assert final_items("맛집형", [over1, ok, over2], 100000) == ok
    assert final_items("맛집형", [over1, over2], 100000) == over2


def test_compact_price_result_trims_big_mcp_json():
    """마이리얼트립 큰 JSON 이 목록 몇 줄로 줄고, 파싱 실패 시 앞부분만 돌려준다."""
    import json

    from src.tools import compact_price_result

    widget = {"widget": {"type": "ListView", "children": [
        {"type": "Text", "value": "디즈니랜드 1일권"}, {"type": "Text", "value": "⭐ 4.8 (449)"}, {"type": "Text", "value": "48,401원~"},
        {"type": "Text", "value": "공항 픽업"}, {"type": "Text", "value": "30,000원~"},
    ]}}
    lines = compact_price_result("searchTnas", json.dumps(widget)).splitlines()
    assert lines == ["- 디즈니랜드 1일권 | ⭐ 4.8 (449) | 48,401원~", "- 공항 픽업 | 30,000원~"]
    calendar = {"result": {"items": [{"departureDate": "2026-12-12", "returnDate": "2026-12-15", "airline": "BX", "totalPrice": 328400}]}}
    out = compact_price_result("flightsFareCalendar", json.dumps(calendar))
    assert "328,400원" in out and "실시간 아님" in out
    flight = {"result": {"items": [{
        "airline": {"name": "에어부산"}, "travelInfo": {"isDirect": True},
        "legs": [
            {"origin": "PUS", "destination": "NRT", "departDate": "2026-12-10", "departTime": "08:20", "arriveDate": "2026-12-10",
             "arriveTime": "10:25", "segments": [{"flightNumber": "0717"}]},
            {"origin": "NRT", "destination": "PUS", "departDate": "2026-12-14", "departTime": "11:25", "arriveDate": "2026-12-14",
             "arriveTime": "13:55"},
        ],
        "price": {"total": 335600},
    }]}}
    text = compact_price_result("searchInternationalFlights", json.dumps(flight))
    assert "에어부산 0717 직항" in text and "1인 왕복 합계 335,600원" in text
    assert "가는 편 PUS→NRT 2026-12-10 08:20→2026-12-10 10:25" in text  # 가는 편 날짜·시각
    assert "오는 편 NRT→PUS 2026-12-14 11:25→2026-12-14 13:55" in text  # 귀국 편 날짜·시각
    domestic = {"result": {"items": [{
        "airline": {"name": "에어서울"},
        "outbound": {"flightNumber": "RS6439", "departCity": "GMP", "arriveCity": "CJU", "departDate": "2026-12-05",
                     "departTime": "18:40", "arriveDate": "2026-12-05", "arriveTime": "19:55"},
        "inbound": {"flightNumber": "7C112", "departCity": "CJU", "arriveCity": "GMP", "departDate": "2026-12-07",
                    "departTime": "11:00", "arriveDate": "2026-12-07", "arriveTime": "12:15"},
        "price": {"total": 85400},
    }]}}
    text = compact_price_result("searchDomesticFlights", json.dumps(domestic))
    assert "RS6439 GMP→CJU 2026-12-05 18:40→2026-12-05 19:55" in text and "7C112 CJU→GMP 2026-12-07 11:00→2026-12-07 12:15" in text
    assert compact_price_result("searchTnas", "not json " * 1000) == ("not json " * 1000)[:3000]


def test_compact_stays_keeps_per_night_price():
    """숙소 검색 결과도 이름·평점·1박 요금 한 줄로 줄어든다."""
    import json

    from src.tools import compact_price_result

    widget = {"widget": {"children": [
        {"type": "Text", "value": "신주쿠 워싱턴 호텔"}, {"type": "Text", "value": "⭐ 4.1 (4,457)"}, {"type": "Text", "value": "195,280원/박"},
        {"type": "Text", "value": "APA 호텔 롯폰기"}, {"type": "Text", "value": "137,307원/박"},
    ]}}
    assert compact_price_result("searchStays", json.dumps(widget)).splitlines() == [
        "- 신주쿠 워싱턴 호텔 | ⭐ 4.1 (4,457) | 195,280원/박", "- APA 호텔 롯폰기 | 137,307원/박"]


def test_no_budget_means_no_limit_and_always_within():
    """예산이 없으면 목표 예산도 없고 합계가 아무리 커도 충족이며 재조정 대상이 아니다."""
    from src.tools import check_budget, needs_readjust, target_budget

    item = {"day": 1, "time_slot": "아침", "activity": "a", "cost_krw": 9_999_999, "transport": "도보", "note": ""}
    assert target_budget("맛집형", None) is None and target_budget("가성비형", None) is None
    assert check_budget("가성비형", [item], None) == {"within": True, "total": 9_999_999, "overage": 0}
    assert needs_readjust("맛집형", [[item]], None) is False


def test_apply_headcount_multiplies_only_per_person_items():
    """per_person 항목만 코드가 1인 가격 x 인원으로 전체 비용을 다시 계산하고 원본은 바꾸지 않는다."""
    from src.tools import apply_headcount, total_cost

    base = {"day": 1, "time_slot": "아침", "transport": "도보", "note": ""}
    items = [
        {**base, "activity": "입장권", "cost_krw": 1, "per_person": True, "unit_cost_krw": 13_000},  # LLM 값(1)은 무시
        {**base, "activity": "숙박", "cost_krw": 200_000, "per_person": False, "unit_cost_krw": 0},
        {**base, "activity": "택시", "cost_krw": 30_000},
        {**base, "activity": "단가 없음", "cost_krw": 5_000, "per_person": True, "unit_cost_krw": 0},
    ]
    out = apply_headcount(items, 3)
    assert [i["cost_krw"] for i in out] == [39_000, 200_000, 30_000, 5_000]
    assert items[0]["cost_krw"] == 1  # 원본 불변
    assert total_cost(out) == 274_000
    assert apply_headcount(items, None)[0]["cost_krw"] == 1  # 인원을 모르면 건드리지 않음


def test_required_fields_are_destination_days_people_and_budget_is_optional():
    """필수 입력은 여행지·기간·인원이고 예산은 없어도 되며, 있으면 병합 때 유지된다."""
    from src.tools import merge_trip, missing_fields

    assert missing_fields({"destination": "도쿄", "days": 4, "people": 2}) == []
    assert missing_fields({"destination": "도쿄", "days": 4, "budget_krw": 1_000_000}) == ["인원"]
    assert missing_fields({}) == ["여행지", "기간", "인원"]
    merged = merge_trip({"destination": "도쿄", "budget_krw": 500_000}, {"days": 4, "people": 2, "budget_krw": None})
    assert merged == {
        "destination": "도쿄", "budget_krw": 500_000, "days": 4, "people": 2, "origin": None, "start_date": None}


def test_plain_answer_without_budget_says_no_limit():
    """예산이 없으면 plainAnswer 에 인원과 '예산 제한 없음'이 나오고 합계만 보여준다."""
    from src.tools import render_plain_answer

    item = {"day": 1, "time_slot": "아침", "activity": "a", "cost_krw": 50_000, "transport": "도보", "note": ""}
    plans = {pt: {"plan_type": pt, "attempts": [[item]]} for pt in ("맛집형", "명소형", "가성비형")}
    text = render_plain_answer({"destination": "제주", "days": 3, "people": 2, "budget_krw": None}, plans)
    assert "2명" in text and "예산 제한 없음 기준" in text
    assert "- 맛집형: 합계 50,000원 — 예산 제한 없음" in text and "미충족" not in text


def test_normalize_time_and_origin_defaults():
    """시각은 HH:MM 으로 맞추고 잘못된 값은 빈 문자열이며, 출발지가 없으면 서울(기본값)이다."""
    from src.tools import normalize_time, origin_label, trip_origin

    assert [normalize_time(v) for v in ("9:05", "09:05", " 7:30 ", "23:59")] == ["09:05", "09:05", "07:30", "23:59"]
    assert [normalize_time(v) for v in ("9:5", "abc", "", None, "24:00", "12:60", "<b>1:00</b>")] == [""] * 7
    assert trip_origin({}) == ("서울", True) and trip_origin({"origin": None}) == ("서울", True)
    assert trip_origin({"origin": "부산"}) == ("부산", False)
    assert origin_label({}) == "서울(기본값)" and origin_label({"origin": "부산"}) == "부산"


def test_trip_dates_uses_input_date_or_d_plus_30():
    """명시한 미래 날짜가 있으면 그 날짜, 없거나 지났으면 오늘+30일을 쓰고, 종료일은 시작일 + 기간 - 1 이다."""
    from datetime import date

    from src.tools import period_label, trip_dates

    today = date(2026, 11, 5)
    assert trip_dates({"start_date": "2026-12-20", "days": 4}, today) == (date(2026, 12, 20), date(2026, 12, 23), False)
    assert trip_dates({"days": 3}, today) == (date(2026, 12, 5), date(2026, 12, 7), True)  # D+30
    assert trip_dates({"start_date": "2026-01-01", "days": 3}, today)[2] is True  # 지난 날짜는 기본값
    assert trip_dates({"start_date": "잘못된 값", "days": 1}, today) == (date(2026, 12, 5), date(2026, 12, 5), True)
    assert period_label({"start_date": "2026-12-20", "days": 4}, today) == "2026-12-20 ~ 2026-12-23 (입력)"
    assert period_label({"days": 3}, today) == "2026-12-05 ~ 2026-12-07 (입력이 없어 D+30 기본값)"
