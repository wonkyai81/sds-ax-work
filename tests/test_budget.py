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
             "arriveTime": "13:55", "segments": [{"flightNumber": "0718"}]},
        ],
        "price": {"total": 335600},
    }]}}
    text = compact_price_result("searchInternationalFlights", json.dumps(flight))
    assert "에어부산 직항" in text and "1인 왕복 합계 335,600원" in text
    assert "가는 편 0717 PUS→NRT 2026-12-10 08:20→2026-12-10 10:25" in text  # 가는 편 편명·날짜·시각
    assert "오는 편 0718 NRT→PUS 2026-12-14 11:25→2026-12-14 13:55" in text  # 귀국 편은 자기 편명을 쓴다
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


def test_note_is_clipped_to_max_chars_by_code():
    """비고는 상한(NOTE_MAX_CHARS)을 넘으면 코드가 '…'로 줄이고, 짧은 비고와 다른 필드는 그대로 둔다."""
    from src.tools import NOTE_MAX_CHARS, apply_headcount, clip_note

    base = {"day": 1, "time_slot": "아침", "activity": "a", "cost_krw": 1000, "transport": "도보"}
    long_note = "가" * (NOTE_MAX_CHARS + 25)
    out = apply_headcount([{**base, "note": long_note}, {**base, "note": "짧은 비고"}, {**base, "note": None}], 2)
    assert len(out[0]["note"]) == NOTE_MAX_CHARS and out[0]["note"].endswith("…")
    assert out[1]["note"] == "짧은 비고" and out[2]["note"] == ""
    assert out[0]["activity"] == "a" and out[0]["cost_krw"] == 1000
    exact = "나" * NOTE_MAX_CHARS
    assert clip_note(exact) == exact  # 정확히 상한이면 자르지 않는다
    assert len(clip_note(exact + "x")) == NOTE_MAX_CHARS


def _it(day, slot, activity, cost, depart=""):
    """테스트용 일정 항목."""
    return {"day": day, "time_slot": slot, "activity": activity, "cost_krw": cost, "transport": "", "note": "", "depart_time": depart}


def test_budget_usage_line_for_under_over_none_and_value_plan():
    """예산 대비 문구: 사용 비율·잔액, 초과(소수 1자리 올림), 예산 없음, 가성비형 목표(80%) 이내/초과."""
    from src.tools import budget_usage_line

    item = lambda cost: [_it(1, "아침", "a", cost)]  # noqa: E731
    assert budget_usage_line("맛집형", item(950_000), 1_000_000) == "95% 사용, 잔액 50,000원"
    assert budget_usage_line("맛집형", item(1_000_000), 1_000_000) == "100% 사용, 잔액 0원"
    assert budget_usage_line("맛집형", item(1_001_000), 1_000_000) == "100.1% 사용, 1,000원 초과"
    assert budget_usage_line("맛집형", item(1_000_001), 1_000_000) == "100.1% 사용, 1원 초과"  # 초과인데 100.0% 로 보이지 않는다
    assert budget_usage_line("맛집형", item(10_000_000), None) == "예산 제한 없음"
    assert budget_usage_line("가성비형", item(700_000), 1_000_000) == "70% 사용, 잔액 300,000원, 목표 80% 이내"
    assert budget_usage_line("가성비형", item(900_000), 1_000_000) == "90% 사용, 잔액 100,000원, 목표 80% 초과"


def test_main_schedule_is_sight_focused_and_skips_logistics_and_restaurants():
    """주요 일정 5개는 명소 위주다. 이동·항공·체크인과(통계가 없는) 식당은 넣지 않고, 시간순으로 돌려준다."""
    from src.tools import main_schedule

    items = [
        _it(1, "아침", "서울→도쿄 항공", 900_000, "07:10"),
        _it(1, "오후", "센소지 관광", 0, "13:00"),
        _it(1, "저녁", "스시 저녁", 120_000, "19:00"),
        _it(1, "저녁", "숙소 체크인: 호텔", 600_000, "21:00"),
        _it(2, "아침", "츠키지 시장", 40_000, "08:00"),
        _it(2, "오후", "디즈니씨 입장", 200_000, "11:00"),
        _it(3, "오후", "도쿄타워 전망대", 15_000, "14:00"),
        _it(3, "저녁", "공항 이동", 30_000, "18:00"),
        _it(4, "아침", "메이지 신궁 참배", 0, "09:00"),
        _it(4, "아침", "귀국 항공", 0, "10:15"),
    ]
    lines = main_schedule(items)
    joined = " ".join(lines)
    assert len(lines) == 5
    for sight in ("센소지", "츠키지", "디즈니씨", "도쿄타워", "메이지 신궁"):
        assert sight in joined
    assert "스시" not in joined and "항공" not in joined and "체크인" not in joined and "이동" not in joined
    assert lines == sorted(lines, key=lambda l: (int(l[0]), l[2:]))  # 일차 순서(같은 일차는 시각 순)


def test_main_schedule_handles_few_items_and_long_names():
    """명소가 5개 미만이면 이동 항목으로만 채우고(기준 미달 식당은 넣지 않음), 긴 활동명은 40자에서 줄인다."""
    from src.tools import main_schedule

    few = main_schedule([_it(1, "아침", "서울→도쿄 항공", 500_000, "07:00"), _it(2, "오후", "우에노 공원 산책", 0),
                         _it(2, "저녁", "스시 저녁", 100_000)])
    assert len(few) == 2 and "서울→도쿄 항공" in few[0] and "우에노" in few[1]  # 식당(스시)은 빠진다
    assert main_schedule([]) == []
    long_line = main_schedule([_it(1, "아침", "가" * 80 + " 관람", 1000)])[0]
    assert "가" * 39 + "…" in long_line and "가" * 41 not in long_line


def test_difference_sentence_rules_and_clipping():
    """차별점은 2~3문장·250자 이내다. 4문장 이상이면 앞 3문장만, 1문장이면 폴백으로 2문장, 소수점은 문장 끝이 아니다."""
    from src.tools import (DIFF_MAX_CHARS, clip_difference, difference_ok, fit_difference, split_sentences)

    assert split_sentences("평점은 4.5점입니다. 리뷰가 많습니다!  좋아요?") == ["평점은 4.5점입니다.", "리뷰가 많습니다!", "좋아요?"]
    two = "합계가 가장 높습니다. 식사 비중이 큽니다."
    assert difference_ok(two) and difference_ok(two + " 대표 일정은 A입니다.")
    assert not difference_ok("한 문장뿐입니다.") and not difference_ok(" ".join(["문장입니다."] * 4))
    assert not difference_ok(("가" * 150 + ". ") * 2)  # 250자 초과
    assert not difference_ok(None)
    five = " ".join(f"{i}번째 문장입니다." for i in range(1, 6))
    assert clip_difference(five) == "1번째 문장입니다. 2번째 문장입니다. 3번째 문장입니다."
    long_pair = ("가" * 140 + ".") + " " + ("나" * 140 + ".")
    assert clip_difference(long_pair) == "가" * 140 + "."  # 두 번째 문장을 넣으면 250자를 넘으므로 첫 문장까지만
    huge = clip_difference("다" * 400 + ".")
    assert len(huge) == DIFF_MAX_CHARS and huge.endswith("…")
    fallback = "폴백 첫 문장입니다. 폴백 둘째 문장입니다. 폴백 셋째 문장입니다."
    assert fit_difference("한 문장뿐입니다.", fallback) == "한 문장뿐입니다. 폴백 첫 문장입니다."
    assert fit_difference(None, fallback) == "폴백 첫 문장입니다. 폴백 둘째 문장입니다."
    assert fit_difference(two, fallback) == two  # 이미 2문장이면 그대로


def test_fallback_difference_is_short_and_uses_numbers_from_the_plans():
    """폴백은 2~3문장이고 합계 순위, 가장 싼/비싼 안과의 차이, 대표 일정(명소)을 실제 값으로 말한다."""
    from src.tools import difference_ok, fallback_difference

    rich = [_it(1, "아침", "서울→제주 항공", 170_000, "07:00"), _it(1, "저녁", "숙소 체크인: 호텔", 300_000), _it(2, "오후", "성산일출봉 관람", 20_000)]
    plans = {"맛집형": rich, "명소형": [_it(1, "오후", "한라산 등반", 10_000)], "가성비형": [_it(1, "오후", "올레길 산책", 0)]}
    text = fallback_difference("맛집형", plans)
    assert difference_ok(text)
    assert "490,000원" in text and "1번째로 큽니다" in text and "가장 저렴한 가성비형보다 490,000원 더 듭니다" in text
    assert "성산일출봉" in text
    cheapest = fallback_difference("가성비형", plans)
    assert "가장 비싼 맛집형보다 490,000원 덜 듭니다" in cheapest and difference_ok(cheapest)
    same = fallback_difference("명소형", {"명소형": [_it(1, "오후", "남산 관람", 5)], "맛집형": [_it(1, "오후", "경복궁 관람", 5)]})
    assert "다른 안과 합계가 같습니다" in same
    assert "합계는 7원입니다" in fallback_difference("명소형", {"명소형": [_it(1, "오후", "남산 관람", 7)]})  # 안이 하나뿐일 때


def test_plain_answer_blocks_have_budget_difference_and_schedule():
    """3안 plainAnswer 는 첫 줄 형식을 유지한 채 안마다 예산 대비·차별점(2~3문장)·주요 일정 블록이 들여쓰기로 붙는다."""
    from src.tools import difference_ok, render_plain_answer

    items = [_it(d, s, f"명소{d}{s} 관람", 10_000 * d, "09:00") for d in (1, 2, 3) for s in ("아침", "오후", "저녁")]
    plans = {pt: {"plan_type": pt, "attempts": [items]} for pt in ("맛집형", "명소형", "가성비형")}
    trip = {"destination": "도쿄", "days": 3, "people": 2, "budget_krw": 1_000_000}
    text = render_plain_answer(trip, plans, comparison={"맛집형": "맛집형은 식사가 많습니다. 합계는 가장 높습니다."})
    lines = text.splitlines()
    heads = [l for l in lines if l.startswith("- ")]
    assert len(heads) == 3 and heads[0].startswith("- 맛집형: 합계 ") and "목표" in heads[0]  # 기존 첫 줄 형식 유지
    block = lines[lines.index(heads[0]) + 1: lines.index(heads[1])]
    assert block[0].startswith("  예산 대비: ") and "% 사용" in block[0]
    assert block[1] == "  차별점: 맛집형은 식사가 많습니다. 합계는 가장 높습니다."
    assert block[2] == "  주요 일정:" and len([l for l in block if l.startswith("  · ")]) == 5
    assert all(l.startswith("  ") for l in block)  # 들여쓴 줄만 카드 내용
    other = lines[lines.index(heads[1]) + 1: lines.index(heads[2])]  # comparison 이 없는 안은 폴백(2~3문장)
    assert difference_ok(other[1].removeprefix("  차별점: "))




def test_is_restaurant_prefers_category_then_meal_words():
    """식당 판별: category 가 있으면 그것을 따르고, 없으면 식사 단어로 보되 관람·산책 같은 단어가 있으면 식당이 아니다."""
    from src.tools import is_restaurant

    assert is_restaurant({"activity": "Ichiran Shibuya", "category": "식당"}) is True
    assert is_restaurant({"activity": "맛집 거리 구경", "category": "명소"}) is False
    assert is_restaurant({"activity": "Momo Paradise Shinjuku 저녁"}) is True
    assert is_restaurant({"activity": "고집돌우럭에서 브런치"}) is True
    assert is_restaurant({"activity": "롯폰기 힐즈 저녁 야경 감상"}) is False
    assert is_restaurant({"activity": "아침 산책"}) is False
    assert is_restaurant({"activity": "센소지 관광"}) is False


def test_place_stats_matching_ignores_case_spaces_and_symbols():
    """활동명에 장소 이름이 들어 있으면(대소문자·공백·기호 무시) 그 통계를 찾고, 짧은 이름은 무시하며, 여러 개면 가장 긴 이름을 쓴다."""
    from src.tools import place_stats_for

    stats = {"MO-MO-PARADISE Shinjuku": {"rating": 4.7, "reviews": 8000}, "MO-MO": {"rating": 3.0, "reviews": 10},
             "Momo Paradise": {"rating": 4.0, "reviews": 100}, "AB": {"rating": 5.0, "reviews": 99999}}
    assert place_stats_for("momo paradise shinjuku 저녁 식사", stats) == {"rating": 4.7, "reviews": 8000}  # 긴 이름이 우선
    assert place_stats_for("MoMo Paradise 점심", stats) == {"rating": 4.0, "reviews": 100}
    assert place_stats_for("AB 카페", stats) is None  # 4글자 미만 이름은 무시
    assert place_stats_for("센소지 관광", stats) is None and place_stats_for("아무거나", None) is None


def test_main_schedule_adds_at_most_one_restaurant_that_meets_rating_and_reviews():
    """식당은 평점 4.5 이상이고 리뷰 5,000개 이상일 때만, 최대 1개(리뷰가 가장 많은 것)가 근거(★·리뷰 수)와 함께 들어간다."""
    from src.tools import main_schedule

    stats = {
        "Ichiran Shibuya": {"rating": 4.6, "reviews": 12_000},          # 충족, 리뷰 최다
        "Momo Paradise Shinjuku": {"rating": 4.7, "reviews": 8_200},     # 충족
        "Halal Wagyu Yakiniku": {"rating": 4.4, "reviews": 9_000},       # 평점 미달
        "Tokyo Saikabo": {"rating": 4.8, "reviews": 4_999},              # 리뷰 미달
    }
    items = [
        _it(1, "오후", "센소지 관광", 0, "13:00"), _it(2, "오후", "도쿄타워 전망대", 15_000, "14:00"),
        _it(3, "오후", "메이지 신궁 참배", 0, "10:00"), _it(4, "오후", "디즈니씨 입장", 100_000, "11:00"),
        _it(4, "오후", "우에노 공원 산책", 0, "16:00"), _it(1, "저녁", "스카이트리 전망대", 20_000, "17:00"),
        {**_it(1, "저녁", "Ichiran Shibuya", 25_000, "19:00"), "category": "식당"},
        _it(2, "저녁", "Momo Paradise Shinjuku 저녁", 40_000, "19:00"),
        _it(3, "저녁", "Halal Wagyu Yakiniku 저녁", 48_000, "19:00"),
        _it(4, "저녁", "Tokyo Saikabo 저녁", 60_000, "19:00"),
    ]
    with_stats = main_schedule(items, place_stats=stats)
    joined = " ".join(with_stats)
    assert len(with_stats) == 5 and sum("★" in l for l in with_stats) == 1  # 식당은 최대 1개
    assert "Ichiran Shibuya" in joined and "★4.6 리뷰 12,000개" in joined
    assert "Momo Paradise" not in joined and "Halal" not in joined and "Saikabo" not in joined
    without = " ".join(main_schedule(items))  # 통계를 모르면 식당은 하나도 넣지 않는다
    assert "★" not in without and not any(n in without for n in ("Ichiran", "Momo", "Halal", "Saikabo"))


def test_google_maps_search_results_are_collected_as_place_stats():
    """장소 검색 결과(JSON)에서 이름·평점·리뷰 수를 모으고, 검색 도구가 아니거나 JSON 이 아니면 무시한다."""
    import json

    from src.tools import GoogleMapsMCP

    m = GoogleMapsMCP("dummy-key")  # 서버는 띄우지 않고 결과 관측만 시험한다
    results = [
        {"name": "Mugyo-dong Bugeo-guk", "rating": 4.5, "total_ratings": 3998, "address": "Seoul"},
        {"name": "No Reviews", "rating": 4.0},
        {"place": {"name": "Nested Place", "rating": 4.8, "user_ratings_total": 7000}},
    ]
    m._observe("maps_search_places", json.dumps(results))
    m._observe("maps_directions", json.dumps([{"name": "Ignored", "rating": 5, "total_ratings": 1}]))
    m._observe("maps_search_places", "not json")
    snapshot = m.stats_snapshot()
    assert snapshot == {"Mugyo-dong Bugeo-guk": {"rating": 4.5, "reviews": 3998}, "Nested Place": {"rating": 4.8, "reviews": 7000}}
    snapshot["x"] = 1
    assert "x" not in m.stats_snapshot()  # 복사본이다


def test_budget_phrase_breaks_down_target_per_person_and_per_day():
    """예산이 있으면 코드가 계산한 목표·1인당·1일당 금액과 인원 곱셈 규칙이 프롬프트 문장에 들어간다."""
    from src.agent import _budget_phrase

    text = _budget_phrase("가성비형", 1_000_000, 4, 2)  # 가성비형 목표 800,000
    assert "800000원" in text and "1인당 약 200000원" in text and "1일당 약 400000원" in text
    assert "x 4명" in text and "항공권과 숙박비" in text
    # 인원·일수를 주지 않으면 기존 한 줄 문장만 쓴다
    assert _budget_phrase("맛집형", 1_000_000) == "목표 총액은 1000000원 이하입니다. "


def test_budget_phrase_without_budget_keeps_no_limit_sentence():
    """예산이 없으면 인원·일수를 줘도 금액 상한 없음 문장 그대로다."""
    from src.agent import _budget_phrase

    assert _budget_phrase("맛집형", None, 3, 2) == "예산 제한이 없으므로 금액 상한 없이 만드세요. "


def test_top_cost_text_lists_most_expensive_items_first():
    """재조정 프롬프트용으로 비용이 큰 항목 상위 3개를 큰 순서로 보여준다."""
    from src.agent import _top_cost_text

    items = [
        {"day": 1, "time_slot": "아침", "activity": f"항목{c}", "cost_krw": c, "transport": "", "note": ""}
        for c in (10, 500, 30, 400, 20)
    ]
    assert _top_cost_text(items) == "항목500(500원), 항목400(400원), 항목30(30원)"


# ---- 계산은 LLM 이 아니라 코드가 한다: finalize_items ----
def _fi(activity, cost=0, **kw):
    return {"day": 1, "time_slot": "저녁", "activity": activity, "cost_krw": cost, "transport": "도보", "note": "", **kw}


def test_finalize_lodging_total_is_room_rate_x_rooms_x_nights_in_code():
    """숙박은 LLM 이 준 객실 1박 요금에 코드가 객실 수(2명당 1실 올림)와 박 수를 곱한다. LLM 이 쓴 총액은 무시한다."""
    from src.tools import finalize_items

    out = finalize_items([_fi("숙소 체크인: A호텔", cost=1, unit_cost_krw=100_000, category="숙박")], people=3, days=4)
    assert out[0]["cost_krw"] == 100_000 * 2 * 3 and out[0]["rooms"] == 2 and out[0]["nights"] == 3
    # 당일치기는 숙박을 계산하지 않는다
    assert finalize_items([_fi("숙소 체크인: A", cost=50_000)], people=2, days=1)[0]["cost_krw"] == 50_000


def test_finalize_floors_zero_lodging_and_meals_so_budget_cannot_be_gamed():
    """숙박·식사를 0원으로 쓰면 코드가 최소 단가로 올려 정직한 합계가 나온다(원본은 불변)."""
    from src.tools import MIN_LODGING_ROOM_NIGHT_KRW, MIN_MEAL_PER_PERSON_KRW, finalize_items, total_cost

    items = [_fi("숙소 체크인: A", 0, category="숙박"), _fi("점심", 0, category="식당"), _fi("관람", 0, category="명소")]
    out = finalize_items(items, people=2, days=3)
    assert out[0]["cost_krw"] == MIN_LODGING_ROOM_NIGHT_KRW * 1 * 2 and "최소 단가 적용" in out[0]["note"]
    assert out[1]["cost_krw"] == MIN_MEAL_PER_PERSON_KRW * 2 and out[1]["per_person"] is True
    assert out[2]["cost_krw"] == 0  # 무료 명소는 그대로
    assert total_cost(out) == 60_000 + 6_000 and items[0]["cost_krw"] == 0
    # 최소 단가보다 높은 식사는 건드리지 않는다
    assert finalize_items([_fi("저녁", 0, category="식당", per_person=True, unit_cost_krw=20_000)], 2, 1)[0]["cost_krw"] == 40_000


def test_finalize_converts_local_currency_with_code_not_llm():
    """현지 통화 단가는 환율(코드)을 곱해 원화로 바꾼다. 통화 코드가 다르면 무시한다."""
    from src.tools import finalize_items

    fx = {"currency": "JPY", "krw_per_unit": 9.5, "date": "2026-11-05"}
    ticket = _fi("입장권", 0, category="명소", per_person=True, local_unit_price=2000, local_currency="jpy")
    hotel = _fi("숙소 체크인: A", 0, category="숙박", local_unit_price=20_000, local_currency="JPY")
    taxi = _fi("택시", 0, local_unit_price=5000, local_currency="JPY")
    out = finalize_items([ticket, hotel, taxi], people=2, days=2, fx=fx)
    assert out[0]["unit_cost_krw"] == 19_000 and out[0]["cost_krw"] == 38_000
    assert out[1]["unit_cost_krw"] == 190_000 and out[1]["cost_krw"] == 190_000  # 2명 1실 1박
    assert out[2]["cost_krw"] == 47_500
    wrong = finalize_items([_fi("입장권", 0, category="명소", per_person=True, local_unit_price=2000, local_currency="USD")], 2, 2, fx=fx)
    assert wrong[0].get("unit_cost_krw", 0) == 0 and wrong[0]["cost_krw"] == 0  # 통화가 달라 환산하지 않는다


def test_fit_difference_drops_sentences_with_money_numbers_written_by_llm():
    """LLM 이 쓴 금액·비율 숫자 문장은 버리고 코드가 만든 폴백 문장으로 채운다."""
    from src.tools import drop_money_sentences, fit_difference

    text = "합계가 1,507,510원으로 가장 비쌉니다. 맛집 위주 일정입니다. 예산의 95%를 씁니다. 4일 동안 3곳을 갑니다."
    assert drop_money_sentences(text) == "맛집 위주 일정입니다. 4일 동안 3곳을 갑니다."
    fitted = fit_difference("합계가 1,507,510원입니다. 예산 대비 95%입니다.", "폴백 첫 문장입니다. 폴백 둘째 문장입니다.")
    assert fitted == "폴백 첫 문장입니다. 폴백 둘째 문장입니다."

# ---- 식사 가격은 Google 가격대 기준표로 확정한다 ----

def test_place_stats_collect_price_level_and_skip_unsupported_values():
    """검색 결과의 price_level 을 기준표에 있는 값만 담는다(UNSPECIFIED·없음은 담지 않는다)."""
    import json

    from src.tools import GoogleMapsMCP

    m = GoogleMapsMCP("dummy-key")
    results = [
        {"name": "Ichiran Shibuya", "rating": 4.4, "total_ratings": 5327, "price_level": "PRICE_LEVEL_MODERATE"},
        {"name": "Itamae Sushi Prime", "rating": 4.7, "total_ratings": 1076, "price_level": "PRICE_LEVEL_UNSPECIFIED"},
        {"name": "No Price Cafe", "rating": 4.0, "total_ratings": 100},
    ]
    m._observe("maps_search_places", json.dumps(results))
    snap = m.stats_snapshot()
    assert snap["Ichiran Shibuya"] == {"rating": 4.4, "reviews": 5327, "price_level": "MODERATE"}
    assert "price_level" not in snap["Itamae Sushi Prime"] and "price_level" not in snap["No Price Cafe"]


def test_restaurant_price_comes_from_google_price_level_not_from_llm():
    """가격대를 아는 식당은 LLM 이 쓴 금액과 무관하게 기준표 1인 가격 x 인원으로 확정한다. 모르는 식당은 LLM 값을 쓴다."""
    from src.tools import MEAL_PRICE_BY_LEVEL, finalize_items

    assert MEAL_PRICE_BY_LEVEL == {"INEXPENSIVE": 10_000, "MODERATE": 20_000, "EXPENSIVE": 50_000, "VERY_EXPENSIVE": 100_000}
    stats = {
        "Ichiran Shibuya": {"rating": 4.4, "reviews": 5327, "price_level": "MODERATE"},
        "Ginza Kyubey Honten": {"rating": 4.4, "reviews": 2638, "price_level": "VERY_EXPENSIVE"},
    }
    items = [
        _fi("Ichiran Shibuya 점심", 0, category="식당", per_person=True, unit_cost_krw=7_000),
        _fi("Ginza Kyubey Honten 저녁", 0, category="식당", per_person=True, unit_cost_krw=30_000),
        _fi("이름 모르는 식당 저녁", 0, category="식당", per_person=True, unit_cost_krw=15_000),
    ]
    out = finalize_items(items, people=2, days=1, place_stats=stats)
    assert [(i["unit_cost_krw"], i["cost_krw"]) for i in out] == [(20_000, 40_000), (100_000, 200_000), (15_000, 30_000)]
    assert out[0]["note"].startswith("Google 가격대 보통 기준")
    assert out[1]["note"].startswith("Google 가격대 매우 비쌈 기준")
    assert "Google 가격대" not in out[2]["note"]
    assert items[0]["unit_cost_krw"] == 7_000  # 원본은 바꾸지 않는다


# ---- 항공·숙소 가격은 마이리얼트립 조회값으로 코드가 확정한다 ----

def _intl_flights_raw(total=337_800, airline="제주항공", direct=True):
    import json

    def leg(i, o, d, dd, dt, at, no):
        return {"legIndex": i, "origin": o, "destination": d, "departDate": dd, "departTime": dt, "arriveDate": dd, "arriveTime": at,
                "segments": [{"airlineCode": "7C", "flightNumber": no}]}

    item = {"airline": {"code": "7C", "name": airline}, "travelInfo": {"isDirect": direct},
            "legs": [leg(1, "ICN", "NRT", "2026-12-06", "09:50", "12:15", "1183"), leg(2, "NRT", "ICN", "2026-12-09", "14:00", "17:05", "1104")],
            "price": {"currency": "KRW", "total": total}}
    return json.dumps({"success": True, "result": {"items": [item]}})


def _stays_raw():
    import json

    def stay(name, price):
        return {"type": "ListViewItem", "children": [{"type": "Text", "value": name}, {"type": "Text", "value": "⭐ 4.5 (1,000)"}, {"type": "Text", "value": price}]}

    return json.dumps({"widget": {"type": "ListView", "children": [stay("토부 호텔 레밴트 도쿄", "181,584원/박"), stay("APA 호텔 롯폰기", "97,891원/박")]}})


def test_parse_price_quotes_reads_flights_and_stays_and_divides_by_passengers():
    from src.tools import parse_price_quotes, record_price_quotes

    one = parse_price_quotes("searchInternationalFlights", _intl_flights_raw(), {"passengers": {"adults": 1}})["flights"][0]
    assert one["price_krw"] == 337_800 and one["direct"] is True and one["airline"] == "제주항공"
    assert one["out"] == {"date": "2026-12-06", "depart": "09:50", "arrive": "12:15", "no": "7C1183"}
    assert one["back"]["no"] == "7C1104" and one["back"]["depart"] == "14:00"
    two = parse_price_quotes("searchInternationalFlights", _intl_flights_raw(total=675_600), {"passengers": {"adults": 2}})["flights"][0]
    assert two["price_krw"] == 337_800  # 조회 합계는 전체 승객 총액이라 1인 가격으로 나눈다
    stays = parse_price_quotes("searchStays", _stays_raw())["stays"]
    assert stays == [{"name": "토부 호텔 레밴트 도쿄", "price_krw": 181_584}, {"name": "APA 호텔 롯폰기", "price_krw": 97_891}]
    assert parse_price_quotes("searchStays", "not json") == {} and parse_price_quotes("searchTnas", _stays_raw()) == {}
    sink: dict = {}
    record_price_quotes(sink, "searchStays", _stays_raw())
    record_price_quotes(sink, "searchInternationalFlights", _intl_flights_raw(), {})
    assert len(sink["stays"]) == 2 and len(sink["flights"]) == 1


def _trip_items():
    return [
        _fi("서울→도쿄 항공", 0, category="이동", per_person=True, unit_cost_krw=500_000, depart_time="08:00", arrive_time="10:00", note="대한항공"),
        _fi("숙소 체크인: 토부 호텔 레밴트 도쿄", 0, category="숙박", unit_cost_krw=999_999),
        _fi("도쿄→서울 귀국 항공", 0, category="이동", per_person=False),
    ]


def test_flight_and_lodging_prices_are_overwritten_by_quotes():
    from src.tools import finalize_items, record_price_quotes

    quotes: dict = {}
    record_price_quotes(quotes, "searchInternationalFlights", _intl_flights_raw(), {})
    record_price_quotes(quotes, "searchStays", _stays_raw())
    items = _trip_items()
    out = finalize_items(items, people=2, days=4, price_quotes=quotes, plan_type="맛집형")
    flight, hotel, back = out
    assert (flight["unit_cost_krw"], flight["cost_krw"]) == (337_800, 675_600)  # 1인 왕복 x 인원, LLM 이 쓴 500,000 은 버림
    assert (flight["depart_time"], flight["arrive_time"]) == ("09:50", "12:15") and "7C1183" in flight["note"]
    assert (back["depart_time"], back["arrive_time"]) == ("14:00", "17:05") and back["cost_krw"] == 0 and "7C1104" in back["note"]
    assert (hotel["unit_cost_krw"], hotel["rooms"], hotel["nights"], hotel["cost_krw"]) == (181_584, 1, 3, 181_584 * 3)
    assert items[0]["unit_cost_krw"] == 500_000  # 원본은 바꾸지 않는다


def test_unmatched_lodging_and_missing_quotes_keep_llm_values():
    from src.tools import finalize_items, record_price_quotes

    quotes: dict = {}
    record_price_quotes(quotes, "searchStays", _stays_raw())
    items = _trip_items()
    items[1] = _fi("숙소 체크인: 내가 고른 다른 호텔", 0, category="숙박", unit_cost_krw=120_000)  # 조회 목록에 없는 숙소(사용자가 직접 고른 경우)
    out = finalize_items(items, people=2, days=4, price_quotes=quotes)
    assert out[1]["unit_cost_krw"] == 120_000 and out[0]["unit_cost_krw"] == 500_000  # 항공 조회값이 없으면 LLM 값 유지
    assert finalize_items(_trip_items(), people=2, days=4)[0]["unit_cost_krw"] == 500_000  # 조회값 자체가 없을 때도 같음


def test_flight_choice_prefers_direct_then_cheapest_and_value_plan_takes_cheapest():
    from src.tools import apply_price_quotes

    quotes = {"flights": [
        {"airline": "A항공", "direct": False, "price_krw": 200_000, "out": {}, "back": {}},
        {"airline": "B항공", "direct": True, "price_krw": 300_000, "out": {}, "back": {}},
        {"airline": "C항공", "direct": True, "price_krw": 280_000, "out": {}, "back": {}},
    ]}
    price = lambda plan, note="": apply_price_quotes([_fi("서울→도쿄 항공", 0, note=note)], quotes, plan)[0]["unit_cost_krw"]
    assert price("맛집형") == 280_000 and price("명소형") == 280_000  # 직항 중 최저가
    assert price("가성비형") == 200_000  # 경유라도 최저가
    assert price("맛집형", note="B항공 희망") == 300_000  # 항목에 적힌 항공사가 후보에 있으면 그 항공사
