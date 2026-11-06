"""HTML 리포트 테스트: 일정표, 합계, 잔액, 초과 경고, AI 추정 안내, 이스케이프."""
from src.tools import render_report_html


def _item(day, slot, activity="활동", cost=10000, transport="도보", note="비고"):
    """테스트용 일정 항목을 만든다."""
    return {"day": day, "time_slot": slot, "activity": activity, "cost_krw": cost,
            "transport": transport, "note": note}


def test_report_contains_schedule_total_balance_notice():
    """일차·시간대·활동·비용·교통·비고, 합계, 잔액, 안내 문구가 모두 들어간다."""
    items = [_item(1, "저녁", "야시장", 30000, "버스", "인기"), _item(1, "아침", "카페", 20000, "지하철", "브런치")]
    html = render_report_html("맛집형", items, 100000)
    for text in ("1일차", "아침", "저녁", "야시장", "카페", "버스", "지하철", "인기", "브런치", "30,000원", "20,000원"):
        assert text in html
    assert "비용 합계: 50,000원" in html
    assert "예산 대비 잔액: 50,000원" in html
    assert "AI 추정값" in html
    assert "경고" not in html
    assert html.index("카페") < html.index("야시장")  # 아침이 저녁보다 먼저


def test_report_warns_when_over_budget():
    """예산 초과 시 경고와 음수 잔액이 표시된다."""
    html = render_report_html("명소형", [_item(1, "아침", cost=150000)], 100000)
    assert "경고" in html
    assert "50,000원 초과" in html
    assert "예산 대비 잔액: -50,000원" in html


def test_value_plan_uses_80_percent_target():
    """가성비형은 예산의 80%를 넘으면 경고한다."""
    html = render_report_html("가성비형", [_item(1, "아침", cost=90000)], 100000)
    assert "경고" in html
    assert "10,000원 초과" in html


def test_report_escapes_llm_strings():
    """LLM이 만든 문자열은 HTML 이스케이프된다."""
    bad = "<script>alert(1)</script>"
    items = [_item(1, "아침", activity=bad, transport="<b>x</b>", note="a&b")]
    html = render_report_html("맛집형", items, 100000)
    assert bad not in html
    assert "<b>x</b>" not in html
    assert "&lt;script&gt;" in html
    assert "a&amp;b" in html


def test_report_guide_tabs_render_and_escape():
    """가이드가 있으면 2·3탭에 맛집·명소, 준비물, 목적지 사전 유의 정보가 들어가고 이스케이프된다."""
    guide = {
        "places": [{"kind": "맛집", "name": "<i>라멘집</i>", "area": "신주쿠", "description": "진한 국물", "tip": "점심 전 방문"},
                   {"kind": "명소", "name": "센소지", "area": "아사쿠사", "description": "사찰", "tip": "아침 방문"}],
        "packing": ["여권", "110V 어댑터"],
        "cautions": [{"title": "소매치기", "detail": "붐비는 역 주의"}],
        "destination_notes": [{"title": "입국", "detail": "출발 전 입국 요건 확인"}],
    }
    html = render_report_html("맛집형", [_item(1, "아침")], 100000, guide)
    for text in ("맛집", "명소", "센소지", "여권", "110V 어댑터", "목적지 사전 유의 정보", "입국", "소매치기"):
        assert text in html
    assert "<i>라멘집</i>" not in html and "&lt;i&gt;" in html
    assert "생성하지 못했습니다" in render_report_html("맛집형", [_item(1, "아침")], 100000)


def test_report_schedule_has_origin_arrival_duration_columns():
    """일정표에 출발지·도착지·소요시간 열이 있고 값이 이스케이프되어 들어간다."""
    item = _item(1, "아침") | {"origin": "호텔", "arrival": "<센소지>", "duration": "약 30분"}
    html = render_report_html("맛집형", [item], 100000)
    for text in ("출발지", "도착지", "소요시간", "호텔", "약 30분", "&lt;센소지&gt;"):
        assert text in html
    assert "<센소지>" not in html
    assert "출발지" in render_report_html("맛집형", [_item(1, "아침")], 100000)  # 값이 없어도 깨지지 않음


def test_report_shows_escaped_price_note_when_given():
    """참고 가격이 있으면 일정표 탭에 출처·비합산 안내와 함께 이스케이프되어 표시된다."""
    html = render_report_html("맛집형", [_item(1, "아침")], 100000, None, "- <b>디즈니</b> 48,401원~")
    assert "참고 가격" in html and "마이리얼트립" in html and "1인 가격 × 인원" in html
    assert "<b>디즈니</b>" not in html and "&lt;b&gt;디즈니" in html
    assert "참고 가격" not in render_report_html("맛집형", [_item(1, "아침")], 100000)


def test_report_shows_unit_price_times_people_equals_total():
    """1인 단위 항목은 '1인 가격 × 인원 = 전체 가격', 공용 항목은 합계만, 상단에 여행 인원이 표시된다."""
    items = [
        _item(1, "아침", "입장권", 26_000) | {"per_person": True, "unit_cost_krw": 13_000},
        _item(1, "저녁", "숙박", 200_000),
    ]
    html = render_report_html("맛집형", items, 1_000_000, None, None, 2)
    assert "13,000원 × 2명 = 26,000원" in html
    assert "<td>200,000원</td>" in html
    assert "여행 인원: 2명" in html and "1인 가격 × 인원 = 전체 가격" in html
    assert "비용 합계: 226,000원" in html
    plain = render_report_html("맛집형", [_item(1, "아침")], 1_000_000)
    assert "×" not in plain and "여행 인원" not in plain


def test_report_without_budget_has_no_balance_or_warning():
    """예산이 없으면 '제한 없음'만 표시하고 잔액·초과 경고는 없다."""
    html = render_report_html("맛집형", [_item(1, "아침", cost=9_999_999)], None, None, None, 2)
    assert "사용자 예산: 제한 없음" in html
    assert "잔액" not in html and "경고" not in html and "목표 예산" not in html


def test_report_has_depart_and_arrive_time_columns_next_to_day():
    """일차 바로 옆에 출발시간·도착시간 열이 있고 HH:MM 으로 표시되며, 잘못된 값은 비워지고 이스케이프된다."""
    import re

    items = [
        _item(1, "아침", "공항 이동") | {"depart_time": "9:05", "arrive_time": "10:30"},
        _item(1, "오후", "점심") | {"depart_time": "<b>x</b>", "arrive_time": ""},
    ]
    html = render_report_html("맛집형", items, 100000)
    header = re.search(r"<thead><tr>(.*?)</tr></thead>", html).group(1)
    assert header.startswith("<th>날짜</th><th>일차</th><th>출발시간</th><th>도착시간</th><th>시간대</th>")
    first_row = re.findall(r"<tr>(.*?)</tr>", html)[1]
    cells = re.findall(r"<td>(.*?)</td>", first_row)
    assert cells[:5] == ["", "1일차", "09:05", "10:30", "아침"]  # 시작일을 안 주면 날짜 칸은 비어 있다
    assert "<b>x</b>" not in html and "&lt;b&gt;" not in html  # 잘못된 시각은 값 자체를 보여주지 않는다


def test_report_shows_default_origin_note():
    """출발지 줄이 표시된다(기본값이면 그 사실도 표시)."""
    html = render_report_html("맛집형", [_item(1, "아침")], 100000, None, None, 2, "서울 (입력이 없어 기본값)")
    assert "출발지: 서울 (입력이 없어 기본값)" in html
    assert "출발지:" not in render_report_html("맛집형", [_item(1, "아침")], 100000)


def test_report_shows_period_basis():
    """리포트 상단에 조회 기준 날짜가 표시된다(없으면 줄 자체가 없다)."""
    html = render_report_html("맛집형", [_item(1, "아침")], 100000, None, None, 2, "서울", "2026-12-20 ~ 2026-12-22 (입력)")
    assert "조회 기준 날짜: 2026-12-20 ~ 2026-12-22 (입력)" in html
    assert "조회 기준 날짜" not in render_report_html("맛집형", [_item(1, "아침")], 100000)


def test_overnight_flight_sorts_before_post_midnight_transfer_and_checkin():
    """18:10 출발 야간 항공(00:10 도착) 뒤의 00:40 공항→숙소 이동과 01:30 체크인은 항공 다음에 나온다."""
    import re

    from src.tools import render_report_html

    def row(activity, depart, arrive):
        return {"day": 1, "time_slot": "저녁", "activity": activity, "cost_krw": 1, "transport": "x", "note": "",
                "depart_time": depart, "arrive_time": arrive}

    items = [row("공항→숙소 이동", "00:40", "01:20"), row("숙소 체크인", "01:30", "01:45"), row("하남→싱가포르 항공", "18:10", "00:10")]
    html = render_report_html("맛집형", items, None)
    order = [m for m in re.findall(r"하남→싱가포르 항공|공항→숙소 이동|숙소 체크인", html)]
    assert order[:3] == ["하남→싱가포르 항공", "공항→숙소 이동", "숙소 체크인"]


def test_report_date_column_shows_month_day_before_day_number():
    """시작일이 있으면 일차 앞 열에 M/D(예: 3/20)가 일차마다 하루씩 늘어나며, 자정을 넘긴 항목은 다음 날짜가 된다."""
    import re
    from datetime import date

    items = [
        _item(1, "아침", "출발") | {"depart_time": "09:00"},
        _item(2, "아침", "관광") | {"depart_time": "09:00"},
        _item(1, "저녁", "야간 도착 이동") | {"depart_time": "00:40"},
    ]
    html = render_report_html("맛집형", items, None, start_date=date(2027, 3, 20))
    rows = [re.findall(r"<td>(.*?)</td>", r)[:2] for r in re.findall(r"<tr>(.*?)</tr>", html)[1:]]
    assert rows == [["3/20", "1일차"], ["3/21", "1일차"], ["3/21", "2일차"]]  # 00:40 은 자정 뒤라 3/21


def test_weather_summary_lists_period_range_rain_days_and_tips():
    """날씨 요약: 기간·기온 범위·주된 날씨·비 가능성 높은 날과 그에 맞는 준비 제안을 코드가 만든다."""
    from src.tools import weather_summary

    days = [
        {"date": "2026-11-22", "min": 26, "max": 31, "description": "맑음", "pop": 10},
        {"date": "2026-11-23", "min": 25, "max": 29, "description": "비", "pop": 80},
        {"date": "2026-11-24", "min": 25, "max": 30, "description": "맑음", "pop": 20},
    ]
    lines = weather_summary({"city": "Singapore", "in_range": True, "days": days})
    assert lines[0] == "예보 기간: 11/22 ~ 11/24"
    assert lines[1] == "기온: 25~31℃, 주된 날씨: 맑음"
    assert lines[2] == "비 가능성이 높은 날: 11/23(80%)"
    assert "우산·우비" in lines[3] and "자외선 차단제" in lines[3] and "겉옷" not in lines[3]
    dry = weather_summary({"days": [{"date": "2026-11-22", "min": 5, "max": 12, "description": "맑음", "pop": 0}]})
    assert dry[0] == "예보 기간: 11/22" and dry[2] == "강수 가능성은 낮은 편입니다(최대 0%)." and dry[3] == "준비 제안: 겉옷·방한 용품"
    assert weather_summary(None) == [] and weather_summary({"days": []}) == []


def test_report_weather_card_lives_in_prep_tab_and_marks_reference_forecast():
    """날씨 카드는 준비물·유의 사항 탭 맨 위에 오고, 여행 기간이 예보 범위 밖이면 참고용이라고 밝힌다. 예보가 없으면 카드가 없다."""
    days = [{"date": "2026-11-05", "min": 20, "max": 25, "description": "흐림", "pop": 30}]
    html = render_report_html("맛집형", [_item(1, "아침")], None, weather={"city": "Tokyo", "in_range": False, "days": days})
    schedule, prep = html.split("class='panel p3'>")
    assert "날씨 요약 (참고용) — Tokyo" in prep and "5일 예보 범위 밖" in prep and "날씨 요약" not in schedule
    assert "날씨 요약" not in render_report_html("맛집형", [_item(1, "아침")], None)
    assert "날씨 요약 — Tokyo" in render_report_html("맛집형", [_item(1, "아침")], None, weather={"city": "Tokyo", "in_range": True, "days": days})


def test_report_places_tab_shows_sight_image_with_credit_and_escapes():
    """2번째 탭: 이미지가 있는 명소 카드에만 썸네일·출처가 나오고, 없거나 허용되지 않은 주소면 기존 카드 그대로다."""
    guide = {"places": [
        {"kind": "명소", "name": "센소지", "area": "아사쿠사", "description": "사찰", "tip": "아침에 가세요",
         "image": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/x.jpg/480px-x.jpg", "title": "센소지 <b>", "page": "p"}},
        {"kind": "명소", "name": "스카이트리", "area": "스미다", "description": "타워", "tip": "t"},
        {"kind": "명소", "name": "수상한", "area": "a", "description": "d", "tip": "t", "image": {"url": "https://evil.example.com/x.jpg", "title": "x"}},
        {"kind": "맛집", "name": "이치란", "area": "신주쿠", "description": "라멘", "tip": "t"},
    ]}
    html = render_report_html("맛집형", [_item(1, "아침")], None, guide)
    tab2 = html.split("class='panel p2'>")[1].split("class='panel p3'>")[0]
    from src.tools import DEFAULT_SIGHT_IMAGE

    assert tab2.count("<img ") == 3 and "has-img" in tab2  # 센소지(위키 이미지) + 이미지 없는 명소 둘(기본 이미지). 맛집은 이미지 없음
    assert tab2.count(f"src='{DEFAULT_SIGHT_IMAGE}'") == 2 and tab2.count("대표 이미지 없음") == 2
    assert tab2.count("https://upload.wikimedia.org/") == 1  # 허용되지 않은 주소(evil)는 기본 이미지로 바뀐다
    assert "alt='센소지'" in tab2 and "loading='lazy'" in tab2 and "referrerpolicy='no-referrer'" in tab2
    assert "이미지: 위키백과 「센소지 &lt;b&gt;」" in tab2 and "<b>" not in tab2.split("figcaption")[1]  # 출처 문구도 이스케이프
    assert "evil.example.com" not in html and "이치란" in tab2
    assert tab2.split("<h3>명소</h3>")[0].count("<img ") == 0  # 맛집 카드에는 기본 이미지도 넣지 않는다


def test_report_shows_destination_chip():
    """리포트 상단에 파싱된 여행지가 보여 잘못 읽었을 때 알아볼 수 있다."""
    html = render_report_html("맛집형", [_item(1, "아침")], 100000, destination="장가계")
    assert "여행지: 장가계" in html
    assert "여행지:" not in render_report_html("맛집형", [_item(1, "아침")], 100000)
