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
