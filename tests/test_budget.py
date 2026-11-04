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
