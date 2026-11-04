"""AC6: POST /query 가 그래프를 구동하고 interrupt 를 재개하는지 테스트 (가짜 모델, 네트워크 없음)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from src.agent import build_graph  # noqa: E402
from src.api import create_app, parse_selection, parse_trip  # noqa: E402
from tests.test_plans import FakeModel, make_output  # noqa: E402

Q = "도쿄, 2박 3일, 100만원으로 여행 일정 짜줘"


def make_client(planner=None, script=None):
    """가짜 모델 그래프를 쓰는 테스트 클라이언트."""
    planner = planner or (lambda p: make_output(50_000, days=3))
    graph = build_graph(FakeModel(planner=planner, script=script or []))
    return TestClient(create_app(graph), raise_server_exceptions=False)


def test_parse_trip_and_selection():
    assert parse_trip(Q) == {"destination": "도쿄", "days": 3, "budget_krw": 1_000_000}
    assert parse_trip("예산 1만원으로 제주 5박 6일 계획해줘") == {"destination": "제주", "days": 6, "budget_krw": 10_000}
    assert parse_trip("100만원으로 3박 4일 해외여행 가고 싶은데")["destination"] == ""
    assert parse_selection("2번 안으로 할게") == "명소형"
    assert parse_selection("가성비형으로") == "가성비형"
    assert parse_selection("글쎄") is None


def test_first_request_returns_three_keys_and_session_id():
    r = make_client().post("/query", json={"question": Q})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"answer", "contexts", "trace"}
    assert set(body["answer"]) == {"session_id", "plainAnswer", "html"}
    assert body["answer"]["session_id"] and body["answer"]["html"] == ""
    assert any(c["tool"] == "check_budget" for c in body["contexts"])
    assert body["trace"][0] == "validate_request" and body["trace"][-1] == "collect_plans"


def test_select_then_chat_then_done_resume_by_pending_interrupt():
    client = make_client()
    sid = client.post("/query", json={"question": Q}).json()["answer"]["session_id"]
    sel = client.post("/query", json={"question": "2번 안으로 할게", "session_id": sid}).json()
    assert sel["answer"]["session_id"] == sid and sel["answer"]["html"]
    assert sel["trace"][-1] == "render_answer"
    chat = client.post("/query", json={"question": "2일차 저녁을 스시로 바꿔줘", "session_id": sid})
    assert chat.status_code == 200 and "chat_edit" in chat.json()["trace"]
    done = client.post("/query", json={"question": "종료", "session_id": sid})
    assert done.status_code == 200
    again = client.post("/query", json={"question": "또", "session_id": sid})
    assert again.status_code == 404


def test_errors_422_404_500():
    client = make_client()
    for payload in ({"question": ""}, {"question": "   "}, {}, {"question": 5}):
        r = client.post("/query", json=payload)
        assert r.status_code == 422 and set(r.json()["error"]) == {"code", "message"}
    assert client.post("/query", json={"question": "도쿄 여행 알려줘"}).status_code == 422  # 기간·예산 없음
    r = client.post("/query", json={"question": "안녕", "session_id": "nope"})
    assert r.status_code == 404 and set(r.json()["error"]) == {"code", "message"}
    sid = client.post("/query", json={"question": Q}).json()["answer"]["session_id"]
    assert client.post("/query", json={"question": "글쎄", "session_id": sid}).status_code == 422

    def boom(prompt):
        raise RuntimeError("bug")

    r = make_client(boom).post("/query", json={"question": Q})
    assert r.status_code == 500 and set(r.json()["error"]) == {"code", "message"}
