"""Streamlit 검증 UI: 입력 → 3안 → 선택 → 대화 수정 → 리포트 흐름을 POST /query 로만 구동한다 (에이전트 로직 없음)."""
import os

import requests
import streamlit as st
import streamlit.components.v1 as components

API_URL = os.environ.get("TRAVEL_API_URL", "http://localhost:8000")
ESTIMATE_NOTICE = "⚠️ 화면의 일정과 가격은 AI가 추정한 값이며 실시간 조회 결과가 아닙니다. 실제 예약 전 직접 확인하세요."


def call_query(question: str, session_id: str | None) -> dict:
    """POST /query 를 호출해 JSON 을 돌려준다. 실패하면 {"error": {...}} 형태를 돌려준다."""
    body = {"question": question}
    if session_id:
        body["session_id"] = session_id
    try:
        resp = requests.post(f"{API_URL}/query", json=body, timeout=600)
        return resp.json()
    except (requests.RequestException, ValueError) as exc:
        return {"error": {"code": "connection_error", "message": f"서버에 연결하지 못했습니다: {exc}"}}


def send(question: str) -> None:
    """질문을 보내고 응답을 세션 상태에 반영한다. session_id 는 answer 안에서 읽어 같은 탭 동안 보관한다."""
    data = call_query(question, st.session_state.get("session_id"))
    st.session_state["history"].append(("user", question))
    if "error" in data:
        st.session_state["error"] = data["error"]["message"]
        return
    st.session_state["error"] = None
    answer = data["answer"]
    st.session_state["session_id"] = answer["session_id"]
    st.session_state["plain"] = answer["plainAnswer"]
    st.session_state["html"] = answer.get("html", "")
    st.session_state["history"].append(("assistant", answer["plainAnswer"]))


def main() -> None:
    """화면을 그린다: 안내 문구(항상 표시), 입력 폼, 3안/선택, 대화 입력, 리포트."""
    st.set_page_config(page_title="여행 일정 에이전트", layout="wide")
    st.title("여행 일정 에이전트")
    st.warning(ESTIMATE_NOTICE)
    for key, default in (("session_id", None), ("history", []), ("plain", ""), ("html", ""), ("error", None)):
        st.session_state.setdefault(key, default)

    if st.session_state["session_id"] is None:
        with st.form("trip_form"):
            destination = st.text_input("여행지", value="도쿄")
            nights = st.number_input("숙박 일수(박)", min_value=1, value=3, step=1)
            budget_man = st.number_input("예산(만원)", min_value=1, value=100, step=10)
            if st.form_submit_button("3안 만들기"):
                send(f"{destination}, {nights}박 {nights + 1}일, {budget_man}만원으로 여행 일정 짜줘")
                st.rerun()
    else:
        if st.button("새 일정 시작"):
            for key in ("session_id", "plain", "html", "error"):
                st.session_state[key] = None if key in ("session_id", "error") else ""
            st.session_state["history"] = []
            st.rerun()

    if st.session_state["error"]:
        st.error(st.session_state["error"])

    if st.session_state["plain"]:
        st.subheader("현재 응답")
        st.text(st.session_state["plain"])

    if st.session_state["session_id"] and not st.session_state["html"]:
        cols = st.columns(3)
        for i, name in enumerate(("맛집형", "명소형", "가성비형"), start=1):
            if cols[i - 1].button(f"{i}번 {name} 선택"):
                send(f"{i}번 안으로 할게")
                st.rerun()

    if st.session_state["html"]:
        st.subheader("선택한 안의 리포트")
        components.html(st.session_state["html"], height=900, scrolling=True)
        st.caption(ESTIMATE_NOTICE)
        st.caption("수정할 내용을 입력하거나 '종료'를 입력해 마칩니다.")

    if st.session_state["session_id"]:
        prompt = st.chat_input("대화로 일정 수정 (예: 2일차 저녁을 스시로 바꿔줘)")
        if prompt:
            send(prompt)
            st.rerun()


main()
