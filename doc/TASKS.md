# 여행 일정 에이전트 웹앱 — TASKS

> 기준 문서: `doc/SERVICE.md`(제품 결정), `doc/DESIGN.md`(상세 설계), `CLAUDE.md`(제출 규약).
> 작업 하나는 30분~1시간 안에 끝나도록 나눴습니다. 예상 시간은 30m / 45m / 60m 중 하나입니다.
> 모든 자동 테스트는 Bedrock을 모킹하고 네트워크를 쓰지 않습니다.

## 진행 규칙

- 순서는 **기반 → 결정적 도구 → 그래프 → API → UI → 평가** 입니다. `선행` 열의 작업이 끝난 뒤에 시작합니다.
- 작업마다 `완료 기준`을 만족하면 끝입니다. 테스트가 있는 작업은 해당 테스트가 통과해야 합니다.
- 파일 규칙(CLAUDE.md): 한 파일에 한 역할, 한국어 docstring, 비밀 값은 `.env`, 요청하지 않은 파일을 새로 만들지 않음, 기존 파일은 통째로 다시 쓰지 않고 바뀐 부분만 고침.
- `승인 필요` 표시는 고정 구조에 없는 새 파일입니다. 사용자 허가를 받은 뒤 만듭니다 (본문 9절).

## 1. 기반 (A)

| ID | 작업 | 파일 | 완료 기준 | 예상 | 선행 |
|---|---|---|---|---|---|
| A1 | 의존성 목록과 `.env` 예시, `.gitignore` 작성 (`승인 필요`) | `requirements.txt`, `.env.example`, `.gitignore` | LangChain, LangGraph, langchain-aws, FastAPI, Streamlit, pytest 포함. `.env` 는 git 제외 | 30m | - |
| A2 | 설정 로더 구현 | `src/config.py`, `tests/test_config.py` | `AWS_DEFAULT_REGION`, `BEDROCK_MODEL_ID` 를 `.env` 에서 읽음. 값이 코드에 없음. `pytest tests/test_config.py` 통과 (AC5) | 45m | A1 |
| A3 | 도메인 타입 정의 (`TripRequest`, `ItineraryItem`, `PlanType`, `PlanState`) | `src/tools.py` 상단 | DESIGN 3절의 타입과 일치(인원·출발지·시작일, 시각·1인 단가 선택 필드 포함). `time_slot` 은 아침/오후/저녁 고정 | 30m | - |

## 2. 결정적 도구 (B) — LLM 호출 없음

| ID | 작업 | 파일 | 완료 기준 | 예상 | 선행 |
|---|---|---|---|---|---|
| B1 | `target_budget`, `total_cost` | `src/tools.py`, `tests/test_budget.py` | 맛집형·명소형 = 예산, 가성비형 = 예산×0.8. 합계 계산이 맞음 | 30m | A3 |
| B2 | `check_budget` | `src/tools.py`, `tests/test_budget.py` | `{within, total, overage}` 반환. 경계값(합계 = 목표)은 충족 | 30m | B1 |
| B3 | `closest_attempt` | `src/tools.py`, `tests/test_budget.py` | 시도 목록에서 초과 금액이 가장 작은 시도를 고름. 동률·빈 목록 처리 | 45m | B2 |
| B4 | `render_plain_answer` | `src/tools.py`, `tests/test_plans.py` | 3안 요약, 선택 안 요약, 수정 결과 텍스트 생성 | 45m | B2 |
| B5 | `render_report_html` — 일정표·합계·잔액 | `src/tools.py`, `tests/test_report.py` | 일차·시간대별 일정, 항목별 활동·비용·교통·비고, 합계, 잔액(예산 − 합계) 포함 | 60m | B2 |
| B6 | `render_report_html` — 경고·안내·이스케이프 | `src/tools.py`, `tests/test_report.py` | 초과 경고, 가장 근접한 안 문구, AI 추정 안내 문구 포함. LLM 문자열이 이스케이프됨. `pytest tests/test_report.py` 통과 (AC4) | 45m | B5 |
| B7 | mock 비용 샘플 작성과 로더 (`확인 필요`: 9절 2번) | `data/`, `src/tools.py` | 항목별 원화 비용 샘플 파일과 읽기 함수. 형식은 확정 후 결정 | 45m | A3 |

## 3. 에이전트 그래프 (C)

| ID | 작업 | 파일 | 완료 기준 | 예상 | 선행 |
|---|---|---|---|---|---|
| C1 | `GraphState` 와 누적 리듀서 | `src/agent.py` | `chat_history`(`add_messages`), `tool_results`, `trace`, `errors` 가 누적됨. 상태에 프롬프트·파생 값 없음 | 30m | A3 |
| C2 | Bedrock 모델 팩토리와 구조화 출력 스키마 | `src/agent.py`, `src/config.py` 사용 | `ChatBedrockConverse` 를 `config` 값으로 생성. 테스트에서 가짜 모델로 교체 가능 | 45m | A2, C1 |
| C3 | `validate_request` 노드 | `src/agent.py` | 여행지 누락, 예산 0 이하 등 오류 시 오류 상태 반환. 입력 범위 제한 없음 | 30m | C1 |
| C4 | `generate_plan` 노드 (프롬프트 빌더, 구조화 출력, 오류 시 `errors` 기록) | `src/agent.py` | 가짜 모델로 안 1개 생성. 파싱 실패 시 1회 되돌림 | 60m | C2, C3 |
| C5 | `check_budget` 노드와 조건부 라우터 | `src/agent.py` | 충족 → collect, 초과·시도 < 3 → readjust, 초과·시도 = 3 → pick_closest 로 분기 | 45m | B2, C1 |
| C6 | `readjust_plan` 노드 | `src/agent.py` | 초과 금액을 프롬프트에 넣어 재생성. 시도가 `attempts` 에 쌓임 | 45m | C4, C5 |
| C7 | `pick_closest`, `collect_plans` 노드 | `src/agent.py` | 가장 근접한 안 선택, 3안을 `plans` 에 모음 | 30m | B3, C5 |
| C8 | `Send` 팬아웃과 3안 생성 그래프 조립 | `src/agent.py`, `tests/test_plans.py` | 안 3개가 병렬 생성되어 `collect_plans` 로 모임. `pytest tests/test_plans.py` 통과 (AC1) | 60m | C6, C7 |
| C9 | 재조정 루프 테스트 | `tests/test_readjust.py` | 안별 최대 3회, 3회 후 가장 근접한 안과 경고 확인. 통과 (AC2) | 45m | C8 |
| C10 | `await_selection` (interrupt), `InMemorySaver`, `thread_id` 연결 | `src/agent.py` | 3안 후 interrupt. `Command(resume=...)` 로 재개해 선택 반영 | 60m | C8 |
| C11 | `chat_edit` 노드 (`create_agent` + `get_plan_items`, `submit_plan_items`) | `src/agent.py` | 선택 안이 수정되어 `selected_items` 에 저장. 에이전트는 합계를 말하지 않음 | 60m | C2, C10 |
| C12 | `check_budget_selected`, `readjust_selected` | `src/agent.py` | 수정 후 초과 시 재조정. 시도 횟수는 수정 1회마다 새로 3회 | 45m | C6, C11 |
| C13 | `await_chat`, `render_answer`, 종료 분기와 전체 그래프 조립 | `src/agent.py` | 대화 대기 → 수정 → 렌더 → 다시 대기. `done` 이면 END | 60m | B4, B6, C12 |
| C14 | 대화 수정 테스트 | `tests/test_chat_edit.py` | 수정 반영, 초과 시 재조정, 요청한 부분만 변경 확인. 통과 (AC3) | 45m | C13 |
| C15 | `RetryPolicy`·timeout 적용 | `src/agent.py` | LLM 노드에 일시 오류 재시도. 예산 재조정 횟수와 분리됨. 일시 오류 테스트 통과 | 45m | C4, C6, C11 |
| C16 | 일부 안 실패 처리 | `src/agent.py` | 성공한 안만 표시. 3안 모두 실패하면 오류. 테스트 통과 | 30m | C8 |

## 4. API (D) — `src/api.py`

| ID | 작업 | 파일 | 완료 기준 | 예상 | 선행 |
|---|---|---|---|---|---|
| D1 | 요청·응답 모델과 `session_id` 발급 | `src/api.py` | `question`, `session_id`(선택) 수신. 첫 요청에서 새 ID 발급. `answer` 안에 `session_id` 포함 | 45m | C13 |
| D2 | `/query`: 대기 중인 interrupt 종류에 따라 resume 값 결정 | `src/api.py` | 선택 대기 → select, 대화 대기 → chat, 종료 → done. 자연어 분류에 LLM을 쓰지 않음 | 60m | D1 |
| D3 | 응답 조립 (`answer`, `contexts`, `trace`) | `src/api.py` | `answer` 는 `{session_id, plainAnswer, html}`. `contexts` 는 도구 결과, `trace` 는 노드 이름 | 45m | D2 |
| D4 | 오류 응답과 상태 코드, 테스트 | `src/api.py`, `tests/` | `{"error": {"code", "message"}}`. 422(빈 `question`), 404(모르는 `session_id`), 500. 테스트 통과 | 45m | D3 |

## 5. 검증용 UI (E) — `src/ui.py`

| ID | 작업 | 파일 | 완료 기준 | 예상 | 선행 |
|---|---|---|---|---|---|
| E1 | 입력 폼, `/query` 호출, 세션 유지 | `src/ui.py` | 입력 → 호출. `session_id` 를 `st.session_state` 에 보관 (같은 탭 동안) | 45m | D3 |
| E2 | 3안 표시, 선택, 대화 입력 | `src/ui.py` | 3안을 보여주고 선택·대화 요청을 `/query` 로 보냄. 로직은 UI에 두지 않음 | 60m | E1 |
| E3 | 리포트 `html` 표시와 AI 추정 안내 문구 | `src/ui.py` | `answer.html` 을 화면에 렌더. 일정·가격이 AI 추정값이라는 안내가 항상 보임 | 30m | E2 |

## 6. 평가 (F)

| ID | 작업 | 파일 | 완료 기준 | 예상 | 선행 |
|---|---|---|---|---|---|
| F1 | 자동 점검 스크립트 (일수·예산·필수 장소) (`승인 필요`) | `evaluation/run_eval.py` | `test_queries.csv` 의 `expected_traits` 중 자동 점검 항목을 12건에 적용 | 60m | D4 |
| F2 | PM 체크리스트와 평가 리포트 양식 | `evaluation/` | 체크 3항목(불필요한 왕복 없음, 식사 시간대 식사, 실존 장소로 보임)과 결과 기록 양식 | 45m | - |
| F3 | 12건 실행과 결과 기록 | `evaluation/` | 12건을 1회씩 실행해 자동 점검·PM 체크리스트 결과 기록 | 60m | E3, F1, F2 |
| F4 | 실패 건 수정과 재평가 (여유 시간) | 해당 파일 | 12/12 통과 (PRD 합격선) | 60m | F3 |

- 진행 현황: F1~F3 은 구현됨 (`evaluation/run_eval.py`, 회차별 `round*_report.md`, `eval_results_*.csv`, `eval_summary_*.json`). 가장 최근 5회차 자동 채점은 7/12 통과이고 실패 건은 500 오류와 도구 미호출입니다. F4(12/12)는 아직 완료되지 않았습니다.

## 6-2. Guardrail (G) — DESIGN 16절

규칙 함수는 `src/tools.py`, 연결은 `src/api.py`, 테스트는 `tests/test_api.py` 에 둡니다. 새 파일은 만들지 않습니다.

| ID | 작업 | 파일 | 완료 기준 | 예상 | 선행 |
|---|---|---|---|---|---|
| G1 | `classify_request`: 비밀 값·개인정보·실시간 조회 거절 규칙 | `src/tools.py`, `tests/test_api.py` | TQ-06·07·11·12 입력이 각각 `unsupported_lookup`, `secret_request`, `personal_data_request` 로 분류됨. 일반 여행 요청은 통과 | 45m | D4 |
| G2 | 필수 정보 누락 검사와 `known_trip` 병합 | `src/tools.py`, `tests/test_api.py` | 여행지·기간·예산 중 빠진 항목 목록을 반환. 이전 값과 새 입력을 합침 | 45m | G1 |
| G3 | 조건 충돌 감지 규칙 | `src/tools.py`, `tests/test_api.py` | TQ-10 입력에서 충돌을 감지. 충돌이 없는 입력은 오탐하지 않음 | 30m | G1 |
| G4 | `/query` 에 Guardrail 연결: 422 코드, `known_trip` 서버 메모리, `session_id` 이어가기, 충돌 안내 | `src/api.py`, `tests/test_api.py` | 거절·누락은 `{"error": {"code", "message", ...}}` 422. 누락 응답에 `session_id`, `missing` 포함. 같은 `session_id` 로 보완 입력 시 그래프 시작. 충돌은 200 + 안내 | 60m | G2, G3 |
| G5 | 여행지 파서 보정과 TQ-06~12 점검 테스트 | `src/api.py`, `tests/test_api.py` | '정확한' 등 수식어를 장소로 읽지 않음. 16절 대응표의 6개 시나리오가 기대한 코드·상태로 응답 | 45m | G4 |

## 6-3. 확장 (H) — 구현 후 추가된 기능

인터뷰 이후 구현 단계에서 추가된 기능입니다. 모두 구현되어 테스트가 통과합니다 (`python -m pytest tests -q`, 157개). 외부 연동(H3~H5, H7의 이미지)은 `MOCKDATA_USE=False` 일 때만 켜지고, 실패하면 LLM 추정으로 대체합니다. 새 파일은 만들지 않았습니다 (`web/index.html` 은 CLAUDE.md 고정 구조).

| ID | 작업 | 파일 | 완료 기준 | 예상 | 선행 |
|---|---|---|---|---|---|
| H1 | 인원·출발지·날짜 파싱과 필수·선택 입력 | `src/api.py`, `src/tools.py`, `tests/test_api.py` | `parse_people`, `parse_route`, `parse_dates` 가 인원·구간·날짜를 규칙으로 읽음. 필수는 여행지·기간·인원, 예산·출발지(기본 서울)·날짜(기본 D+30)는 선택 | 45m | G5 |
| H2 | 인원 비용 계산과 예산 선택 | `src/tools.py`, `src/agent.py`, `tests/test_budget.py` | `apply_headcount` 가 1인 단위 항목을 `1인 가격 × 인원` 으로 계산. 숙박은 2명당 1실 올림, 1일 여행은 숙박 없음. 예산이 없으면 `check_budget` 이 항상 충족 | 60m | B2, H1 |
| H3 | 환율·날씨 Tool과 `research_conditions` | `src/tools.py`, `src/agent.py`, `tests/test_plans.py` | `get_exchange_rate`, `get_weather_forecast` 가 실패 시 `{"error": …}`. 노드가 해외 여부·통화·날씨 검색어를 정해 호출하고 프롬프트에 반영 | 60m | C4, A2 |
| H4 | MCP 연결과 `research_places`, `research_prices` | `src/tools.py`, `src/agent.py`, `src/config.py`, `tests/test_plans.py` | Google Maps·마이리얼트립 MCP 로 장소(평점·리뷰 수)와 가격을 조회. 키·연결 실패 시 해당 노드만 건너뛰고 진행 | 60m | H3 |
| H5 | 경로 조회 (`lookup_routes`, `lookup_domestic_routes`) | `src/tools.py`, `src/agent.py`, `tests/test_plans.py` | 해외는 Google Maps, 국내는 네이버로 구간·소요시간·통행료를 채움. 비용·시각 등 원래 값은 유지. 가이드와 병렬 실행 | 60m | H4, C13 |
| H6 | `compare_plans` 와 3안 카드 정보 | `src/tools.py`, `src/agent.py`, `tests/test_budget.py` | 차별점 2~3문장·250자 이내(코드 검사, 폴백 문장). `budget_usage_line`, `main_schedule`(식당은 평점 4.5 이상·리뷰 5,000개 이상일 때 최대 1개)이 `plainAnswer` 에 포함 | 60m | C8, H4 |
| H7 | `generate_guide`, 위키백과 이미지, 3탭 리포트 | `src/tools.py`, `src/agent.py`, `tests/test_report.py` | 일정표·가이드북·준비물·유의 사항 3탭. 날씨 요약 카드, 명소 대표 이미지(없으면 기본 이미지), 이미지 주소는 위키미디어 https 만 허용 | 60m | B6, H3 |
| H8 | 웹 UI와 `POST /query/stream`, `GET /` | `web/index.html`, `src/api.py`, `tests/test_api.py` | NDJSON 진행 표시, 3안 카드 선택 버튼, 선택 전·종료 후 입력 잠금, 리포트 iframe 표시, "HTML로 내보내기"(이미지 포함) | 60m | D3, H6, H7 |
| H9 | 선택 단계 규칙 | `src/api.py`, `tests/test_api.py` | `parse_selection`(`1번`·`3안`·숫자만·안 이름)과 수정 표현 거절(422 `edit_before_selection`), 못 읽으면 422 `invalid_selection`. 모르는·종료된 세션은 404 | 30m | D2 |

## 7. 의존 순서 요약

```
A1 ─ A2 ─┐
A3 ─ B1 ─ B2 ─ B3 ─────────────┐
      │      └ B4, B5 ─ B6 ─────┤
      └ C1 ─ C3 ─ C4 ─ C6 ──────┤
          └ C5 ───── C7 ─ C8 ─ C9
A2 ─ C2 ─┘                │
                          C10 ─ C11 ─ C12 ─ C13 ─ C14
                                          │
                          C15, C16 ───────┤
                                          D1 ─ D2 ─ D3 ─ D4 ─ F1 ─┐
                                                      E1 ─ E2 ─ E3 ─ F3 ─ F4
                                                                 F2 ─┘

G5 ─ H1 ─ H2           H3 ─ H4 ─ H5
          C4 ─ H3      C8 ─ H6 ─┐
                       B6 ─ H7 ─┴─ H8   (D3 → H8, D2 → H9)
```

## 8. 작업량 요약

| 구분 | 작업 수 | 예상 시간 |
|---|---|---|
| 기반 (A) | 3 | 1시간 45분 |
| 결정적 도구 (B) | 7 | 5시간 |
| 그래프 (C) | 16 | 12시간 15분 |
| API (D) | 4 | 3시간 15분 |
| UI (E) | 3 | 2시간 15분 |
| 평가 (F) | 4 | 3시간 45분 |
| Guardrail (G) | 5 | 3시간 45분 |
| 확장 (H) | 9 | 8시간 15분 |
| **합계** | **51** | **약 40시간 15분** |

## 9. 사용자 확인이 필요한 항목

1. **새 파일 허가**: `requirements.txt`, `.env.example`, `.gitignore` (A1), `evaluation/run_eval.py` (F1). 고정 구조에 없고 CLAUDE.md는 요청하지 않은 파일 생성을 금지합니다. 현재 `requirements.txt`, `.gitignore`, `evaluation/run_eval.py` 는 만들어졌고 `.env.example` 은 없습니다.
2. **`data/` mock 데이터의 사용 방식**: SERVICE는 항목별 비용 샘플을 두라고 했지만, 코드에서 어떻게 쓸지(프롬프트 참고 자료인지, 테스트 입력인지)는 정해지지 않았습니다. 그 사이 외부 조회(H3~H5)로 가격을 채우게 되어 `data/` 폴더는 만들지 않았고 B7은 보류 상태입니다.
3. **도메인 타입 위치**: 순환 import를 피하려고 `src/tools.py` 상단(A3)에 두었습니다. `src/agent.py` 에 두면 `tools.py` 가 `agent.py` 를 가져와야 하므로 구조가 꼬입니다. 다른 위치를 원하면 알려주세요.
4. **PRD 열린 항목**: 대화 수정 후 일수·필수 장소·예산 표시 유지 여부, 화면 언어(한국어만인지)는 아직 답이 없어 C14, E1~E3의 세부 기준이 바뀔 수 있습니다.
5. **Seed**: 이 작업 목록 기준으로 `seed_ec1db24038dc` 를 재생성했습니다. 기존 `seed_ebc6e7d8804c` 는 대체되었습니다. 위 1~4번이 확정되면 필요에 따라 Seed를 다시 생성합니다.
