# 여행 일정 에이전트 — 상세 설계 (구현 기준)

이 문서는 **현재 구현된 코드**를 요청이 흘러가는 순서대로 설명합니다. 제품 결정은 `doc/SERVICE.md`, 설계 의도와 결정 이유는 `doc/DESIGN.md`·`doc/langgraph-design.md`, 제출 규약은 `CLAUDE.md`가 기준입니다. 이 문서와 그 문서들이 다르면 코드가 맞는 쪽입니다(`DESIGN.md`·`SERVICE.md`는 환율·날씨 도구, 가이드 병렬 생성, 선택 전 수정 거절 등 구현 변경을 2026-11-06 기준으로 반영했습니다).

## 1. 구성

| 파일 | 역할 |
|---|---|
| `src/api.py` | FastAPI 앱. 요청 문장 파싱, 규칙 기반 가드레일, 세션 재개, 응답 조립, `GET /` |
| `src/agent.py` | LangGraph 그래프(노드·엣지·상태), 프롬프트, LLM 호출·재시도 |
| `src/tools.py` | 결정적인 도메인 로직(예산 판정, 비용 계산, 정렬, 렌더링)과 외부 연동 클라이언트·Tool |
| `src/config.py` | `.env`·환경변수 읽기(리전, 모델 ID, API 키). 코드에 값을 두지 않음 |
| `src/ui.py` | Streamlit 검증용 UI(`POST /query`만 호출, 에이전트 로직 없음) |
| `web/index.html` | HTML+JS+Tailwind 웹 UI. `GET /`로 제공, 진행 표시는 `POST /query/stream` |
| `tests/` | pytest. 모델은 가짜로 대체하고 네트워크를 쓰지 않음 |
| `evaluation/` | `test_queries.csv`(문항), `run_eval.py`(실제 모델로 채점), 결과 파일 |

기술 스택: Python, LangChain, LangGraph. 모델은 Amazon Bedrock `ChatBedrockConverse`, 에이전트는 `langchain.agents.create_agent`.

핵심 원칙 두 가지.
- **LLM은 추정만, 판단은 코드가 합니다.** LLM은 항목별 원화 금액을 추정하고 대화 수정 결과를 만듭니다. 합계, 예산 판정, 재조정 횟수, 가장 근접한 시도 선택, 인원 곱셈, 정렬은 모두 코드입니다.
- **상태는 원본만 저장합니다.** 합계·HTML·프롬프트는 상태에 두지 않고 필요할 때 계산합니다.

## 2. 전체 흐름 한눈에

```
브라우저(web/index.html)
  │ POST /query/stream {question, session_id?}
  ▼
api.process
  ├─ 새 요청:  classify_request(원문 규칙) → resolve_trip(모델 추출 + 빈 값은 parse_trip) → classify_request(가드레일) → 그래프 시작
  └─ 이어서:   체크포인트의 대기 stage 읽기
                 select → (수정 표현이면 거절) → 안 선택으로 재개
                 chat   → '종료'면 종료, 아니면 수정 요청으로 재개

그래프(agent.py)
  validate_request
    → [research_conditions]   환율·날씨 Tool        (연결된 경우)
    → [research_places]       Google Maps 장소 조사   (MCP 연결 시)
    → [research_prices]       마이리얼트립 가격 조사  (MCP 연결 시)
    → generate_plan ×3 (병렬)
    → check_budget ⇄ readjust_plan  (안마다 최대 3회)
    → [pick_closest] → collect_plans → compare_plans
    → await_selection ◀ interrupt(stage=select)
    → ┬ 경로 조회(lookup_routes | lookup_domestic_routes)  ─┐ 병렬
      └ generate_guide                                      ─┴→ render_answer
    → await_chat ◀ interrupt(stage=chat)
    → chat_edit → check_budget_selected ⇄ readjust_selected → render_answer → await_chat …
    → (종료) END
```
`[ ]`는 외부 연동이 연결됐을 때만 그래프에 들어가는 노드입니다(`MOCKDATA_USE=False`일 때 로드).

## 3. API 계층 (`src/api.py`)

### 3.1 엔드포인트
- `POST /query`: 처리가 끝난 뒤 `{answer, contexts, trace}`를 한 번에 반환. `answer`는 `{session_id, plainAnswer, html}`.
- `POST /query/stream`: 같은 처리를 하되 NDJSON으로 보냄. 노드가 끝날 때마다 `{"step": 노드}`, 마지막에 `{"result": …}` 또는 `{"error": …, "status": …}`. 워커 스레드가 큐에 넣고 응답 제너레이터가 읽음.
- `GET /`: `web/index.html`.
- 오류는 항상 `{"error": {"code", "message"}}`, 상태는 422/404/500.

### 3.2 새 요청 처리 순서
0. 비밀·개인정보·실시간 조회 요청은 모델을 부르기 전에 `classify_request`(원문 규칙)로 거절. 이어서 `resolve_trip`이 `extract_trip`(구조화 출력 `TripExtraction`)으로 여행지·일수·인원·예산·출발지·시작일을 모델에서 받고, 빈 항목은 규칙 파서 값으로 채움(모델이 없거나 실패하면 규칙 파서 결과를 그대로 사용).
1. `parse_trip(question)`은 규칙(정규식) 파서로, 빈 값 보완과 폴백에 씀. 날짜는 `parse_dates`(`12월 20일부터 23일까지`, `2026-12-20~22` 등), 인원은 `parse_people`, 구간은 `parse_route`.
2. `merge_trip`: 정보가 부족해 되묻는 중인 세션(`known_trips`, 서버 메모리)의 이전 값에 새 값을 덮어 합침.
3. `classify_request`(LLM 미사용, `tools.py`):
   - 비밀 요청(`secret_request`), 개인정보 요청(`personal_data_request`)은 모든 요청에 적용.
   - 실시간 조회 요청(`unsupported_lookup`), 필수 항목(여행지·기간·인원) 누락(`missing_fields`)은 새 요청에만 적용. 누락이면 `session_id`와 `missing`을 돌려주고 파싱값을 보관.
   - 모두 HTTP 422이며 그래프는 시작하지 않음. 예산은 선택 항목.
4. `has_conflict`(최고급+최소비용 동시 요구)이면 응답 앞에 안내 문구를 붙임.
5. `session_id`를 만들어 `thread_id`로 쓰고 그래프를 실행.

### 3.3 이어지는 요청 (세션 재개)
- `_pending_stage`가 체크포인트의 interrupt에서 `stage`(`select`/`chat`)를 읽음. 입력 종류를 별도 상태로 저장하지 않음.
- `select` 단계:
  - `has_edit_request`(변경·바꿔·수정 등)가 있으면 422 `edit_before_selection`으로 거절. 안을 고르기 전에는 수정 요청을 받지 않음.
  - `parse_selection`: `1번`·`3안`처럼 `번`·`안`이 붙었거나 숫자만 입력, 또는 안 이름(`가성비` 등)이 있을 때 선택으로 인식. `숙소를 2박으로`의 2는 선택이 아님. 못 찾으면 422 `invalid_selection`.
- `chat` 단계: `종료` 등(`DONE_WORDS`)이면 종료, 아니면 수정 요청으로 재개.
- 대기 중이 아니거나 종료된 세션은 404 `session_closed`, 모르는 세션은 404 `session_not_found`.

### 3.4 응답 조립 (`_build_response`)
상태의 `plans`가 비어 있으면 오류(검증 오류 → 422 `invalid_trip`, 그 외 500 `generation_failed`). 정상이면 `build_answer(state)`로 `plainAnswer`·`html`을 만들고, `contexts`는 `tool_results`, `trace`는 노드 실행 이름 목록.

## 4. 그래프 상태 (`GraphState`)

원본 데이터만 가진 `TypedDict(total=False)`. 주요 필드:

| 구분 | 필드 | 내용 |
|---|---|---|
| 입력 | `trip` | 여행지, 일수, 인원, 예산(없으면 None), 출발지, 시작일 |
| 조사 | `research`, `place_stats`, `price_research`, `overseas`, `fx`, `weather` | MCP·환율·날씨 조회값. 실패하면 비어 있고 LLM 추정으로 진행 |
| 3개 안 | `plans`, `comparison` | 안 종류별 `attempts`(원본+재조정 시도 목록), 안별 차별점 문장 |
| 선택 이후 | `selected`, `selected_items`, `selected_attempts`, `guide` | 고른 안, 현재 항목, 수정 1회당 재조정 시도, 가이드 원본 |
| 대화 | `chat_history`, `next_action` | 수정 요청 이력, 재개 값 |
| 진단 | `tool_results`, `trace`, `errors` | 응답의 `contexts`, `trace`, 오류 |

리듀서: `plans`는 안 종류 키로 병합(`_merge_plans`, 3안 병렬 생성 결과를 합침), `tool_results`·`trace`·`errors`는 이어 붙임(`_concat`), `chat_history`는 `add_messages`, 나머지는 덮어쓰기.

세션 상태는 `InMemorySaver` 체크포인터에 있으며 서버를 재시작하면 사라짐. DB·파일·로그인 없음.

## 5. 그래프 흐름 상세 (`src/agent.py`)

### 5.1 검증 → 사전 조회
- `validate_request`: 여행지 비어 있지 않음, 일수·인원은 1 이상의 정수, 예산은 생략하거나 양의 정수. 실패하면 `errors`를 남기고 종료.
- 검증 뒤 경로는 연결된 구성요소에 따라 달라짐(`route_after_validate` → `route_after_conditions`).
- **`research_conditions`**(환율·날씨 Tool이 있을 때): 모델이 `RegionOutput` 구조화 출력 한 번으로 해외 여부·통화 코드·날씨 검색어를 정하고(`_destination_profile`), 값은 Tool이 조회.
  - `get_exchange_rate`(Frankfurter, ECB 기준, 키 불필요): 해외이고 통화가 KRW가 아닐 때만 호출. 결과 `fx = {currency, krw_per_unit, date}`.
  - `get_weather_forecast`(OpenWeatherMap 5일 예보, `OPENWEATHER_API_KEY`): 여행 기간과 겹치는 날을 고르고, 겹치는 날이 없으면 `in_range=False`로 전체 예보를 참고용으로 둠.
  - 실패하면 Tool이 `{"error": …}`를 돌려주고 일정 생성은 계속.
- **`research_places`**(Google Maps MCP): 에이전트가 검색 도구로 실제 맛집·명소를 조사해 `research`와 평점·리뷰 수 `place_stats`를 만들고 해외 여부를 확정(이미 판정됐으면 재질문하지 않음).
- **`research_prices`**(마이리얼트립 MCP): 에이전트가 입장권·투어(`searchTnas`), 숙소(`searchStays`), 항공(`searchInternationalFlights`/`searchDomesticFlights`)을 조회해 `price_research`를 만듦. 날짜는 입력값, 없으면 오늘+30일(D+30). 결과는 `compact_price_result`로 줄여 LLM에 전달.
- 조사 단계가 실패해도 예외로 끝내지 않고 경고만 남기고 LLM 추정으로 진행.

### 5.2 3안 생성 (`generate_plan`, 병렬 3개)
- `fan_out`이 맛집형·명소형·가성비형을 `Send`로 동시에 실행. 각 Send에 `research`, `price_research`, `fx`, `weather`를 함께 전달.
- 프롬프트는 안 스타일(`PLAN_STYLE`), 여행 정보, 예산 문장(`_budget_phrase`), 규칙 모음으로 구성:
  - `_budget_phrase`: 안별 목표 총액, 1인당·1일당 목표, "1인 단위 항목은 1인 가격 × 인원으로 합산" 규칙, 항공·숙박을 먼저 정하고 남은 금액을 배분하라는 지시. 예산이 없으면 "예산 제한이 없으므로 금액 상한 없이".
  - `COST_RULES`(1인 단위/공용 비용 구분), `TIME_RULE`(HH:MM), `NOTE_RULE`(비고 40자), `CATEGORY_RULE`, `_flight_rule`(가는 편·귀국 편 두 항목), `_lodging_rule`(1일 여행은 숙박 없음, 그 외 2명당 1실 올림×박수로 총 숙박비 한 항목), `_conditions_rule`(환율 환산 지시, 여행 기간 날씨가 있으면 비 오는 날 실내 일정).
- `_call_structured`가 `PlanOutput` 구조화 출력을 받고 파싱 오류면 최대 3번 다시 시도. 실패하면 그 안은 `plans`에 넣지 않음.
- 성공하면 `apply_headcount`가 `per_person` 항목의 전체 비용을 `unit_cost_krw × 인원`으로 **코드가 다시 계산**하고 비고를 40자로 자름.

### 5.3 예산 판정과 재조정 (`check_budget` ⇄ `readjust_plan`)
- 목표 예산: 맛집형·명소형은 사용자 예산 이하, 가성비형은 예산의 80%(`VALUE_PLAN_RATIO`). 예산이 없으면 항상 충족(`target_budget`, `check_budget`).
- `check_budget_node`가 안마다 합계·초과액을 계산해 `tool_results`에 남김.
- `route_after_check`: 초과이고 재조정이 3회 미만인 안만 `readjust_plan`으로 `Send`. 재조정 횟수는 `len(attempts) - 1`(`needs_readjust`, `MAX_READJUST=3`). 초과인데 3회를 다 썼으면 `pick_closest`, 모두 충족이면 `collect_plans`.
- `readjust_plan`(LLM): 합계·목표·초과액과 비용이 큰 상위 3개 항목을 알려 주고 항목 금액을 줄여 다시 만들게 함. 결과에도 `apply_headcount` 적용. 구조화 출력이 실패해도 한 번의 시도로 세어 무한 루프를 막음.
- `pick_closest`/`final_items`: 충족한 시도가 있으면 그것, 없으면 초과액이 가장 작은 시도(`closest_attempt`, 동률이면 나중 시도). 이 경우 리포트에 경고 문구가 붙음.

### 5.4 비교와 선택 대기
- `collect_plans` 뒤 `route_after_collect`: 안이 하나도 없으면 종료(오류), 있으면 `compare_plans`.
- `compare_plans`: 안이 2개 이상이면 LLM이 안마다 다른 안과의 차별점을 2~3문장으로 작성. 길이·문장 수 검사(`fit_difference`)를 통과하지 못하거나 실패하면 수치 기반 문장(`fallback_difference`)으로 대체.
- `await_selection`: `render_plain_answer`로 3안 요약(안별 합계, 예산 대비, 차별점, 주요 일정)을 만들어 `interrupt({"stage": "select", …})`. `main_schedule`이 주요 일정 최대 5개를 코드로 고름(명소 위주, 식당은 평점 4.5 이상·리뷰 5,000개 이상일 때만 최대 1개).
- 재개 값이 유효하면 `selected`, `selected_items`(`final_items`), `selected_attempts`를 채움. 유효하지 않으면 다시 대기.

### 5.5 선택 후 병렬 처리와 리포트
- `route_after_selection`이 **경로 조회 노드와 `generate_guide`를 동시에** 시작(연결된 것만).
  - 해외 + Google Maps 연결: `lookup_routes` — 에이전트가 경로 도구로 구간별 출발지·도착지·소요시간·교통수단을 채움. 비용·시각·`per_person` 등은 원래 값을 유지.
  - 국내 + 네이버 연결: `lookup_domestic_routes` — 모델이 항목별 장소 검색어를 정하고 코드가 네이버 지역검색·Directions 15로 자동차 경로·시간·통행료·연료비를 채움(`apply_domestic_routes`). 항공·기차·귀국 항목은 건드리지 않음.
  - `generate_guide`: 맛집 4·명소 4, 준비물, 유의 사항, 목적지 사전 정보를 한 번에 구조화 출력으로 생성(`GuideOutput`). 이어서 이미지 조회기가 연결돼 있으면 명소마다 위키백과 대표 이미지를 병렬로 찾아 `place["image"]`로 붙임(`attach_place_images`). 맛집은 조회하지 않고, 실패해도 가이드는 그대로.
- 두 갈래는 모두 `render_answer`로 합류해 같은 단계에서 한 번만 실행됨. 경로 노드와 가이드는 서로 다른 상태 키를 써서 충돌이 없음.
- `render_answer`: `build_answer`로 `plainAnswer`와 HTML을 만들어 대화 이력에 남기고 `await_chat`으로 이어짐.

### 5.6 대화 수정 루프
- `await_chat`: 현재 선택 안을 `plainAnswer`로 보여 주며 `interrupt({"stage": "chat", …})`. 재개 값이 `chat`이면 수정, 아니면 종료.
- `chat_edit`: `create_agent` 에이전트가 `get_plan_items`로 현재 항목을 보고 요청된 부분만 고쳐 `submit_plan_items`로 전체 항목을 제출. 결과에 `apply_headcount` 적용, `selected_attempts`는 새로 시작(수정 1회마다 재조정 횟수 초기화). 제출이 없으면 기존 항목을 유지하고 오류를 기록.
- `check_budget_selected`: 수정 후 합계·판정. 초과이고 재조정 3회 미만이면 `readjust_selected`(초과액과 비용이 큰 항목을 알리고 사용자 수정 내용은 유지하며 금액 축소)로 루프, 다 쓰면 가장 근접한 시도를 선택.
- 이후 `render_answer` → `await_chat`을 반복. `종료`면 `END`.
- 가이드는 선택 직후 한 번만 만들며 채팅 수정 뒤에는 다시 만들지 않음.

### 5.7 LLM 호출 안정성
- Bedrock 읽기 타임아웃 120초, 연결 10초(`BEDROCK_READ_TIMEOUT`, `BEDROCK_CONNECT_TIMEOUT`). 출력 토큰 상한 16,384.
- `LLM_RETRY`: 노드 단위로 최대 3회(초기 대기 2초, 배수 3). 재시도 대상은 시간 초과, 쓰로틀링·서비스 일시 오류이고, 값·타입 오류 같은 코드 문제는 재시도하지 않음(`is_retryable`).
- 구조화 출력 파싱 실패는 별도로 최대 3회(`STRUCTURED_ATTEMPTS`).

## 6. 비용·시간 규칙 (`src/tools.py`)

- **통화**: 모든 금액은 KRW. 환율이 있으면 LLM이 현지 통화 단가(`local_unit_price`, `local_currency`)를 쓰고 코드(`finalize_items`)가 환율을 곱해 원화로 바꿈(통화 코드가 환율의 통화와 같을 때만). 환율이 없으면 LLM이 원화로 추정. 합계는 코드. 마이리얼트립 가격은 원화로 그대로 사용.
- **항공·숙소 가격**: `research_prices`가 마이리얼트립 도구 호출 결과를 요청별 sink(`quote_tools(sink)`)에 구조화해 `price_quotes`로 상태에 남김(`parse_price_quotes`; 항공은 조회 합계를 승객 수로 나눠 1인 왕복 요금, 숙소는 `원/박` 요금). `finalize_items`의 0단계 `apply_price_quotes`가 LLM 금액을 조회값으로 덮어씀.
  - 항공: '항공' 항목(귀국 아님)에 1인 왕복 요금·가는 편 시각·`항공사 편명 M/D` 비고, '귀국 항공' 항목에 오는 편 시각·비고. 후보는 한 번만 골라 두 항목에 같이 적용. 가성비형은 최저가, 그 밖은 직항 중 최저가(직항이 없으면 최저가), 항목에 적힌 항공사가 후보에 있으면 그 항공사 우선. 항공 항목이 없으면 새로 넣지 않음.
  - 숙소: `체크인` 항목의 숙소 이름이 조회한 숙소 이름과 포함 관계(공백·기호 무시, 3자 이상)면 그 요금을 객실 1박 요금으로 사용. 맞지 않으면 LLM 값 유지.
  - 3안 생성·재조정·대화 수정·선택 후 재조정이 모두 같은 함수를 지나므로, 재조정이 조회가 확정한 항공·숙소 금액을 낮출 수는 없고 숙소는 조회 목록의 더 싼 곳으로 바꾸는 식으로만 줄일 수 있음.
- **식사 가격**: `finalize_items`가 식당 항목의 활동명에 들어 있는 장소(`place_stats_for`)의 Google 가격대(`price_level`)를 보고, 아는 경우 LLM 금액 대신 `MEAL_PRICE_BY_LEVEL`(INEXPENSIVE 1만, MODERATE 2만, EXPENSIVE 5만, VERY_EXPENSIVE 10만원)을 1인 가격으로 확정. 비고 앞에 `Google 가격대 … 기준`을 붙임. 가격대를 모르면 LLM 1인 단가를 쓰고 3천원 미만이면 3천원으로 올림. 3안 생성·재조정·채팅 수정·선택 후 재조정 모두 같은 함수를 지나므로 재조정이 이런 식당의 금액을 낮출 수는 없음.
- **1인 단위 항목**(입장권·식사·항공): `unit_cost_krw`(1인 가격)를 LLM이 쓰고 코드가 `× 인원`으로 `cost_krw`를 계산. 리포트는 `1인 가격 × 인원 = 전체 가격`으로 표시(`_cost_cell`).
- **공용 항목**(숙박 객실, 택시, 렌터카): `cost_krw`에 전체 금액. 숙박은 2명당 1실(홀수는 올림) × 박수.
- **날짜**: `trip_dates` — 입력한 미래 날짜가 있으면 그것, 없거나 과거면 오늘+30일. 리포트에 입력값인지 기본값인지 표시.
- **출발지**: 기본값 서울(`trip_origin`), 기본값이면 그 사실을 표시.
- **일정 정렬**: `schedule_sort_key` = (일차, 시간대, 시작 시각). 저녁 시간대에서 06:00 전에 시작하는 항목은 자정을 넘긴 것으로 보고 같은 시간대 뒤로 보냄(야간 항공 도착 후 이동·체크인).

## 7. 리포트 렌더링 (`render_report_html`)

3개 탭을 CSS(라디오 입력)만으로 전환하는 단일 HTML 조각.
1. **일정표**: 칩(출발지, 조회 기준 날짜, 인원), 예산 타일·진행 막대(합계, 사용자 예산, 목표, 잔액), 미충족 경고, 표, 참고 가격 카드, 환율 카드.
   - 열: 날짜(`M/D`, 자정을 넘긴 항목은 다음 날짜) | 일차 | 출발시간 | 도착시간 | 시간대 | 활동 | 출발지 | 도착지 | 비용(원) | 교통수단 | 소요시간 | 비고.
   - 시간대 열은 내용 폭만큼만 차지하도록 CSS로 지정. 열 번호 기반 CSS(`nth-child`)이므로 열을 추가·삭제하면 `REPORT_STYLE`도 함께 바꿔야 함.
2. **맛집 및 명소 가이드북**: `guide.places`. 명소 카드는 왼쪽에 대표 이미지를 보여 줌(`_place_card`). 위키백과 이미지(출처 문구 포함)가 있으면 그것을, 못 찾았거나 허용되지 않는 주소면 기본 이미지(`DEFAULT_SIGHT_IMAGE`, 인라인 SVG 데이터라 외부 주소가 필요 없음)와 "대표 이미지 없음"을 표시. 맛집 카드에는 이미지가 없음. 이미지 주소는 위키미디어 https 만 허용(`safe_image_url`).
3. **준비물·유의 사항**: 맨 위에 날씨 요약 카드(`weather_summary`: 기간, 기온 범위, 비 가능성이 높은 날, 준비 제안을 코드로 계산. 여행 기간이 예보 범위 밖이면 "참고용"), 이어서 준비물, 목적지 사전 유의 정보, 여행 중 유의 사항. 가이드 생성이 실패해도 날씨 카드는 나옴.

## 8. 외부 연동

`MOCKDATA_USE`가 미설정이거나 True이면 연동 없이 LLM 추정만 사용. `False`일 때만 아래를 연결하며, 연결·키 실패는 경고만 남기고 해당 기능 없이 진행.

| 대상 | 구현 | 용도 | 키 |
|---|---|---|---|
| Google Maps MCP | `GoogleMapsMCP`(stdio, `npx`) | 장소 검색, 평점·리뷰, 해외 경로 | `GOOGLE_MAPS_API_KEY` |
| 마이리얼트립 MCP | `MyRealTripMCP`(HTTP) | 항공·입장권·투어·숙소 가격 | 불필요(`MYREALTRIP_MCP_URL`로 변경 가능) |
| 네이버 | `NaverMaps` | 국내 장소 좌표, 자동차 경로·통행료·연료비 | `NAVER_LOCAL_API_KEY(_ID)`, `X_NCP_APIGW_API_KEY(_ID)` |
| Frankfurter | `FrankfurterRates` | 환율 | 불필요 |
| OpenWeatherMap | `OpenWeatherForecast` | 5일 예보(도시 현지 날짜별로 묶음) | `OPENWEATHER_API_KEY` |
| 위키백과 | `WikipediaImages` | 명소 대표 이미지(썸네일). 검색된 문서 제목이 장소 이름과 맞을 때만 사용(`image_title_matches`) | 불필요 |

- MCP 클래스는 `_BackgroundMCP`가 별도 스레드의 이벤트 루프에서 서버를 유지하고, 도구를 동기 호출로 감싸며 호출 타임아웃은 60초.
- 키는 `config.py`가 `.env`(또는 환경변수, 환경변수 우선)에서만 읽고 로그·예외 메시지에 남기지 않음.
- 환율·날씨는 LangChain Tool(`make_condition_tools`)이며 실패 시 예외 대신 `{"error": …}`.

## 9. UI

### 9.1 웹 UI (`web/index.html`)
- 단계 상태: `new → select → chat → done`. `setStage`가 안내 문구와 입력창 잠금을 갱신.
- 입력창 잠금(`inputLocked`): `select`(안 선택 전)와 `done`(종료)에서 입력창·보내기 버튼 비활성. 안 선택은 카드의 "이 안으로 선택" 버튼으로만 하며 버튼이 `N번 안으로 할게`를 직접 전송. 선택이 실패하면 카드를 되돌리고 입력창은 잠긴 채 유지.
- 진행 표시: `/query/stream`의 `step` 줄을 읽어 노드 이름을 한국어 라벨로 보여 주고 경과 시간을 표시.
- 3안 카드: `plainAnswer`의 안별 블록(`- 안이름: …` 아래 들여쓴 줄)을 파싱해 합계, 예산 대비, 차별점, 주요 일정으로 그림.
- 리포트: 응답 `html`을 `iframe srcdoc`로 표시. **"HTML로 내보내기" 버튼**이 현재 리포트를 `<!doctype html>`을 갖춘 단독 문서로 만들어 `travel-report-YYYYMMDD-HHMM.html`로 내려받음(채팅 수정 후에는 최신 내용). 내려받기 전에 `inlineImages`가 리포트의 원격 이미지를 브라우저에서 받아 base64 data URI로 파일 안에 포함시켜, 오프라인에서도 이미지가 보임. 받지 못한 이미지는 원본 링크로 남기고 개수를 알림(위키미디어가 CORS를 허용하므로 가능).
- 정보 부족 오류(`missing_fields`)는 같은 세션으로 이어서 입력받음.

### 9.2 Streamlit (`src/ui.py`)
검증 전용. 리포트가 생기기 전에는 채팅 입력을 비활성화하고, `POST /query`만 호출.

## 10. 설정 (`src/config.py`, `.env`)

| 키 | 의미 |
|---|---|
| `AWS_DEFAULT_REGION`, `BEDROCK_MODEL_ID` | 필수. 없으면 `ConfigError` |
| `AWS_*`, `LANGFUSE_*` | 있으면 환경변수로 내보냄(boto3·langfuse는 `.env`를 읽지 않음). Langfuse 키가 있으면 추적 콜백 연결 |
| `MOCKDATA_USE` | `False/0/no/off`면 외부 연동 사용, 그 외는 LLM 추정만 |
| `GOOGLE_MAPS_API_KEY`, `NAVER_*`, `X_NCP_APIGW_*`, `OPENWEATHER_API_KEY`, `MYREALTRIP_MCP_URL` | 위 연동용 |

사용자는 자격 증명을 입력하지 않음.

## 11. 테스트와 평가

- `tests/`: `test_api.py`(API 계약, 세션 재개, 선택 인식, 수정 거절, UI 정적 검사), `test_plans.py`(그래프 흐름, MCP·환율·날씨 연결), `test_budget.py`(예산·비용 함수), `test_readjust.py`(재조정 규칙), `test_chat_edit.py`, `test_report.py`(HTML 렌더링), `test_config.py`. 가짜 모델은 프롬프트 내용으로 응답을 정함.
- 실행: `python -m pytest tests -q`.
- `evaluation/run_eval.py`: `test_queries.csv` 문항을 실제 Bedrock으로 `POST /query`에 보내 `expected_tools`·`forbidden`을 자동 채점하고 `expected_traits`는 사람이 보도록 결과 CSV에 남김. 결과는 `evaluation/`에 타임스탬프 파일로 저장.

## 12. 알려진 한계

- 서버 재시작 시 세션이 모두 사라짐(`InMemorySaver`).
- 환율 곱셈은 코드가 하지만 현지 통화 단가는 LLM이 쓰므로 그 값의 오차는 잡지 못함. Frankfurter가 지원하지 않는 통화는 LLM 추정으로 대체.
- 식사 가격 확정은 활동명이 Google 검색 결과의 장소 이름(영문 표기가 많음)을 포함할 때만 걸림. 한글로 쓴 활동명이나 이름이 다른 식당은 매칭되지 않아 LLM 추정으로 남음. 매칭된 식당은 재조정으로 금액을 낮출 수 없어, 예산이 빠듯하면 가장 근접한 안과 경고로 끝날 수 있음.
- 입장권·투어 가격은 조회 결과가 프롬프트 참고값일 뿐 코드가 확정하지 않음(활동명과 상품명이 달라 짝짓기 어려움). LLM이 옮겨 쓰는 값이라 조회값과 다를 수 있음.
- 항공 조회는 성인 1명 기준이고 숙소 요금은 조회 결과의 `원/박`을 객실 1박 요금으로 봄(인원별 객실 요금인지는 확인하지 못함). 조회 시점과 예약 시점 요금은 다를 수 있음.
- 항공 조회 결과는 마지막 항공 조회 한 번만 보관하고(숙소는 이름별로 누적), 조회가 여러 번이면 마지막 것이 후보가 됨.
- 날씨는 오늘부터 5일 예보뿐이라 먼 날짜는 참고용으로만 표시되고 일정 배치에는 반영되지 않음.
- 가이드는 선택 직후 한 번만 생성되어, 채팅으로 숙소·장소를 바꾸면 가이드와 어긋날 수 있음.
- 가능한 수정 요청 판별(`EDIT_RE`)은 단어 목록 기반이라 목록에 없는 표현은 서버에서 잡지 못함(웹 UI는 입력 자체를 잠가 보완).
- 출발지가 공항 없는 도시(예: 하남)여도 항공 구간이 그 도시 이름으로 나올 수 있음(항공 프롬프트가 출발지 이름을 그대로 사용).
- 재조정 이후에도 예산을 못 맞추면 초과액이 가장 작은 시도를 경고와 함께 보여 줌.
