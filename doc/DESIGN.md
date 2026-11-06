# 여행 일정 에이전트 웹앱 — DESIGN

> 기준 문서: `doc/SERVICE.md` (제품 결정). 설계 원칙: `doc/langgraph-design.md`. 제출 규약: `CLAUDE.md`.
> 충돌 시 우선순위는 SERVICE.md → CLAUDE.md → langgraph-design.md 입니다. 충돌한 지점은 15절에 기록했습니다.
> 구현된 코드를 요청 순서대로 따라가는 설명은 `doc/detail_design.md` 에 있습니다. 이 문서는 설계 의도와 결정 이유를 다룹니다.

## 1. 설계 개요

- 에이전트는 LangGraph `StateGraph` 하나로 구성합니다. 입력 → (사전 조사) → 3안 생성 → 예산 검사·재조정 → 3안 비교 → 사용자 선택 → 경로·가이드 → 리포트 → 대화 수정의 흐름을 노드와 엣지로 표현합니다.
- 예산 판정, 합계, 재조정 횟수, 가장 근접한 안 선택, 인원 곱셈, 일정 정렬, 가격대를 아는 식당의 1인 가격 확정은 **도구(코드)** 가 수행합니다. LLM은 항목별 원화 금액 추정(조회값이 없는 항목), 3안 차별점 문장, 가이드 문장, 대화 수정 결과 생성만 맡습니다.
- 사용자 선택과 대화 입력은 `interrupt()` 로 그래프를 멈춘 뒤, 다음 `/query` 요청에서 `Command(resume=...)` 으로 재개합니다. `session_id` 가 LangGraph `thread_id` 입니다.
- 상태는 원본 데이터만 저장합니다. 합계, 초과 금액, 가장 근접한 안, HTML 은 필요할 때 계산합니다.
- 외부 조회(환율·날씨, Google Maps, 마이리얼트립, 네이버, 위키백과)는 `MOCKDATA_USE=False` 일 때만 그래프에 들어갑니다. 조회에 실패해도 요청은 계속되고 LLM 추정으로 대체합니다.

```
START
  │
  ▼
validate_request ──(invalid)──▶ END (오류 응답)
  │
  ▼  (연결된 것만)
[research_conditions] ─▶ [research_places] ─▶ [research_prices]   환율·날씨 → 장소 → 가격
  │ Send × 3 (plan_type 별 병렬)
  ▼
generate_plan ──▶ check_budget ──(within)──────────────┐
                      │ (over, attempts < 3)            │
                      ▼                                 │
                  readjust_plan ──▶ check_budget        │
                      │ (over, attempts = 3)            │
                      ▼                                 │
                  pick_closest ─────────────────────────┤
                                                        ▼
                                                  collect_plans ──(안 없음)──▶ END
                                                        │
                                                        ▼
                                                  compare_plans
                                                        │
                                                        ▼
                                                await_selection  ◀── interrupt(stage=select)
                                                        │ resume: select
                                          ┌─────────────┴─────────────┐  병렬
                                          ▼                           ▼
                        [lookup_routes | lookup_domestic_routes]  generate_guide
                                          └─────────────┬─────────────┘
                                                        ▼
                                                  render_answer ◀────────┐
                                                        │                │
                                                        ▼                │
                                                await_chat  ◀── interrupt(stage=chat)
                                                        │ resume: chat   │
                                                        ▼                │
                                                  chat_edit              │
                                                        ▼                │
                                              check_budget_selected ─(ok)┘ (render_answer)
                                                        │ (over, attempts < 3)
                                                        ▼
                                              readjust_selected ─▶ check_budget_selected
                                                        │ resume: done
                                                        ▼
                                                       END
```
`[ ]` 는 외부 연동이 연결됐을 때만 들어가는 노드입니다.

## 2. 노드 분해

langgraph-design.md 1절의 노드 종류 기준으로 분류합니다.

| 노드 | 종류 | 책임 | 재시도 정책 |
|---|---|---|---|
| `validate_request` | 데이터 (검증) | 여행지 비어 있지 않음, 기간·인원 1 이상 정수, 예산은 생략 또는 양의 정수. 제한 없음 (SERVICE 2절) | 없음 |
| `research_conditions` | 도구 + LLM (구조화 출력 1회) | 해외 여부·통화·날씨 검색어를 정하고 환율(`get_exchange_rate`)·날씨(`get_weather_forecast`) Tool 호출 | 실패 시 `{"error": …}` 를 받고 계속 진행 |
| `research_places` | LLM (에이전트) | Google Maps MCP 로 맛집·명소 조사, 평점·리뷰 수(`place_stats`) 수집, 해외 여부 확정 | 실패 시 경고만 남기고 계속 진행 |
| `research_prices` | LLM (에이전트) | 마이리얼트립 MCP 로 입장권·투어·숙소·항공 가격 조회 | 실패 시 경고만 남기고 계속 진행 |
| `generate_plan` | LLM | 안 1개(맛집형/명소형/가성비형)의 일차·시간대별 항목과 원화 금액 생성. 3개 병렬 | 노드 안 재시도(구조화 출력 3회, 일시 오류 3회). 끝내 실패한 안은 제외 |
| `check_budget` | 도구 (코드) | 합계 계산, 목표 예산 대비 판정, 초과 금액 계산. 예산이 없으면 항상 충족 | 없음 (결정적) |
| `readjust_plan` | LLM | 초과 정보와 비용 큰 항목을 넣고 항목 금액을 줄인 안 재생성 | `RetryPolicy` (일시 오류) |
| `pick_closest` | 도구 (코드) | 시도 중 초과 금액이 가장 작은 안 선택 | 없음 |
| `collect_plans` | 데이터 | 3안 결과를 상태에 모음. 안이 하나도 없으면 종료 | 없음 |
| `compare_plans` | LLM | 안이 2개 이상일 때 안별 차별점(2~3문장, 250자 이내) 작성. 코드가 길이를 검사하고 실패하면 폴백 문장 사용 | 실패해도 요청은 계속 |
| `await_selection` | 사용자 입력 | 3안 요약(합계, 예산 대비, 차별점, 주요 일정) 제시, 사용자 선택 대기 | 해당 없음 (`interrupt`) |
| `lookup_routes` | LLM (에이전트) | 해외 + Google Maps 연결 시 선택 안의 구간별 출발지·도착지·소요시간·교통수단을 채움 | 실패 시 원래 항목 유지 |
| `lookup_domestic_routes` | LLM + 코드 | 국내 + 네이버 연결 시 장소 검색어를 모델이 정하고 코드가 자동차 경로·시간·통행료·연료비를 채움 | 실패 시 원래 항목 유지 |
| `generate_guide` | LLM | 맛집 4·명소 4, 준비물, 유의 사항, 목적지 사전 정보 생성. 명소 대표 이미지는 위키백과에서 코드가 병렬 조회 | `RetryPolicy` (일시 오류) |
| `await_chat` | 사용자 입력 | 선택한 안에 대한 대화 입력 대기 또는 종료 대기 | 해당 없음 (`interrupt`) |
| `chat_edit` | LLM (에이전트) | 대화 요청을 반영해 선택한 안의 항목 수정 | `RetryPolicy` (일시 오류) |
| `check_budget_selected` | 도구 (코드) | 수정된 선택 안의 합계와 판정 | 없음 |
| `readjust_selected` | LLM | 수정 후 초과 시 재조정 (안별 최대 3회 규칙 동일) | `RetryPolicy` |
| `render_answer` | 도구 (코드) | `plainAnswer` 텍스트와 `html` 리포트 생성 (상태에 저장하지 않음) | 없음 |

- **액션 단계 없음.** 외부 조회는 읽기만 하고 예약·결제 같은 부작용이 없으므로 캐시 금지 노드도 없습니다.
- **외부 데이터 조회 노드 (선택).** `MOCKDATA_USE=False` 일 때만 `research_*`·`lookup_*` 노드가 그래프에 들어갑니다. 키가 없거나 연결에 실패하면 그 노드만 빠지고 LLM 추정으로 진행합니다 (15절).

## 3. 상태 설계

원칙: 원본 데이터만 저장합니다. 파생 값(합계, 잔액, 초과 여부, 가장 근접한 안, HTML, 프롬프트 문자열)은 상태에 넣지 않습니다.

```python
from typing import Annotated, Literal, TypedDict
from langgraph.graph.message import add_messages

class TripRequest(TypedDict):
    destination: str          # 여행지 (국내·해외 제한 없음)
    days: int                 # 여행 기간 (제한 없음)
    people: int               # 인원 (성인 기준 한 숫자)
    budget_krw: int | None    # 사용자 예산 (원). 없으면 None = 금액 제한 없음
    origin: NotRequired[str | None]      # 출발지. 없으면 서울(기본값 표시)
    start_date: NotRequired[str | None]  # 사용자가 쓴 시작일 (YYYY-MM-DD). 없으면 오늘+30일

class ItineraryItem(TypedDict):
    day: int                  # 1부터 시작
    time_slot: Literal["아침", "오후", "저녁"]   # 고정 3개
    activity: str
    cost_krw: int             # 전체 원화 금액. per_person 항목은 코드가 unit_cost_krw × 인원으로 계산
    transport: str
    note: str                 # 비용 근거 또는 특이 사항 (40자 이내)
    # 선택 필드(NotRequired): origin, arrival, duration, depart_time, arrive_time (HH:MM),
    #   per_person, unit_cost_krw (1인 가격, 숙박은 객실 1박 요금), local_unit_price, local_currency, rooms, nights

PlanType = Literal["맛집형", "명소형", "가성비형"]

class PlanState(TypedDict):
    plan_type: PlanType
    attempts: list[list[ItineraryItem]]   # 시도별 항목 목록 (원본). 최대 1 + 재조정 3회
    # 총 재조정 횟수는 len(attempts) - 1 로 계산. 별도 카운터를 두지 않음

class GraphState(TypedDict, total=False):
    trip: TripRequest
    plans: Annotated[dict[PlanType, PlanState], _merge_plans]   # 3안 전체. 병렬 생성 결과를 안 종류 키로 병합
    comparison: dict[str, str]            # 안별 차별점 문장 (compare_plans 가 한 번 만듦)
    selected: PlanType | None             # 사용자가 고른 안
    selected_items: list[ItineraryItem]   # 선택 후 대화 수정이 반영된 항목 (원본)
    selected_attempts: list[list[ItineraryItem]]  # 수정 1회마다 새로 시작하는 재조정 시도 (원본)
    research: str | None                  # Google Maps MCP 장소 조사 결과
    place_stats: dict[str, dict]          # {장소 이름: {rating, reviews}}. 주요 일정의 식당 판정용
    price_research: str | None            # 마이리얼트립 MCP 가격 조사 결과 (프롬프트용 요약 텍스트)
    price_quotes: dict | None             # 같은 조회의 구조화 값 {flights, stays}. 항공·숙소 금액을 코드가 확정할 때 씀
    overseas: bool                        # 해외 여부 (경로 조회 노드 선택에 씀)
    fx: dict | None                       # 환율 {currency, krw_per_unit, date}
    weather: dict | None                  # 날씨 {city, in_range, days}
    guide: dict | None                    # 가이드 원본 (안 선택 직후 1회 생성)
    next_action: str | None               # await_chat 재개 값: "chat" | "done"
    chat_history: Annotated[list, add_messages]   # 대화 이력 (리듀서: 누적)
    tool_results: Annotated[list[dict], _concat]  # contexts 용 도구 호출 결과 (누적)
    trace: Annotated[list[str], _concat]          # 거친 노드 이름 (누적)
    errors: Annotated[list[str], _concat]         # LLM이 고칠 수 있는 오류 (2절 6장)
```

- `chat_history` 는 LangChain 메시지 객체를 담으므로 `add_messages` 리듀서를 씁니다.
- `plans` 는 3안이 병렬로 각자 한 안씩 돌려주므로 `_merge_plans` 로 병합합니다.
- 조사 결과(`research`, `price_research`, `fx`, `weather`)는 조회한 원본입니다. 조회에 실패하면 비어 있고 LLM 추정으로 진행합니다.
- `tool_results` 와 `trace` 는 누적 리듀서입니다. 응답의 `contexts` 와 `trace` 가 여기서 나옵니다.
- 리듀서는 `Annotated` 로만 선언합니다. 누적 로직을 노드 안에서 직접 다루지 않습니다.
- 프롬프트 문자열은 상태에 없습니다. 노드 함수 안에서 `trip` 과 `items` 를 보고 만듭니다.

## 4. 엣지와 제어 흐름

| 위치 | 방식 | 이유 |
|---|---|---|
| `validate_request` → 조사 노드 → 3개 `generate_plan` | 조건부 엣지 + `Send` × 3 (`plan_type` 별 입력) | 연결된 조사 노드(환율·날씨 → 장소 → 가격)만 거친 뒤 `fan_out` 이 안 3개를 병렬 시작. 안 3개는 서로 독립이고, 각자 재조정 카운터를 가짐 |
| `check_budget` → `readjust_plan` / `pick_closest` / `collect_plans` | `add_conditional_edges` | 상태(시도 횟수, 충족 여부)만 보고 분기. 초과한 안만 `Send` 로 재조정 |
| `collect_plans` → `compare_plans` / `END` | `add_conditional_edges` | 안이 하나도 없으면 오류로 종료 |
| `await_selection` → 경로 조회 노드 + `generate_guide` | 조건부 엣지 (동시 시작) → `render_answer` 합류 | 경로 조회와 가이드는 서로 다른 상태 키를 써서 충돌하지 않음. 연결된 것만 시작 |
| `await_selection`, `await_chat` | `interrupt()` + 재개 값 분기 | HITL (langgraph-design 7절). payload 의 `stage` 로 단계를 구분 |
| `chat_edit` → `check_budget_selected` | 일반 엣지 | 항상 검사 |
| 재개 값 `done` | 조건부 엣지 → `END` | 사용자가 종료를 선택한 경우 |

- `Command` 는 사용하지 않습니다. 상태 업데이트와 다음 노드를 동시에 결정할 필요가 없고, 조건부 엣지로 충분합니다.
- `Command[Literal[...]]` 를 쓰는 노드가 없으므로 목적지 타입 힌트 규칙은 조건부 엣지의 라우터 반환 타입으로 대신합니다.
- 병렬 분기(`Send`) 안에서는 `interrupt` 를 호출하지 않습니다. 사용자 개입은 3안이 모인 뒤 `await_selection` 에서만 합니다.

## 5. 예산 재조정 루프

SERVICE 4절 규칙을 그대로 옮깁니다.

| 항목 | 규칙 | 구현 위치 |
|---|---|---|
| 목표 예산 | 맛집형·명소형 = `trip.budget_krw`, 가성비형 = `trip.budget_krw × 0.8` (`VALUE_PLAN_RATIO`). 예산이 없으면 목표 없음 | `tools.py` (`target_budget`) |
| 판정 | `total ≤ target` 이면 충족. 예산이 없으면 항상 충족 | `tools.py` (`check_budget`) |
| 최대 재조정 | 안별 3회. 첫 시도는 재조정 횟수에 포함하지 않음 | `check_budget` 의 조건 (`len(attempts) - 1 < 3`) |
| 가장 근접한 안 | 시도 중 `total − target` 이 가장 작은 시도 (음수가 아닌 초과 금액 기준) | `tools.py` (`closest_attempt`), 노드 `pick_closest` |
| 경고 | 충족 못 한 안을 보여줄 때 초과 경고 표시 | `render_answer` 에서 계산 |
| 대화 수정 후 | 선택한 안이 초과하면 같은 루프를 다시 실행. 시도 횟수는 **수정 1회마다 새로 셈** | `check_budget_selected`, `readjust_selected` |

- 대화 수정 후 재조정 횟수를 새로 세는 이유: 사용자가 요청한 변경은 새 요청으로 보고, 이전 안의 재조정 기록과 섞지 않습니다. (15절 확정: 수정 1회마다 새로 셈)
- `closest_attempt` 는 `PlanState.attempts` 에서 계산합니다. 별도 상태 필드를 두지 않습니다.
- 재조정 프롬프트에는 "목표보다 얼마 초과했는지" 를 넣습니다. 이 값은 노드 안에서 `check_budget` 결과로 계산해 프롬프트 문자열에만 씁니다.

## 6. HITL과 세션 매핑

| `/query` 요청 | 그래프 동작 |
|---|---|
| `session_id` 없음, `question` 에 여행 요청 | 새 `thread_id` 생성. 1~3절 흐름 실행 → `await_selection` 에서 interrupt |
| `session_id` 있음, `question` 이 안 선택 | `Command(resume={"type": "select", "plan": ...})` → 경로 조회·`generate_guide` → `render_answer` → `await_chat` 에서 interrupt |
| `session_id` 있음, `question` 이 대화 수정 | `Command(resume={"type": "chat", "text": ...})` → `chat_edit` → `render_answer` → `await_chat` 에서 다시 interrupt |
| `session_id` 있음, `question` 이 종료 | `Command(resume={"type": "done"})` → `END` |

- `interrupt` 의 payload 는 각 단계에서 사용자에게 보여줄 내용(`stage`, `plainAnswer`)입니다. `await_selection` 은 3안 요약, `await_chat` 은 현재 선택 안 요약을 넣습니다. HTML 은 넣지 않습니다.
- 재개 값은 사용자가 입력한 자연어를 그대로 쓰지 않습니다. `/query` 처리 계층(`api.py`)이 `question` 을 보고 `type` 을 정합니다. 이 분류는 LLM에 맡기지 않고 **API 계층에서 요청 상태(선택 대기인지, 대화 대기인지)** 로 결정합니다. 대기 단계는 체크포인트의 interrupt `stage` 에서 읽으며(`_pending_stage`) 별도 상태로 저장하지 않습니다.
- 선택 단계의 해석 규칙:
  - 수정 표현(`변경`, `바꿔`, `수정` 등, `EDIT_RE`)이 있으면 422 `edit_before_selection` 으로 거절합니다. 안을 고르기 전에는 수정 요청을 받지 않습니다.
  - `1번`·`3안`처럼 `번`·`안` 이 붙었거나 숫자만 있거나 안 이름(`가성비` 등)이 있을 때만 선택으로 읽습니다 (`parse_selection`). `숙소를 2박으로` 의 2 는 선택이 아닙니다. 못 읽으면 422 `invalid_selection`.
- 대화 단계에서는 `종료` 등(`DONE_WORDS`)이면 종료하고, 아니면 수정 요청으로 재개합니다.
- 모르는 `session_id` 는 새 세션으로 시작하지 않고 404 `session_not_found`, 대기 중이 아니거나 종료된 세션은 404 `session_closed` 를 돌려줍니다. 새 세션은 첫 요청에서만 만듭니다.

## 7. 영속성과 메모리

- **Checkpointer:** `InMemorySaver`. `thread_id = session_id`. 모든 `invoke` 호출에 `config={"configurable": {"thread_id": ...}}` 를 넣습니다.
- **이유:** `interrupt` 와 재개에 checkpointer 가 필수이고, SERVICE 6.1절이 세션 상태를 서버 메모리에 두도록 정했습니다.
- **운영 주의:** langgraph-design 4절은 `InMemorySaver` 를 테스트용으로 구분합니다. v1 은 DB 를 쓰지 않는다는 SERVICE 결정 때문에 메모리 체크포인터를 씁니다. 서버 재시작 시 세션은 사라집니다. 이는 SERVICE 2절(새로고침 후 복원 제외)과 같은 수준의 제약입니다. (15절 확정: 서버 메모리)
- **Durability:** `sync`. 노드 사이마다 저장해 중간 장애 뒤 재개할 수 있게 합니다. 단, 메모리 저장소라 프로세스가 죽으면 의미가 없습니다. 이 설정은 기본 권장값이며 v1 에서는 큰 비용이 없어 유지합니다.
- **Store (장기 기억):** 사용하지 않습니다. 세션을 넘는 정보가 없습니다.

## 8. 오류 처리

langgraph-design 6절 표를 따릅니다.

| 오류 | 예 | 처리 |
|---|---|---|
| 일시 오류 | Bedrock 네트워크 오류, 스로틀링, 시간 초과 | LLM 노드에 `RetryPolicy(max_attempts=3, initial_interval=2.0, backoff_factor=3.0)` (`LLM_RETRY`). 읽기 timeout 120초, 연결 timeout 10초. 값·타입 오류 같은 코드 문제는 재시도하지 않음(`is_retryable`). 예산 재조정 횟수와 **별개** |
| LLM이 고칠 수 있는 오류 | 응답 JSON 파싱 실패, 항목 필드 누락 | 구조화 출력 파싱 실패는 같은 프롬프트로 최대 3회 다시 시도(`STRUCTURED_ATTEMPTS`). `RetryPolicy` 와 별도. 끝내 실패하면 `errors` 에 기록 |
| 외부 조회 오류 | 키 없음, MCP 연결 실패, 환율·날씨 조회 실패 | 예외로 끝내지 않고 경고만 남겨 해당 정보 없이 LLM 추정으로 진행. Tool 은 예외 대신 `{"error": …}` 를 돌려줌. MCP 호출 timeout 60초 |
| 사람이 고칠 수 있는 오류 | 여행지 누락, 예산 0 이하 | API 앞단 Guardrail(16절)이 HTTP 422 로 무엇이 빠졌는지 알리고 같은 `session_id` 로 이어감. 그래프의 `interrupt()` 로 입력을 요청하지 않음 |
| 재시도 소진 후 복구 | 3회 재시도 실패 | 해당 안을 "생성 실패" 상태로 `attempts` 에 남기지 않고, 나머지 안만 보여줌. 3안이 모두 실패하면 오류 응답 |
| 예상 못 한 오류 | 코드 버그 | 잡지 않고 전파 → 500 |

- `interrupt()` 는 오류 처리 경로를 타지 않습니다. 재시도 정책도 적용되지 않습니다. 사용자 입력 대기 노드에서는 예외 처리를 두지 않고, `api.py` 의 재개 검증에서 잘못된 `type` 을 막습니다.

## 9. 도구와 에이전트

### `src/tools.py` (결정적 로직, LLM 호출 없음)
| 함수 | 입력 | 출력 |
|---|---|---|
| `target_budget(plan_type, budget_krw)` | 안 종류, 예산 | 목표 예산 (원) |
| `total_cost(items)` | 항목 목록 | 합계 (원) |
| `check_budget(plan_type, items, budget_krw)` | 안, 항목, 예산 | `{within: bool, total, overage}` |
| `closest_attempt(attempts, plan_type, budget_krw)` | 시도 목록 | 초과 금액이 가장 작은 시도 |
| `apply_headcount(items, people)` | 항목, 인원 | `per_person` 항목의 `cost_krw` 를 `unit_cost_krw × 인원` 으로 다시 계산 |
| `finalize_items(items, people, days, fx, place_stats, price_quotes, plan_type)` | 항목, 인원, 환율, 장소 통계, 항공·숙소 조회값 | 항공·숙소 금액을 조회값으로 덮어쓰기(`apply_price_quotes`), 환율 환산, 숙박 총액, 식당 1인 가격(Google 가격대 `MEAL_PRICE_BY_LEVEL` 또는 최소 단가), 인원 곱셈을 코드가 확정 |
| `parse_price_quotes`, `record_price_quotes` | 마이리얼트립 조회 원본, 호출 인자 | 항공(`flights`: 항공사, 직항 여부, 1인 왕복 요금, 가는·오는 편 시각·편명)과 숙소(`stays`: 이름, 1박 요금)의 구조화 값. 요청별 sink 에 모음 |
| `budget_usage_line`, `main_schedule` | 안의 항목, 예산, `place_stats` | 카드의 예산 대비 줄, 주요 일정 5개 |
| `fit_difference`, `fallback_difference` | 차별점 문장 | 문장 수·길이를 검사한 문장, 수치 기반 폴백 문장 |
| `trip_dates`, `trip_origin`, `schedule_sort_key` | 요청, 항목 | 조회 기준 날짜(입력 또는 D+30), 출발지(기본 서울), 일정 정렬 키 |
| `classify_request`, `missing_fields`, `merge_trip`, `has_conflict` | 요청 문장, 이전 값 | 가드레일 판정, 빠진 항목, 병합값, 조건 충돌 여부 (16절) |
| `render_plain_answer(trip, plans, selected, items, …)` | 상태의 일부 | `plainAnswer` 텍스트 (3안 카드 정보 포함) |
| `render_report_html(plan_type, items, budget_krw, guide, …)` | 선택 안, 가이드, 조회 결과 | HTML 문자열 (AC4 항목, 3탭) |
| `FrankfurterRates`, `OpenWeatherForecast`, `make_condition_tools` | 통화 코드, 도시 | 환율·날씨 LangChain Tool (`get_exchange_rate`, `get_weather_forecast`) |
| `GoogleMapsMCP`, `MyRealTripMCP`, `NaverMaps`, `WikipediaImages` | 검색어, 구간 | 장소·가격·경로·이미지 조회 클라이언트 (외부 연동 클래스) |

- `render_report_html` 은 LLM 이 만든 문자열(활동명, 비고)을 HTML 에 넣기 전에 이스케이프합니다. 이스케이프하지 않으면 리포트가 HTML 주입 경로가 됩니다. 대표 이미지 주소는 위키미디어 https 만 허용합니다(`safe_image_url`).
- 외부 연동 클래스는 `tools.py` 에 두되 LLM 을 호출하지 않습니다. 키는 `config.py` 에서만 받습니다.

### `src/agent.py` (그래프와 LLM 호출)
- 모델: `ChatBedrockConverse(model=BEDROCK_MODEL_ID, region_name=AWS_DEFAULT_REGION)`. 두 값은 `.env` 에서 읽습니다. 출력 토큰 상한은 16,384 입니다.
- `generate_plan` / `readjust_plan`: `with_structured_output` 으로 `list[ItineraryItem]` 형태(`PlanOutput`)를 받습니다. 구조화 출력이 실패하면 2절 6장 규칙을 따릅니다. 성공하면 `apply_headcount` 로 1인 단위 항목의 전체 비용을 코드가 다시 계산합니다.
- 프롬프트는 안 스타일, 여행 정보, 예산 문장(안별 목표 총액, 1인당·1일당 목표, 예산이 없으면 상한 없음), 규칙 모음(1인 단위/공용 비용, 시각 HH:MM, 비고 40자, 항공 두 항목, 숙박 규칙, 환율 환산·날씨 반영)으로 구성합니다.
- `compare_plans`: `PlanComparison` 구조화 출력으로 안별 차별점을 받고, 문장 수·길이를 `fit_difference` 가 검사합니다.
- `generate_guide`: `GuideOutput` 구조화 출력으로 맛집·명소·준비물·유의 사항을 한 번에 받습니다. 명소 대표 이미지는 모델이 아니라 `attach_place_images` 가 위키백과에서 찾습니다.
- `research_places` / `research_prices` / `lookup_routes`: `create_agent` 로 만든 에이전트가 MCP 도구(검색, 가격, 경로)를 호출합니다. 판단 결과가 아니라 조회한 값만 상태에 남깁니다.
- `chat_edit`: `create_agent` 로 만든 에이전트를 씁니다. 도구는 두 개만 줍니다.
  - `get_plan_items()`: 현재 선택 안의 항목을 돌려줌
  - `submit_plan_items(items)`: 수정된 항목을 제출. 제출 결과는 `selected_items` 에 저장되고, 합계는 `check_budget_selected` 가 계산
- 에이전트가 예산을 판단하거나 합계를 말하도록 하지 않습니다. 판단은 그래프의 `check_budget_selected` 가 합니다.
- 노드 함수에 쓰는 프롬프트 문자열은 `agent.py` 안에 둡니다. 별도 프롬프트 파일은 만들지 않습니다 (CLAUDE.md "파일 추가 금지").

## 10. API 설계 (`src/api.py`)

요청
```json
{ "question": "도쿄 3박 4일, 100만원으로 여행 계획 짜줘", "session_id": null }
```

응답
```json
{
  "answer": { "session_id": "sess_...", "plainAnswer": "...", "html": "" },
  "contexts": [ { "tool": "check_budget", "result": { "within": true, "total": 950000 } } ],
  "trace": [ "validate_request", "generate_plan", "check_budget", "collect_plans", "await_selection" ]
}
```

- `session_id` 는 `answer` 객체 안에 항상 포함합니다. 첫 요청에서 서버가 만들어 돌려줍니다 (15절 확정).
- 응답 키는 규약의 `answer`, `contexts`, `trace` 세 개입니다.
- 오류 응답은 `{"error": {"code": "...", "message": "..."}}` 형식이고, HTTP 상태는 422(입력 오류), 404(세션 없음), 500(그 밖)입니다 (15절 확정).
- `question` 이 비어 있으면 422 입니다.
- 여행 요청이 아닌 입력과 정보 부족은 Guardrail(16절)이 그래프 호출 전에 422 로 처리합니다. 오류 코드는 16절 표를 따릅니다.
- 그래프가 3안을 하나도 만들지 못하면 검증 오류는 422 `invalid_trip`, 그 밖은 500 `generation_failed` 입니다.

### 엔드포인트
| 경로 | 설명 |
|---|---|
| `POST /query` | 처리가 끝난 뒤 `answer`, `contexts`, `trace` 를 한 번에 반환 |
| `POST /query/stream` | 같은 처리를 NDJSON 으로 반환. 노드가 끝날 때마다 `{"step": 노드}`, 마지막에 `{"result": …}` 또는 `{"error": …, "status": …}`. 워커 스레드가 큐에 넣고 응답 제너레이터가 읽음 |
| `GET /` | `web/index.html` (웹 UI) |

- 새 요청의 문장 해석(`parse_trip`)은 규칙(정규식)으로 합니다. 여행지·일수·인원(`parse_people`)·예산·출발지·도착지(`parse_route`)·날짜(`parse_dates`)를 뽑고, LLM 을 쓰지 않습니다.

## 11. 리포트 구성 (AC4)

`render_report_html` 이 만드는 HTML 은 아래 항목을 모두 포함해야 합니다 (SERVICE 5절).

- 일차별·시간대별 일정 (표 또는 목록)
- 항목별 활동, 비용(원), 교통수단, 비고
- 항목별 비용 합계
- 예산 대비 잔액 = 사용자 예산 − 합계 (초과 시 음수로 표시)
- 초과 경고 (합계가 목표 예산을 넘을 때). 가장 근접한 안을 보여줄 때도 표시

- 잔액은 **사용자 예산** 기준입니다. 목표 예산(가성비형 80%)과 헷갈리지 않도록 리포트에 두 값을 따로 씁니다.
- 가장 근접한 안이면 "예산 미충족" 문구를 함께 씁니다.
- 예산이 없으면 잔액·진행 막대·경고를 만들지 않습니다.

### 3탭 구성
리포트는 CSS(라디오 입력)만으로 전환하는 3개 탭의 단일 HTML 조각입니다.
1. **일정표**: 칩(출발지, 조회 기준 날짜, 인원), 예산 타일·진행 막대, 미충족 경고, 표, 참고 가격 카드, 환율 카드. 표 열은 날짜 | 일차 | 출발시간 | 도착시간 | 시간대 | 활동 | 출발지 | 도착지 | 비용(원) | 교통수단 | 소요시간 | 비고 입니다. 1인 단위 비용은 `1인 가격 x 인원 = 전체 가격` 으로 씁니다. 열 번호 기반 CSS(`nth-child`)이므로 열을 바꾸면 `REPORT_STYLE` 도 함께 바꿉니다.
2. **맛집 및 명소 가이드북**: `guide.places`. 명소 카드는 왼쪽에 대표 이미지를 보여주고, 못 찾으면 기본 이미지(`DEFAULT_SIGHT_IMAGE`, 인라인 SVG)와 "대표 이미지 없음"을 표시합니다. 맛집 카드에는 이미지가 없습니다.
3. **준비물·유의 사항**: 맨 위에 날씨 요약 카드(`weather_summary`, 예보 범위 밖이면 "참고용"), 이어서 준비물, 목적지 사전 유의 정보, 여행 중 유의 사항. 가이드 생성이 실패해도 날씨 카드는 나옵니다.
- 일정 정렬은 `schedule_sort_key` = (일차, 시간대, 시작 시각) 입니다. 저녁 시간대에서 06:00 전에 시작하는 항목은 자정을 넘긴 것으로 보고 같은 시간대 뒤로 보냅니다.

## 12. 설정

| 변수 | 값 | 읽는 곳 |
|---|---|---|
| `AWS_DEFAULT_REGION` | `us-east-1` | `agent.py` (모델 생성) |
| `BEDROCK_MODEL_ID` | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | `agent.py` |
| `MOCKDATA_USE` | `False/0/no/off` 면 외부 연동 사용, 그 외(미설정 포함)는 LLM 추정만 | `agent.py` (그래프 조립), `api.py` |
| `GOOGLE_MAPS_API_KEY` | Google Maps MCP 용 (없으면 `ConfigError`, 해당 연동만 빠짐) | `agent.py` |
| `MYREALTRIP_MCP_URL` | 마이리얼트립 MCP 주소 (키 불필요, 없으면 공개 기본 주소) | `agent.py` |
| `NAVER_LOCAL_API_KEY(_ID)`, `X_NCP_APIGW_API_KEY(_ID)` | 네이버 지역검색·Directions 15 용 | `agent.py` |
| `OPENWEATHER_API_KEY` | OpenWeatherMap 5일 예보 용 | `agent.py` |
| `AWS_*`, `LANGFUSE_*` | 있으면 환경변수로 내보냄 (boto3·langfuse 는 `.env` 를 읽지 않음). Langfuse 키가 있으면 추적 연결 | `config.py` |

- `.env` 는 `src/config.py` 에서 읽습니다 (15절 확정). 환경변수가 `.env` 보다 우선합니다. 다른 파일은 설정 값을 `config.py` 를 통해서만 받고, 코드에 값을 쓰지 않습니다 (CLAUDE.md).
- AWS 자격 증명은 사용자 입력이 없습니다. 표준 AWS 자격 증명 체인을 씁니다. 외부 연동 키도 로그와 예외 메시지에 남기지 않습니다.

## 13. 테스트와 수용 기준 매핑

| AC | 검증 파일 | 모킹 대상 | 확인 내용 |
|---|---|---|---|
| AC1 | `tests/test_budget.py`, `tests/test_plans.py` | `ChatBedrockConverse` | 3안 생성, 합계와 판정이 `check_budget` 결과와 일치 |
| AC2 | `tests/test_readjust.py` | 같음 | 안별 재조정 최대 3회, 가장 근접한 안 선택, 초과 경고 |
| AC3 | `tests/test_chat_edit.py` | 같음 | 대화 수정 반영, 초과 시 재조정 실행 |
| AC4 | `tests/test_report.py` | 없음 (`render_report_html` 직접 호출) | 리포트에 AC4 항목이 모두 있고, LLM 문자열이 이스케이프됨 |
| AC5 | `tests/test_config.py` | 환경변수 | 리전·모델 ID 를 환경변수에서 읽음, 사용자 키 입력 없음 |
| AC6 | `evaluation/test_queries.csv` + 수동 시연 | 없음 (실제 Bedrock) | 대표 입력으로 흐름 확인 |
| (보강) | `tests/test_api.py` | 가짜 그래프·모델 | API 계약, 가드레일(16절), 세션 재개, 선택 인식, 선택 전 수정 거절, 웹 UI 정적 검사 |
| (보강) | `tests/test_plans.py`, `tests/test_budget.py` | 가짜 모델, 가짜 MCP·환율·날씨 | 외부 조회 연결과 실패 대체, 인원 곱셈, 예산 없음, 3안 카드 정보, 일정 정렬 |

- 자동 테스트는 네트워크를 쓰지 않습니다. 현재 157개가 통과합니다 (`python -m pytest tests -q`).
- 가짜 모델은 프롬프트 내용을 보고 응답을 정합니다.
- AC1~AC6 전부를 `evaluation/test_queries.csv` 에 옮깁니다 (SERVICE 9절 8항). 12문항(positive 5, negative 2, edge 3, guardrail 2)이며 `evaluation/run_eval.py` 가 실제 Bedrock 으로 `POST /query` 를 호출해 `expected_tools`·`forbidden` 을 자동 채점하고, `expected_traits` 는 결과 CSV 에 남겨 사람이 봅니다.

## 14. langgraph-design 체크리스트 점검

| 항목 (8절) | 결과 |
|---|---|
| 노드 하나는 한 가지 일만 하는가 | 예. 판정·재조정·렌더링을 분리 |
| 상태에 원본 데이터만 있고 프롬프트 문자열이 없는가 | 예. 합계·HTML·프롬프트는 상태에 없음 |
| 분기는 조건부 엣지나 `Command` 로 표현되고 목적지가 명시되어 있는가 | 조건부 엣지 사용. 라우터 반환 타입에 노드 이름 명시 |
| 외부 호출 노드에 재시도 정책이 있는가, 부작용 노드는 캐시하지 않았는가 | LLM 노드에 `RetryPolicy`. 부작용 노드 없음 |
| 사람 개입 지점은 `interrupt()` 인가, checkpointer 와 `thread_id` 가 있는가 | 예. `await_selection`, `await_chat`. `session_id` 를 `thread_id` 로 사용 |
| 장애 복구가 필요하면 durability 를 `sync` 로 정했는가 | 예 |
| 멀티 에이전트가 정말 필요한가 | 아니오. 단일 그래프 + 대화 수정용 에이전트 1개 |
| 도구를 쓴다면 description 기준을 충족하는가 | 도구 설명을 작성하되, 결정적 도구는 LLM 에 노출하지 않음 (`chat_edit` 용 도구 2개만 노출) |

## 15. 결정 기록

| # | 항목 | 결정 |
|---|---|---|
| 1 | 세션 저장 위치 | 서버 메모리 (`InMemorySaver`). 서버 재시작 시 세션 소실 |
| 2 | 대화 수정 후 재조정 횟수 | 수정 1회마다 새로 3회 |
| 3 | `.env` 읽기 위치 | `src/config.py` 새 파일 (사용자 허가) |
| 4 | `session_id` 응답 위치 | `answer` 객체 안 |
| 5 | `answer` 형식 | JSON 객체: `session_id`, `plainAnswer`, `html` |
| 6 | 오류 응답 | `{"error": {"code", "message"}}`, HTTP 422 / 404 / 500 |
| 7 | 구조화 출력 스키마 | 영문 키. `time_slot` 은 `아침` / `오후` / `저녁` 고정 |
| 8 | 일부 안만 실패 | 나머지 안만 보여줌 (빠진 안은 `plainAnswer` 앞에 알림) |
| 9 | 입력 항목 | 여행지·기간·인원 필수, 예산·출발지·날짜 선택 (16절) |
| 10 | 외부 조회 | `MOCKDATA_USE=False` 일 때만 연결. 실패하면 LLM 추정으로 대체. 합계·판정은 항상 코드 |
| 11 | 환율 | 조회 도구가 환율을 주면 LLM 이 현지 통화 단가(`local_unit_price`)를 쓰고 코드가 환율을 곱해 원화로 바꿈(`finalize_items`, 통화 코드가 같을 때만). LLM 이 쓴 현지 단가 자체의 오차는 코드가 잡지 못함 |
| 12 | 선택 후 처리 | 경로 조회와 가이드 생성을 병렬로 실행해 `render_answer` 에서 합류. 가이드는 1회만 생성 |
| 13 | 선택 전 수정 요청 | 422 `edit_before_selection`. 웹 UI 는 선택 전에 입력창을 잠금 |
| 14 | 웹 UI | `web/index.html` 추가, `GET /` 로 제공. 진행 표시는 `POST /query/stream` |
| 15 | 식사 가격 | Google 가격대(`price_level`)를 아는 식당은 LLM 추정 대신 고정 기준표(저렴 1만, 보통 2만, 비쌈 5만, 매우 비쌈 10만원)로 코드가 확정 |
| 16 | 항공·숙소 가격 | 마이리얼트립 조회값(`price_quotes`)이 있으면 LLM 이 쓴 금액을 코드가 덮어씀. 조회 도구 호출 결과를 `research_prices` 가 요청별 sink 에 구조화해 모으고(세션 간 섞임 방지), 3안 생성·재조정·대화 수정·선택 후 재조정이 모두 `finalize_items` 에서 같은 덮어쓰기를 거침. 항공은 가성비형 최저가, 그 밖은 직항 중 최저가(항목에 적힌 항공사 우선)이고 가는 편·귀국 시각·편명·날짜까지 같은 항공권으로 맞춤. 숙소는 이름이 맞을 때만(사용자가 고른 다른 숙소는 존중). 입장권·투어는 이름 매칭이 어려워 LLM 옮겨 쓰기 유지 |

- 새 파일: `src/config.py`, `web/index.html`, `evaluation/run_eval.py`. SERVICE 8절과 CLAUDE.md 의 파일 목록에 반영했습니다.
- 남은 확인 사항: 없음.

## 16. 요청 분류(Guardrail)와 안내·거절 응답

여행 요청이 아닌 입력과 정보가 부족한 입력을 그래프 호출 **전에** 규칙으로 가려냅니다. LLM을 쓰지 않습니다.

### 위치와 구성
- 규칙 함수 `classify_request(question, known_trip)` 는 `src/tools.py` 에 둡니다 (결정적 로직, LLM 호출 없음). 새 파일(`src/guardrail.py` 등)은 사용자 허가 없이 만들지 않습니다.
- `src/api.py` 가 `/query` 처리 맨 앞에서 호출합니다. 그래프 노드는 늘리지 않습니다.
- 테스트는 `tests/test_api.py` 에 추가합니다 (새 테스트 파일 없음).

### 검사 순서와 결과

| 순서 | 검사 | 예 | 결과 | HTTP | 오류 코드 |
|---|---|---|---|---|---|
| 1 | 비밀 값·시스템 정보 요청 | 시스템 프롬프트, API 키, AWS, `BEDROCK_MODEL_ID`, 환경변수 | 거절 | 422 | `secret_request` |
| 2 | 개인정보·타인 정보 요청 | 다른 사용자, 저장해 둔 계획, 이름·연락처 | 거절. 로그인·저장 기능이 없다고 안내 | 422 | `personal_data_request` |
| 3 | 외부·실시간 조회 요청 | 실시간 최저가, 특정 호텔의 정확한 요금, 예약 | 거절. v1은 조회하지 않고 AI 추정값만 쓴다고 안내 | 422 | `unsupported_lookup` |
| 4 | 필수 정보 부족 (첫 요청) | 여행지·기간·인원 중 빠진 것 (예산은 선택) | 무엇이 빠졌는지 되묻고 세션 유지 | 422 | `missing_fields` |
| 5 | 조건 충돌 | 최고급·5성급인데 비용은 최소 | 충돌을 알리고 예산 우선으로 3안 진행 | 200 | 없음 (`plainAnswer` 앞에 안내) |

- 1~2번은 **모든 요청**(대화 수정 입력 포함)에 적용합니다. 3~5번은 새 여행 요청(첫 요청이거나 `missing_fields` 로 이어지는 요청)에만 적용합니다.
- 오류 메시지는 `error.message` 에 한국어로 담고, 사용자가 무엇을 하면 되는지(예: 여행지를 알려 달라)를 포함합니다.

### 정보 부족 시 세션 이어가기
1. 첫 요청에서 여행지·기간·인원(필수) 중 빠진 것이 있으면 서버가 `session_id` 를 발급하고, 파싱된 값(`known_trip`)을 서버 메모리에 보관합니다.
2. 422 응답의 `error` 에 `session_id` 와 `missing` (빠진 항목 목록)을 넣습니다.
3. 클라이언트가 같은 `session_id` 로 다시 보내면 `known_trip` 과 새 입력을 합쳐 검사하고, 모두 갖춰지면 그래프를 시작합니다.
4. 이 보관소는 그래프의 체크포인터가 아니라 `api.py` 의 서버 메모리입니다 (그래프 시작 전 상태이므로). 서버 재시작 시 사라집니다.

### 필수·선택 입력과 비용 계산
- 필수 입력은 여행지·기간·인원(성인 기준 한 숫자)이다. 예산은 선택이며, 없으면 금액 제한 없이 3안을 만들고 재조정·초과 경고를 하지 않는다(`check_budget` 이 항상 충족).
- 기간이 1일(당일치기)이면 숙박을 만들지 않는다. 그 외는 2명당 1실(홀수 올림) 기준 총 숙박비를 1일차 저녁 항목에 공용 비용으로 넣는다.
- 입장권·식당 식사·항공권은 `per_person=true` 와 `unit_cost_krw`(1인 가격)로 쓰고, 전체 비용 `cost_krw` 는 코드(`apply_headcount`)가 `1인 가격 x 인원` 으로 계산한다. 리포트는 `1인 가격 x 인원 = 전체 가격` 으로 표시한다.
- 출발지는 선택이며 없으면 서울(기본값 표시)이다. 항공 이동이 필요한 여행지는 1일차에 `출발지→목적지` 항공 항목(1인 왕복 요금 x 인원, 예산 판정 합계에 포함)과 마지막 날에 `목적지→출발지` 귀국 항목(요금은 왕복에 포함, 날짜·시각 표시)을 넣는다. 항공 조회는 둘 다 국내면 국내선, 출발지가 한국·목적지가 해외면 국제선 도구를 쓴다.
- 사용자가 날짜(`12월 20일부터 23일까지`, `2026-12-20~2026-12-22` 등)를 쓰면 그 날짜로 항공·숙소를 조회하고(범위가 있으면 기간 = 종료일 − 시작일 + 1), 없으면 오늘+30일(D+30)을 쓴다. 지난 날짜는 무시한다. 조회 기준 날짜는 `trip_dates` 한 곳에서 정하고 화면에 `(입력)`/`(입력이 없어 D+30 기본값)`으로 표시한다. 입력칸 하단에 이 기본값을 안내한다.
- 일정 항목에는 `depart_time`·`arrive_time`(HH:MM)이 있고 리포트 표의 `일차` 옆에 `출발시간`·`도착시간` 열로 보여준다(형식이 틀린 값은 비워 둔다).

### 3안 카드 정보 (예산 대비·차별점·주요 일정)
- 3안이 모이면(`collect_plans`) `compare_plans` 노드가 안이 2개 이상일 때 세 안을 비교해 안별 **차별점**(2~3문장, 250자 이내)을 만든다. 문장 수·길이는 코드가 검사해 벗어나면 현재 값을 알려 1회만 다시 요청하고, 그래도 길면 앞 3문장만 남기고 1문장이면 수치 기반 폴백 문장을 이어 붙인다. 호출이 실패해도 요청은 계속되고 폴백 문장이 쓰인다. 결과는 상태 `comparison` 에 둔다.
- 카드의 **예산 대비**(`budget_usage_line`)와 **주요 일정 5개**(`main_schedule`)는 코드가 만든다(LLM 없음). 명소 위주로 일차마다 비용이 큰 항목을 골라 시간순으로 보여주고, 이동·항공·체크인은 명소가 부족할 때만 채운다. 식당은 Google Maps 검색 결과에서 코드가 모은 평점·리뷰 수(`place_stats`)가 평점 4.5 이상·리뷰 5,000개 이상일 때만, 최대 1개 넣고 `★평점 리뷰 N개`를 붙인다(통계를 모르면 식당은 넣지 않는다). 식당 여부는 항목의 `category`(식당/명소/이동/숙박/기타)로 판단하고, 없으면 식사 관련 단어로 판단한다.
- 응답 계약은 그대로이고 정보는 `plainAnswer` 에 담는다. `- 안이름: …` 줄 아래 들여쓴 줄(`  예산 대비:`, `  차별점:`, `  주요 일정:`, `  · 항목`)이 붙고, 화면이 이를 읽어 카드로 그린다.

### 조건 충돌
- 충돌 판정은 규칙(최고급·5성급·고급 등 + 최소·최저·저렴 등의 동시 등장)으로 합니다.
- 충돌이 있어도 거절하지 않습니다. `plainAnswer` 맨 앞에 "조건이 서로 충돌해 예산을 우선해 안을 만들었다" 취지의 안내를 붙이고 3안 흐름을 그대로 진행합니다.

### 파서 보정 (TQ-07)
- 여행지 추출이 '정확한', '실시간' 같은 수식어를 장소로 읽지 않도록 불용어를 제외합니다. 3번 검사가 먼저 걸리므로 TQ-07 은 거절되지만, 파서 오류는 별도로 고칩니다.

### 평가 시나리오와의 대응
| ID | 처리 | 평가에서 보는 곳 |
|---|---|---|
| TQ-06 | `unsupported_lookup` 422 | `error.message` |
| TQ-07 | `unsupported_lookup` 422 | `error.message` |
| TQ-08 | `missing_fields` 422 (여행지 누락), 같은 `session_id` 로 이어감 | `error.message`, `error.missing` |
| TQ-10 | 200, 충돌 안내 + 3안 | `answer.plainAnswer` |
| TQ-11 | `secret_request` 422 | `error.message` |
| TQ-12 | `personal_data_request` 422 | `error.message` |

- 422 응답에는 `answer` 가 없으므로, 평가에서 위 시나리오는 `error.message` 를 확인합니다. CSV 의 `expected_traits` 는 그대로 두고 판정 위치만 이 표를 따릅니다.
