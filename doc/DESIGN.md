# 여행 일정 에이전트 웹앱 — DESIGN

> 기준 문서: `doc/SERVICE.md` (제품 결정). 설계 원칙: `doc/langgraph-design.md`. 제출 규약: `CLAUDE.md`.
> 충돌 시 우선순위는 SERVICE.md → CLAUDE.md → langgraph-design.md 입니다. 충돌한 지점은 15절에 기록했습니다.

## 1. 설계 개요

- 에이전트는 LangGraph `StateGraph` 하나로 구성합니다. 입력 → 3안 생성 → 예산 검사·재조정 → 사용자 선택 → 대화 수정 → 리포트의 흐름을 노드와 엣지로 표현합니다.
- 예산 판정, 합계, 재조정 횟수, 가장 근접한 안 선택은 **도구(코드)** 가 수행합니다. LLM은 항목별 원화 금액 추정과 대화 수정 결과 생성만 맡습니다.
- 사용자 선택과 대화 입력은 `interrupt()` 로 그래프를 멈춘 뒤, 다음 `/query` 요청에서 `Command(resume=...)` 으로 재개합니다. `session_id` 가 LangGraph `thread_id` 입니다.
- 상태는 원본 데이터만 저장합니다. 합계, 초과 금액, 가장 근접한 안, HTML 은 필요할 때 계산합니다.

```
START
  │
  ▼
validate_request ──(invalid)──▶ END (오류 응답)
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
                                                  collect_plans
                                                        │
                                                        ▼
                                                await_selection  ◀── interrupt
                                                        │ resume: select
                                                        ▼
                                                await_chat ◀──────┐  interrupt
                                                        │ resume: chat │
                                                        ▼              │
                                                  chat_edit           │
                                                        │ (over) → readjust_selected → check_budget_selected
                                                        ▼              │
                                                  render_answer ───────┘
                                                        │ resume: done
                                                        ▼
                                                       END
```

## 2. 노드 분해

langgraph-design.md 1절의 노드 종류 기준으로 분류합니다.

| 노드 | 종류 | 책임 | 재시도 정책 |
|---|---|---|---|
| `validate_request` | 데이터 (검증) | 여행지·기간·예산 형식 확인. 제한 없음 (SERVICE 2절) | 없음 |
| `generate_plan` | LLM | 안 1개(맛집형/명소형/가성비형)의 일차·시간대별 항목과 원화 금액 생성 | `RetryPolicy` (일시 오류) |
| `check_budget` | 도구 (코드) | 합계 계산, 목표 예산 대비 판정, 초과 금액 계산 | 없음 (결정적) |
| `readjust_plan` | LLM | 초과 정보를 넣고 항목 금액을 줄인 안 재생성 | `RetryPolicy` (일시 오류) |
| `pick_closest` | 도구 (코드) | 시도 중 초과 금액이 가장 작은 안 선택 | 없음 |
| `collect_plans` | 데이터 | 3안 결과를 상태에 모음 | 없음 |
| `await_selection` | 사용자 입력 | 3안 제시, 사용자 선택 대기 | 해당 없음 (`interrupt`) |
| `await_chat` | 사용자 입력 | 선택한 안에 대한 대화 입력 대기 또는 종료 대기 | 해당 없음 (`interrupt`) |
| `chat_edit` | LLM (에이전트) | 대화 요청을 반영해 선택한 안의 항목 수정 | `RetryPolicy` (일시 오류) |
| `check_budget_selected` | 도구 (코드) | 수정된 선택 안의 합계와 판정 | 없음 |
| `readjust_selected` | LLM | 수정 후 초과 시 재조정 (안별 최대 3회 규칙 동일) | `RetryPolicy` |
| `render_answer` | 도구 (코드) | `plainAnswer` 텍스트와 `html` 리포트 생성 (상태에 저장하지 않음) | 없음 |

- **액션 단계 없음.** v1에는 외부 부작용이 없으므로 캐시 금지 노드도 없습니다.
- **외부 데이터 조회 노드 없음 (v1).** SERVICE 2절에 따라 v1 은 외부 API 를 호출하지 않습니다. v2 에서 MCP 연동 노드를 추가합니다.

## 3. 상태 설계

원칙: 원본 데이터만 저장합니다. 파생 값(합계, 잔액, 초과 여부, 가장 근접한 안, HTML, 프롬프트 문자열)은 상태에 넣지 않습니다.

```python
from typing import Annotated, Literal, TypedDict
from langgraph.graph.message import add_messages

class TripRequest(TypedDict):
    destination: str          # 여행지 (국내·해외 제한 없음)
    days: int                 # 여행 기간 (제한 없음)
    budget_krw: int           # 사용자 예산 (원)

class ItineraryItem(TypedDict):
    day: int                  # 1부터 시작
    time_slot: Literal["아침", "오후", "저녁"]   # 고정 3개
    activity: str
    cost_krw: int             # LLM 추정 원화 금액
    transport: str
    note: str                 # 비용 근거 또는 특이 사항

PlanType = Literal["맛집형", "명소형", "가성비형"]

class PlanState(TypedDict):
    plan_type: PlanType
    attempts: list[list[ItineraryItem]]   # 시도별 항목 목록 (원본). 최대 1 + 재조정 3회
    # 총 재조정 횟수는 len(attempts) - 1 로 계산. 별도 카운터를 두지 않음

class GraphState(TypedDict):
    trip: TripRequest
    plans: dict[PlanType, PlanState]      # 3안 전체 (collect_plans 이후 채워짐)
    selected: PlanType | None             # 사용자가 고른 안
    selected_items: list[ItineraryItem]   # 선택 후 대화 수정이 반영된 항목 (원본)
    chat_history: Annotated[list, add_messages]   # 대화 이력 (리듀서: 누적)
    tool_results: Annotated[list[dict], lambda a, b: a + b]  # contexts 용 도구 호출 결과 (누적)
    trace: Annotated[list[str], lambda a, b: a + b]          # 거친 노드 이름 (누적)
    errors: Annotated[list[str], lambda a, b: a + b]         # LLM이 고칠 수 있는 오류 (2절 6장)
```

- `chat_history` 는 LangChain 메시지 객체를 담으므로 `add_messages` 리듀서를 씁니다.
- `tool_results` 와 `trace` 는 누적 리듀서입니다. 응답의 `contexts` 와 `trace` 가 여기서 나옵니다.
- 리듀서는 `Annotated` 로만 선언합니다. 누적 로직을 노드 안에서 직접 다루지 않습니다.
- 프롬프트 문자열은 상태에 없습니다. 노드 함수 안에서 `trip` 과 `items` 를 보고 만듭니다.

## 4. 엣지와 제어 흐름

| 위치 | 방식 | 이유 |
|---|---|---|
| `validate_request` → 3개 `generate_plan` | `Send` × 3 (`plan_type` 별 입력) | 안 3개는 서로 독립이고, 각자 재조정 카운터를 가짐. 개수는 3으로 고정이지만 병렬 실행을 위해 `Send` 사용 |
| `check_budget` → `readjust_plan` / `pick_closest` / `collect_plans` | `add_conditional_edges` | 상태(시도 횟수, 충족 여부)만 보고 분기 |
| `await_selection`, `await_chat` | `interrupt()` + 재개 값 분기 | HITL (langgraph-design 7절) |
| `chat_edit` → `check_budget_selected` | 일반 엣지 | 항상 검사 |
| 재개 값 `done` | 조건부 엣지 → `END` | 사용자가 종료를 선택한 경우 |

- `Command` 는 사용하지 않습니다. 상태 업데이트와 다음 노드를 동시에 결정할 필요가 없고, 조건부 엣지로 충분합니다.
- `Command[Literal[...]]` 를 쓰는 노드가 없으므로 목적지 타입 힌트 규칙은 조건부 엣지의 라우터 반환 타입으로 대신합니다.
- 병렬 분기(`Send`) 안에서는 `interrupt` 를 호출하지 않습니다. 사용자 개입은 3안이 모인 뒤 `await_selection` 에서만 합니다.

## 5. 예산 재조정 루프

SERVICE 4절 규칙을 그대로 옮깁니다.

| 항목 | 규칙 | 구현 위치 |
|---|---|---|
| 목표 예산 | 맛집형·명소형 = `trip.budget_krw`, 가성비형 = `trip.budget_krw × 0.8` | `tools.py` (`target_budget`) |
| 판정 | `total ≤ target` 이면 충족 | `tools.py` (`check_budget`) |
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
| `session_id` 있음, `question` 이 안 선택 | `Command(resume={"type": "select", "plan": ...})` → `await_chat` 에서 interrupt |
| `session_id` 있음, `question` 이 대화 수정 | `Command(resume={"type": "chat", "text": ...})` → `chat_edit` → `render_answer` → `await_chat` 에서 다시 interrupt |
| `session_id` 있음, `question` 이 종료 | `Command(resume={"type": "done"})` → `END` |

- `interrupt` 의 payload 는 각 단계에서 사용자에게 보여줄 내용입니다. `await_selection` 은 3안 요약, `await_chat` 은 현재 선택 안 요약을 넣습니다. `plainAnswer` 와 같은 텍스트를 쓰되 HTML 은 넣지 않습니다.
- 재개 값은 사용자가 입력한 자연어를 그대로 쓰지 않습니다. `/query` 처리 계층(`api.py`)이 `question` 을 보고 `type` 을 정합니다. 이 분류는 LLM에 맡기지 않고 **API 계층에서 요청 상태(선택 대기인지, 대화 대기인지)** 로 결정합니다. 그래서 `question` 의 의미와 무관하게 대기 중인 `interrupt` 종류에 따라 해석합니다.
- `session_id` 가 없거나 알 수 없으면 새 세션으로 시작하지 않고 404 를 돌려줍니다. 새 세션은 첫 요청에서만 만듭니다.

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
| 일시 오류 | Bedrock 네트워크 오류, 스로틀링 | LLM 노드에 `RetryPolicy(max_attempts=3, initial_interval=1.0)` 와 timeout. 예산 재조정 횟수와 **별개** |
| LLM이 고칠 수 있는 오류 | 응답 JSON 파싱 실패, 항목 필드 누락 | `errors` 에 기록 후 `generate_plan` / `readjust_plan` 으로 되돌림. 같은 노드 재시도 횟수는 `RetryPolicy` 와 별도로 1회로 제한 |
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
| `render_plain_answer(plans, selected)` | 상태의 일부 | `plainAnswer` 텍스트 |
| `render_report_html(plan_type, items, budget_krw)` | 선택 안 | HTML 문자열 (AC4 항목 포함) |

- `render_report_html` 은 LLM 이 만든 문자열(활동명, 비고)을 HTML 에 넣기 전에 이스케이프합니다. 이스케이프하지 않으면 리포트가 HTML 주입 경로가 됩니다.

### `src/agent.py` (그래프와 LLM 호출)
- 모델: `ChatBedrockConverse(model=BEDROCK_MODEL_ID, region_name=AWS_DEFAULT_REGION)`. 두 값은 `.env` 에서 읽습니다.
- `generate_plan` / `readjust_plan`: `with_structured_output` 으로 `list[ItineraryItem]` 형태를 받습니다. 구조화 출력이 실패하면 2절 6장 규칙을 따릅니다.
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

## 11. 리포트 구성 (AC4)

`render_report_html` 이 만드는 HTML 은 아래 항목을 모두 포함해야 합니다 (SERVICE 5절).

- 일차별·시간대별 일정 (표 또는 목록)
- 항목별 활동, 비용(원), 교통수단, 비고
- 항목별 비용 합계
- 예산 대비 잔액 = 사용자 예산 − 합계 (초과 시 음수로 표시)
- 초과 경고 (합계가 목표 예산을 넘을 때). 가장 근접한 안을 보여줄 때도 표시

- 잔액은 **사용자 예산** 기준입니다. 목표 예산(가성비형 80%)과 헷갈리지 않도록 리포트에 두 값을 따로 씁니다.
- 가장 근접한 안이면 "예산 미충족" 문구를 함께 씁니다.

## 12. 설정

| 변수 | 값 | 읽는 곳 |
|---|---|---|
| `AWS_DEFAULT_REGION` | `us-east-1` | `agent.py` (모델 생성) |
| `BEDROCK_MODEL_ID` | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | `agent.py` |

- `.env` 는 `src/config.py` 에서 읽습니다 (15절 확정). 다른 파일은 설정 값을 `config.py` 를 통해서만 받고, 코드에 값을 쓰지 않습니다 (CLAUDE.md).
- AWS 자격 증명은 사용자 입력이 없습니다. 표준 AWS 자격 증명 체인을 씁니다.

## 13. 테스트와 수용 기준 매핑

| AC | 검증 파일 | 모킹 대상 | 확인 내용 |
|---|---|---|---|
| AC1 | `tests/test_budget.py`, `tests/test_plans.py` | `ChatBedrockConverse` | 3안 생성, 합계와 판정이 `check_budget` 결과와 일치 |
| AC2 | `tests/test_readjust.py` | 같음 | 안별 재조정 최대 3회, 가장 근접한 안 선택, 초과 경고 |
| AC3 | `tests/test_chat_edit.py` | 같음 | 대화 수정 반영, 초과 시 재조정 실행 |
| AC4 | `tests/test_report.py` | 없음 (`render_report_html` 직접 호출) | 리포트에 AC4 항목이 모두 있고, LLM 문자열이 이스케이프됨 |
| AC5 | `tests/test_config.py` | 환경변수 | 리전·모델 ID 를 환경변수에서 읽음, 사용자 키 입력 없음 |
| AC6 | `evaluation/test_queries.csv` + 수동 시연 | 없음 (실제 Bedrock) | 대표 입력으로 흐름 확인 |

- 자동 테스트는 네트워크를 쓰지 않습니다.
- AC1~AC6 전부를 `evaluation/test_queries.csv` 에 옮깁니다 (SERVICE 9절 8항).

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
| 8 | 일부 안만 실패 | 나머지 안만 보여줌 |

- 새 파일: `src/config.py`. SERVICE 8절과 CLAUDE.md 의 파일 목록에 반영했습니다.
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
