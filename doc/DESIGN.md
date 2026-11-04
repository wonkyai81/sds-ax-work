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
| 사람이 고칠 수 있는 오류 | 여행지 누락, 예산 0 이하 | `validate_request` 에서 오류 응답. `interrupt()` 로 입력 요청하는 방식은 사용하지 않음 (`/query` 는 요청마다 응답을 주므로 재질문은 클라이언트가 다시 보냄) |
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
