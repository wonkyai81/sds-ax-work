# LangGraph 설계 가이드

> 출처: LangChain 공식 문서 (docs.langchain.com, LangGraph Python/JS 공통 개념). 각 절의 `oss/python/langgraph/...` 경로는 해당 문서 페이지입니다.
> 용도: 설계 에이전트(architect)가 LangGraph 기반 기능의 spec을 쓸 때 참고하는 기준 문서입니다.

## 1. 설계 순서

출처: `oss/python/langgraph/thinking-in-langgraph`

1. **워크플로를 이산 단계로 분해합니다.** 각 단계가 노드 하나가 됩니다.
2. **각 노드의 종류를 정합니다.**
   - LLM 단계: 이해, 분석, 생성, 추론 판단
   - 데이터 단계: 외부 조회 (재시도, 캐시 대상)
   - 액션 단계: 외부 부작용 (재시도 대상, 캐시 금지)
   - 사용자 입력 단계: `interrupt()`로 사람 개입
3. **상태(state)를 설계합니다.** 노드 사이에서 유지돼야 하는 값만 넣습니다.
4. **노드를 구현합니다.** 노드는 상태를 받아 상태 업데이트를 반환하는 함수입니다.
5. **엣지와 제어 흐름을 연결합니다.**

**핵심 원칙:** 상태에는 원본 데이터만 저장합니다. 프롬프트 문자열은 노드 안에서 필요할 때 만듭니다. 다른 값으로부터 계산 가능한 값은 상태에 넣지 않습니다.

## 2. 상태(State) 설계

- 상태는 모든 노드가 읽고 쓰는 공유 메모리입니다. Python은 `TypedDict` 또는 Pydantic, JS는 `StateSchema`를 씁니다.
- **넣을 것:** 원본 입력, 여러 노드가 쓰는 중간 결과, 다시 조회하기 비싼 값, 디버깅용 메타데이터
- **넣지 않을 것:** 파생 값, 프롬프트 템플릿, 포맷된 문자열
- 노드가 값을 바꿀 때 덮어쓰기인지 누적인지 정합니다. 누적이 필요하면 리듀서를 씁니다. 예: 메시지 목록은 `MessagesValue` / `add_messages`.

## 3. 노드와 엣지 연결

출처: `oss/python/langgraph/graph-api`, `use-graph-api`

| 방식 | 언제 쓰나 |
|---|---|
| 일반 엣지 `add_edge` | 항상 같은 다음 노드로 갈 때 |
| 조건부 엣지 `add_conditional_edges` | 상태를 보고 분기할 때. 라우터 함수가 노드 이름을 반환 |
| `Command(update=..., goto=...)` | 한 노드에서 상태 업데이트와 다음 노드 결정을 동시에 할 때. 반환 타입 힌트에 가능한 목적지를 `Command[Literal[...]]`로 명시해야 그래프 렌더링이 맞습니다. |
| `Send` | 실행 시점에 개수를 알 수 없는 병렬 작업 (map-reduce). 노드마다 다른 입력 상태를 넘깁니다. |

## 4. 영속성과 메모리

출처: `oss/python/langgraph/persistence`, `checkpointers`, `add-memory`

- **Checkpointer (단기 기억):** 스레드 단위 상태 스냅샷. 대화 연속성, HITL, 타임 트래블, 장애 복구에 씁니다. 호출 시 `config={"configurable": {"thread_id": ...}}`가 필수입니다. 없으면 상태를 저장하거나 `interrupt` 이후 재개할 수 없습니다.
- **Store (장기 기억):** 스레드를 넘나드는 키-값 데이터. 사용자 선호, 사실 등에 씁니다.
- **Durability 모드:** `exit`(종료 시만 저장), `async`(다음 단계와 병렬로 저장), `sync`(다음 단계 전에 저장). 중간 장애 복구가 필요하면 `sync`를 씁니다.
- **운영 환경:** `InMemorySaver` / `MemorySaver`는 테스트용입니다. 운영에서는 Postgres 등 영속 checkpointer를 씁니다.

## 5. 서브그래프와 멀티 에이전트

출처: `oss/python/langgraph/use-subgraphs`, `oss/python/langchain/multi-agent/*`

- **서브그래프 persistence 모드** (`compile(checkpointer=...)`)
  - 기본값(`None`): 호출마다 새로 시작하되 부모의 checkpointer를 써서 interrupt는 가능
  - `True`: 스레드 단위로 상태가 누적됨 (다회 대화 기억)
  - `False`: 체크포인트 없음 (순수 함수처럼 동작)
  - 대부분은 기본값이 맞습니다.
- **멀티 에이전트 패턴** (LangChain 공식 분류)
  - Subagents: 메인 에이전트가 서브에이전트를 도구로 호출합니다. 라우팅이 메인에 집중됩니다.
  - Handoffs: 도구 호출이 상태 변수를 바꾸고, 이후 라우팅이 바뀝니다.
  - Skills: 한 에이전트가 필요할 때 프롬프트와 지식을 불러옵니다.
  - Router: 입력을 분류해 전문 에이전트로 보내고 결과를 합칩니다.
  - Custom workflow: LangGraph로 직접 흐름을 짭니다.
- 문서는 단일 에이전트 + 미들웨어를 대부분의 경우에 권장합니다. 서브그래프는 복잡한 자체 로직이 있는 노드가 필요할 때만 씁니다.

## 6. 오류 처리

출처: `thinking-in-langgraph`, `oss/python/langgraph/fault-tolerance`

| 오류 유형 | 처리 방법 |
|---|---|
| 일시적 오류 (네트워크, rate limit) | `RetryPolicy(max_attempts=..., initial_interval=...)`, `timeout`과 함께 |
| LLM이 고칠 수 있는 오류 (도구 실패, 파싱) | 오류를 상태에 저장하고 LLM 노드로 되돌림 |
| 사람이 고쳐야 하는 오류 (정보 부족) | `interrupt()`로 입력 요청 |
| 재시도 소진 후 복구 | `error_handler`로 보상(compensation) 분기 (`langgraph>=1.2`) |
| 예상 못 한 오류 | 잡지 않고 전파 |

주의: `interrupt()`는 오류 처리 경로를 타지 않습니다. 재시도 정책과 에러 핸들러를 우회하고 그래프를 일시정지합니다.

## 7. Human-in-the-loop

출처: `oss/python/langgraph/interrupts`, `oss/python/concepts/human-in-the-loop`

- 노드 안에서 `interrupt(payload)`를 호출하면 실행이 멈춥니다. `Command(resume=값)`으로 재개하며, `interrupt()`의 반환값이 재개 값입니다.
- interrupt를 쓰려면 checkpointer와 `thread_id`가 필요합니다.
- 도구 호출 승인은 `humanInTheLoopMiddleware`(LangChain)로 설정할 수 있습니다. 허용 결정은 `approve`, `edit`, `reject`, `respond` 중에서 고릅니다.

## 8. 설계 체크리스트

- [ ] 노드 하나는 한 가지 일만 하는가?
- [ ] 상태에 원본 데이터만 있고 프롬프트 문자열이 없는가?
- [ ] 분기는 조건부 엣지나 `Command`로 표현되고, 목적지가 타입 힌트에 명시되어 있는가?
- [ ] 외부 호출 노드에 재시도 정책이 있는가? 부작용 노드는 캐시하지 않았는가?
- [ ] 사람 개입 지점은 `interrupt()`로 설계했는가? checkpointer와 `thread_id`가 있는가?
- [ ] 장애 복구가 필요하면 durability 모드를 `sync`로 정했는가?
- [ ] 멀티 에이전트가 정말 필요한가? 미들웨어 기반 단일 에이전트로 충분하지 않은가?
- [ ] 도구를 쓴다면 description 기준을 충족하는가 (`tool-description` 스킬 참고)?

## 9. 프로젝트 적용 시 확인할 점

[REQ-001](../requirements/REQ-001-langgraph-ai-chat.md)의 채팅 그래프([app/models/chat_graph.py](../../app/models/chat_graph.py))를 이 가이드와 비교할 때 확인할 항목:

- 메모리가 thread_id 기반 checkpointer로 구현되어 있는지
- 상태에 프롬프트 문자열이 섞여 있지 않은지
- LLM 호출 노드에 재시도 정책이 있는지
