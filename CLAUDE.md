# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

The repository has no code yet. It contains only `doc/SERVICE.md`, the service definition for a travel-planning agent web app, and an empty git history (no commits). `doc/SERVICE.md` is the source of truth for product behavior. The runnable Seed is `seed_ec1db24038dc`, generated from the current docs and following `doc/TASKS.md`. The earlier `seed_ebc6e7d8804c` is superseded (it used `boto3` and `app/` and `ui/` paths).

The submission rules in this file are fixed. Follow them even when a planned file would be more convenient. Open decisions are listed in `doc/SERVICE.md` section 9 and must be answered before the files they affect are created.

## Submission rules (fixed)

- Stack: Python, LangChain, LangGraph. Models come from Amazon Bedrock through `ChatBedrockConverse`. Create agents with `create_agent` from `langchain.agents`.
- Layout: `src/agent.py` (main agent graph), `src/tools.py` (domain tools), `src/api.py` (FastAPI app), `src/config.py` (reads `.env`; approved new file), `src/ui.py` (Streamlit verification UI, kept), `web/index.html` (HTML + JS + Tailwind UI served by `GET /`; shows progress through `POST /query/stream`, an NDJSON variant of `/query`), `data/` (mock per-item KRW cost samples only, not real external data; no external lookups in v1), `evaluation/` (`test_queries.csv` covering AC1–AC6, plus the evaluation report). `tests/` is added and approved by the user. `src/retriever.py` is not created because RAG is out of scope. Do not move or rename the listed files.
- API: `POST /query` reads `question` (one natural-language line) and optional `session_id` (omitted on the first request; the server returns one inside `answer`). It returns `answer`, `contexts`, and `trace`. `answer` is a JSON object with `session_id`, `plainAnswer` (text), and `html` (report HTML, empty when there is no report). Errors use `{"error": {"code", "message"}}` with HTTP 422, 404, or 500. Non-trip requests (secrets, personal data, real-time lookup) and missing trip fields are rejected by a rule-based Guardrail in `src/tools.py` and `src/api.py` before the graph runs, as HTTP 422; see `doc/DESIGN.md` section 16. Required trip inputs are destination, trip length, and party size (people, adults only); budget is optional, and with no budget the plans are generated without a spending limit. The departure city is optional and defaults to Seoul (shown as the default); flights are searched origin to destination, and the report shows both the outbound and the return flight with dates and times. The report table has departure and arrival time columns next to the day column (HH:MM). If the user states travel dates (e.g. `12월 20일부터 23일까지`, `2026-12-20~2026-12-22`), MCP lookups use those dates; otherwise they use today + 30 days (D+30), and the input box says so. A 1-day trip has no lodging. Per-person items (tickets, flights, restaurant meals) carry a unit price and the code computes the total as `unit price x people`; the report shows it as `unit price x people = total`. `contexts` holds tool-call results. Session state (selected plan, chat history) lives in server memory.
- Code: one role per file. Write Korean docstrings for functions and tools. Read secrets from `.env` and never hardcode them.
- Do not create files the user has not asked for. Do not rewrite an existing file wholesale; change only the parts that need it.

## Planned commands

No build, lint, or run tooling is set up yet. Once tests exist, they are pytest runs, one file per acceptance criterion in SERVICE.md section 7. The test file location is an open decision (see SERVICE.md section 9).

- Run one test file: `python -m pytest <path>/test_readjust.py -q`
- Run one test: `python -m pytest <path>/test_readjust.py::<test_name> -q`

Streamlit (`src/ui.py`) is used only for manual verification. Run commands are not set up yet.

## Architecture (planned)

- **Split of responsibilities:** the LangGraph agent in `src/agent.py` drives itinerary generation, the re-adjustment flow, and chat edits. Tools in `src/tools.py` do the deterministic work: totals, budget verdicts, and attempt counting. Streamlit (verification only) collects input, shows the plans, sends chat messages, and shows the report. It calls `POST /query` and holds no agent logic.
- **Budget decisions are code, not LLM:** the LLM returns per-item KRW estimates only. Tools sum them and decide each plan's verdict. 맛집형 and 명소형 must be at or under the full budget; 가성비형 at or under 80% of it.
- **Re-adjustment rules:** each over-budget plan gets at most 3 automatic re-adjustment attempts, counted per plan. If none fits, show the attempt with the smallest overage, with a warning. A chat edit that pushes the selected plan over budget runs the same loop again.
- **Bedrock access:** `ChatBedrockConverse`. Region and model come from `AWS_DEFAULT_REGION` and `BEDROCK_MODEL_ID` in `.env`, never from code. The user never enters credentials. Tests mock the model and make no network calls.
- **State:** the selected plan and chat history live only while the browser tab stays open. There is no database, file storage, login, or external data lookup in v1. External data lookup through MCP is planned for v2. RAG is excluded. Do not add any of these without a new interview or updated SERVICE.md.
- **Currency:** all amounts and budget checks are in KRW, including for overseas destinations. The LLM estimates in KRW directly; do not add exchange-rate handling.
