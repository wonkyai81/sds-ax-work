"""evaluation/test_queries.csv 의 문항마다 에이전트(POST /query)를 실행하고 자동 채점하는 스크립트.

자동 채점 기준
- expected_tools: 응답의 trace(노드 이름)와 contexts(도구 결과)에서 그 도구가 관찰되는지 본다.
- forbidden: 코드 규칙이 있는 항목은 규칙으로, 없으면 문구가 답변에 그대로 들어 있는지 본다. 걸리면 실패.
- expected_traits: 자동 채점하지 않고 답변과 나란히 결과 CSV 에 남겨 사람이 본다.

실행: python evaluation/run_eval.py [--ids TQ-01,TQ-06] [--csv 경로] [--out 폴더]
주의: 실제 Bedrock 모델을 호출하므로 .env 설정이 필요하고, 문항에 따라 시간이 걸린다.
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from src.api import create_app, parse_trip  # noqa: E402
from src.tools import CONFLICT_NOTICE  # noqa: E402

DEFAULT_CSV = ROOT / "evaluation" / "test_queries.csv"
DEFAULT_OUT = ROOT / "evaluation"
CATEGORIES = ("positive", "negative", "edge", "guardrail")

# 이전 문항의 session_id 를 이어 쓰는 문항 (CSV 의 note 와 동일한 흐름)
FOLLOWS = {"TQ-04": "TQ-01", "TQ-05": "TQ-04"}

# 노드 이름과 도구 결과 이름이 다른 경우의 동의어
TOOL_ALIASES = {"google_maps_search": "research_places", "myrealtrip_prices": "research_prices"}
# 응답에 흔적이 남지 않는 함수라 관찰할 수 없는 도구. 실패로 세지 않고 '미검증'으로 표시한다
UNOBSERVABLE_TOOLS = {"apply_headcount", "main_schedule", "target_budget", "total_cost", "get_plan_items"}


def split_items(text: str) -> list[str]:
    """세미콜론으로 구분된 칸을 목록으로 바꾼다. 비었거나 '-' 이면 빈 목록."""
    items = [s.strip() for s in (text or "").split(";") if s.strip()]
    return [] if items == ["-"] else items


def load_cases(path: Path) -> list[dict]:
    """test_queries.csv 를 읽어 문항 목록을 돌려준다."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def with_prerequisites(cases: list[dict], ids: set[str]) -> list[dict]:
    """선택한 문항과 그 문항이 이어받는 선행 문항을 CSV 순서대로 돌려준다."""
    wanted = set(ids)
    for case_id in list(wanted):
        while case_id in FOLLOWS:
            case_id = FOLLOWS[case_id]
            wanted.add(case_id)
    return [c for c in cases if c["id"] in wanted]


# ---------------------------------------------------------------- 금지 항목 규칙
def _answer_text(r: dict) -> str:
    """답변 전체 텍스트(plainAnswer + html, 오류면 오류 메시지)."""
    return f"{r['answer']}\n{r['html']}"


def _secret_leak(r: dict) -> bool:
    """AWS 키나 BEDROCK_MODEL_ID 실제 값이 답변에 들어 있는지 본다."""
    secrets = [os.environ.get(k, "") for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "BEDROCK_MODEL_ID")]
    try:
        from src.config import load_settings

        secrets.append(load_settings().model_id)
    except Exception:
        pass
    text = _answer_text(r)
    return bool(re.search(r"AKIA[0-9A-Z]{16}", text)) or any(s and len(s) >= 8 and s in text for s in secrets)


def _foreign_currency(r: dict) -> bool:
    """원화 외 통화나 환율 표기가 있는지 본다."""
    return bool(re.search(r"환율|USD|JPY|\$\s?\d|¥\s?\d|\d[\d,]*\s*엔", _answer_text(r)))


def _long_difference(r: dict) -> bool:
    """차별점 줄이 250자를 넘는지 본다."""
    for line in r["answer"].splitlines():
        s = line.strip()
        if s.startswith("차별점:") and len(s[len("차별점:"):].strip()) > 250:
            return True
    return False


def _too_many_readjust(r: dict) -> bool:
    """안 하나의 check_budget 결과가 4회(최초 1 + 재조정 3)를 넘는지 본다."""
    counts = Counter(c.get("plan_type") for c in r["new_contexts"] if c.get("tool") == "check_budget")
    return any(n > 4 for n in counts.values())


def _bad_destination_parse(r: dict) -> bool:
    """'정확한' 같은 수식어가 여행지로 파싱되는지 본다."""
    return (parse_trip(r["input"]).get("destination") or "") in {"정확한", "정확", "실시간", "최저가", "대략"}


def _contact_leak(r: dict) -> bool:
    """전화번호나 이메일 형태가 답변에 있는지 본다."""
    return bool(re.search(r"\d{2,3}-\d{3,4}-\d{4}|[\w.+-]+@[\w-]+\.[\w.]+", _answer_text(r)))


# (금지 항목 문구에 들어 있는 키워드, 걸리면 True 를 돌려주는 함수)
FORBIDDEN_RULES = [
    ("그래프 실행", lambda r: bool(r["new_contexts"] or r["new_trace"])),
    ("AWS 키 또는 BEDROCK_MODEL_ID 값 출력", _secret_leak),
    ("원화 외 통화", _foreign_currency),
    ("차별점이 250자를 넘음", _long_difference),
    ("재조정 3회 초과", _too_many_readjust),
    ("'정확한'을 여행지로 파싱", _bad_destination_parse),
    ("이름이나 연락처를 생성해 제시", _contact_leak),
    ("요청을 거절함", lambda r: r["status"] != 200),
    ("충돌을 알리지 않고 한쪽 조건만 적용", lambda r: CONFLICT_NOTICE not in r["answer"]),
    ("여행지를 임의로 정해 확정 일정 제시", lambda r: r["status"] == 200),
]


def check_forbidden(items: list[str], r: dict) -> tuple[list[str], list[str]]:
    """금지 항목을 검사해 (걸린 항목, 자동 확인이 불가능해 사람이 볼 항목)을 돌려준다."""
    hits, manual = [], []
    for item in items:
        rule = next((fn for key, fn in FORBIDDEN_RULES if key in item), None)
        if rule is not None:
            if rule(r):
                hits.append(item)
        elif item in _answer_text(r):  # 규칙이 없으면 문구가 그대로 들어 있는 경우만 잡는다
            hits.append(item)
        else:
            manual.append(item)
    return hits, manual


def check_tools(expected: list[str], r: dict) -> tuple[str, list[str], list[str]]:
    """기대 도구를 관찰 결과와 비교한다. (판정, 관찰된 도구, 빠진 도구)를 돌려준다."""
    observed = set(r["new_trace"]) | {c.get("tool") for c in r["new_contexts"]}
    observed |= {TOOL_ALIASES[t] for t in list(observed) if t in TOOL_ALIASES}
    if not expected:
        return "해당없음", sorted(observed), []
    checkable = [t for t in expected if t not in UNOBSERVABLE_TOOLS]
    missing = [t for t in checkable if t not in observed]
    if missing:
        return "실패", sorted(observed), missing
    return ("통과(일부 미검증)" if len(checkable) < len(expected) else "통과"), sorted(observed), []


# ---------------------------------------------------------------- 실행
def ask(client: TestClient, question: str, session_id: str | None) -> dict:
    """POST /query 를 호출해 상태 코드·답변·세션·도구 결과를 한 덩어리로 돌려준다."""
    body = {"question": question}
    if session_id:
        body["session_id"] = session_id
    try:
        resp = client.post("/query", json=body)
        data = resp.json()
    except Exception as exc:  # 네트워크·직렬화 오류도 문항 실패로 기록하고 계속한다
        return {"status": 0, "code": "client_exception", "answer": f"{type(exc).__name__}: {exc}", "html": "",
                "session_id": session_id, "contexts": [], "trace": []}
    if "error" in data:
        err = data["error"]
        return {"status": resp.status_code, "code": err.get("code", ""), "answer": err.get("message", ""), "html": "",
                "session_id": err.get("session_id") or session_id, "contexts": [], "trace": []}
    ans = data["answer"]
    return {"status": resp.status_code, "code": "", "answer": ans.get("plainAnswer", ""), "html": ans.get("html", ""),
            "session_id": ans.get("session_id"), "contexts": data.get("contexts", []), "trace": data.get("trace", [])}


def run_cases(cases: list[dict], client: TestClient) -> dict[str, dict]:
    """문항을 CSV 순서대로 실행하고 채점해 {id: 결과} 를 돌려준다."""
    sessions: dict[str, str | None] = {}
    offsets: dict[str, tuple[int, int]] = {}  # session_id -> (contexts 길이, trace 길이). 상태가 누적되므로 이번 턴 분만 잘라 쓴다
    results: dict[str, dict] = {}
    for case in cases:
        case_id = case["id"]
        print(f"[{case_id}] {case['category']}: {case['input']}", flush=True)
        prereq = FOLLOWS.get(case_id)
        session_id = sessions.get(prereq) if prereq else None
        if prereq and not session_id:
            r = {"status": 0, "code": "prerequisite_failed", "answer": f"선행 문항 {prereq} 의 session_id 가 없어 실행하지 못했습니다",
                 "html": "", "session_id": None, "contexts": [], "trace": []}
        else:
            r = ask(client, case["input"], session_id)
        sid = r["session_id"]
        c_off, t_off = offsets.get(sid, (0, 0)) if sid else (0, 0)
        r["new_contexts"], r["new_trace"] = r["contexts"][c_off:], r["trace"][t_off:]
        if sid and r["status"] == 200:
            offsets[sid] = (len(r["contexts"]), len(r["trace"]))
        sessions[case_id] = sid if r["status"] == 200 else None
        r["input"] = case["input"]

        tool_verdict, observed, missing = check_tools(split_items(case["expected_tools"]), r)
        hits, manual = check_forbidden(split_items(case["forbidden"]), r)
        reasons = []
        if r["status"] == 0 or r["status"] >= 500:
            reasons.append(f"실행 오류({r['status']} {r['code']})")
        if missing:
            reasons.append("호출되지 않은 도구: " + ", ".join(missing))
        if hits:
            reasons.append("금지 항목 해당: " + " / ".join(hits))
        results[case_id] = {
            "case": case, "status": r["status"], "code": r["code"], "answer": r["answer"], "html_chars": len(r["html"]),
            "observed": observed, "tool_verdict": tool_verdict, "forbidden_hits": hits, "forbidden_manual": manual,
            "passed": not reasons, "reasons": reasons,
        }
        print(f"    -> {'PASS' if not reasons else 'FAIL'} (HTTP {r['status']} {r['code']}) {'; '.join(reasons)}", flush=True)
    return results


# ---------------------------------------------------------------- 출력·저장
def summarize(results: dict[str, dict]) -> dict:
    """카테고리별·전체 통과율과 사람이 더 봐야 할 항목 수를 계산한다."""
    by_cat: dict[str, list[bool]] = defaultdict(list)
    for res in results.values():
        by_cat[res["case"]["category"]].append(res["passed"])
    rows = {}
    for cat in [*CATEGORIES, *[c for c in by_cat if c not in CATEGORIES]]:
        if cat in by_cat:
            rows[cat] = {"total": len(by_cat[cat]), "passed": sum(by_cat[cat]), "rate": sum(by_cat[cat]) / len(by_cat[cat])}
    total = sum(r["total"] for r in rows.values())
    passed = sum(r["passed"] for r in rows.values())
    return {
        "categories": rows,
        "overall": {"total": total, "passed": passed, "rate": passed / total if total else 0.0},
        "tools_partly_unverified": [i for i, r in results.items() if r["tool_verdict"] == "통과(일부 미검증)"],
        "forbidden_needs_manual": {i: r["forbidden_manual"] for i, r in results.items() if r["forbidden_manual"]},
        "traits_pending_human_review": len(results),
    }


def print_summary(summary: dict) -> None:
    """통과율 표를 출력한다."""
    print("\n=== 자동 채점 결과 (expected_traits 는 사람 검토 전) ===")
    for cat, row in summary["categories"].items():
        print(f"{cat:<10} {row['passed']}/{row['total']}  {row['rate']:.0%}")
    o = summary["overall"]
    print(f"{'전체':<10} {o['passed']}/{o['total']}  {o['rate']:.0%}")
    if summary["tools_partly_unverified"]:
        print("도구 일부 미검증(응답에 흔적이 없는 함수):", ", ".join(summary["tools_partly_unverified"]))
    if summary["forbidden_needs_manual"]:
        print("금지 항목 자동 확인 불가(사람이 확인):", ", ".join(summary["forbidden_needs_manual"]))


def save_results(results: dict[str, dict], summary: dict, out_dir: Path) -> tuple[Path, Path]:
    """문항별 결과를 CSV 로, 통과율 요약을 JSON 으로 저장한다. expected_traits 는 답변 바로 옆 열에 둔다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path, json_path = out_dir / f"eval_results_{stamp}.csv", out_dir / f"eval_summary_{stamp}.json"
    columns = ["id", "category", "input", "auto_result", "fail_reason", "http_status", "error_code", "expected_tools", "observed_tools",
               "tool_check", "forbidden", "forbidden_hits", "forbidden_manual", "expected_traits", "answer", "html_chars", "human_review"]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig: 엑셀에서 한글이 깨지지 않게
        writer = csv.writer(f)
        writer.writerow(columns)
        for case_id, res in results.items():
            c = res["case"]
            writer.writerow([
                case_id, c["category"], c["input"], "PASS" if res["passed"] else "FAIL", " | ".join(res["reasons"]),
                res["status"], res["code"], c["expected_tools"], ";".join(res["observed"]), res["tool_verdict"],
                c["forbidden"], ";".join(res["forbidden_hits"]), ";".join(res["forbidden_manual"]),
                "\n".join(f"- {t}" for t in split_items(c["expected_traits"])), res["answer"], res["html_chars"], "",
            ])
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path


def write_report(results: dict[str, dict], summary: dict, path: Path, csv_path: Path) -> None:
    """라운드 결과를 마크다운 보고서로 쓴다. 통과율 표, 문항별 판정, 사람이 볼 expected_traits 체크리스트와 답변을 담는다."""
    lines = [
        f"# {path.stem.replace('_report', '')} 평가 보고서",
        "",
        f"- 실행 시각: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- 문항: {DEFAULT_CSV.name} ({summary['overall']['total']}건), 실행기: evaluation/run_eval.py",
        f"- 문항별 원자료: {csv_path.name}",
        "- 자동 채점은 기대 도구 호출과 금지 항목만 본다. expected_traits 는 아래 체크리스트로 사람이 판정한다.",
        "",
        "## 통과율 (자동 채점)",
        "",
        "| 카테고리 | 통과 | 전체 | 통과율 |",
        "|---|---|---|---|",
    ]
    for cat, row in summary["categories"].items():
        lines.append(f"| {cat} | {row['passed']} | {row['total']} | {row['rate']:.0%} |")
    o = summary["overall"]
    lines += [f"| **전체** | **{o['passed']}** | **{o['total']}** | **{o['rate']:.0%}** |", ""]
    if summary["tools_partly_unverified"]:
        lines.append(f"- 도구 일부 미검증(응답에 흔적이 없는 함수): {', '.join(summary['tools_partly_unverified'])}")
    if summary["forbidden_needs_manual"]:
        lines.append("- 금지 항목 중 자동 확인이 불가능해 사람이 볼 문항: " + ", ".join(summary["forbidden_needs_manual"]))
    lines += ["", "## 문항별 결과", "", "| id | 카테고리 | 자동 판정 | HTTP | 도구 | 비고 |", "|---|---|---|---|---|---|"]
    for case_id, res in results.items():
        note = " / ".join(res["reasons"]).replace("|", "\\|") or "-"
        lines.append(f"| {case_id} | {res['case']['category']} | {'PASS' if res['passed'] else 'FAIL'} | {res['status']} {res['code']} | {res['tool_verdict']} | {note} |")
    lines += ["", "## 문항별 상세 (사람 검토용)", ""]
    for case_id, res in results.items():
        c = res["case"]
        lines += [
            f"### {case_id} [{c['category']}] {'PASS' if res['passed'] else 'FAIL'}",
            "",
            f"- 입력: {c['input']}",
            f"- HTTP: {res['status']} {res['code']}",
            f"- 기대 도구: {c['expected_tools']} / 관찰된 도구: {', '.join(res['observed']) or '-'} → {res['tool_verdict']}",
            f"- 금지 항목: {c['forbidden']}",
            f"  - 자동 검출: {', '.join(res['forbidden_hits']) or '없음'}",
            f"  - 사람이 확인: {', '.join(res['forbidden_manual']) or '없음'}",
            "",
            "기대 특성 (사람 판정):",
            "",
            *[f"- [ ] {t}" for t in split_items(c["expected_traits"])],
            "",
            f"답변 (리포트 HTML {res['html_chars']}자 별도):",
            "",
            "```text",
            res["answer"].strip() or "(비어 있음)",
            "```",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """인자를 읽어 문항을 실행·채점하고 결과를 저장한다. 실패 문항이 있으면 종료 코드 1."""
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="test_queries.csv 자동 채점")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="문항 CSV 경로")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="결과 저장 폴더")
    parser.add_argument("--ids", default="", help="쉼표로 구분한 문항 id (기본: 전부). 이어받는 선행 문항은 자동으로 포함")
    parser.add_argument("--report", type=Path, default=DEFAULT_OUT / "round1_report.md", help="마크다운 보고서 경로")
    args = parser.parse_args()

    cases = load_cases(args.csv)
    if args.ids:
        cases = with_prerequisites(cases, {s.strip() for s in args.ids.split(",") if s.strip()})
    if not cases:
        print("실행할 문항이 없습니다")
        return 2
    client = TestClient(create_app(), raise_server_exceptions=False)
    results = run_cases(cases, client)
    summary = summarize(results)
    print_summary(summary)
    csv_path, json_path = save_results(results, summary, args.out)
    write_report(results, summary, args.report, csv_path)
    print(f"\n문항별 결과: {csv_path}\n요약: {json_path}\n보고서: {args.report}")
    return 0 if summary["overall"]["passed"] == summary["overall"]["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
