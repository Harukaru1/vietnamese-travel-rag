"""Merge the human review and the web fact check into the evaluation set.

Inputs:  eval/qa.jsonl, eval/review/review_50_labels.jsonl,
         eval/review/fact_check_web.jsonl, eval/review/resolutions.json
Output:  eval/qa_final.jsonl and eval/review/review_summary.json

Each output item carries:
  reviewed      : True for the 50 human-reviewed items
  review_verdict: ok / fix / drop / None (not reviewed)
  fact_status   : web fact-check result (None if not checked)
  fact_confirmed: True only where the reviewer explicitly agreed with it
"""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval"


def read_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def main() -> None:
    items = read_jsonl(EVAL / "qa.jsonl")
    labels = {r["qid"]: r for r in read_jsonl(EVAL / "review" / "review_50_labels.jsonl")}
    facts = {r["qid"]: r for r in read_jsonl(EVAL / "review" / "fact_check_web.jsonl")}
    res = json.loads((EVAL / "review" / "resolutions.json").read_text(encoding="utf-8"))

    out, dropped = [], []
    for it in items:
        q = it["qid"]
        lab, fc, r = labels.get(q), facts.get(q), res.get(q, {})
        it = {**it, "reviewed": lab is not None,
              "review_verdict": (lab or {}).get("verdict"),
              "fact_status": (fc or {}).get("verdict"),
              "fact_confirmed": (lab or {}).get("fact_check_confirmed") == "agree"}
        action = r.get("action") or ("drop" if it["review_verdict"] == "drop" else "keep")
        if action == "drop":
            dropped.append(q)
            continue
        if action == "fix":
            it["original_question"], it["original_answer"] = it["question"], it["answer"]
            it["question"] = r.get("question", it["question"])
            it["answer"] = r.get("answer", it["answer"])
            it["fact_status"] = r.get("fact_status", it["fact_status"])
        out.append(it)

    with (EVAL / "qa_final.jsonl").open("w", encoding="utf-8") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    reviewed = [labels[q] for q in labels]
    summary = {
        "reviewed": len(reviewed),
        "review_verdicts": dict(Counter(r["verdict"] or "blank" for r in reviewed)),
        "review_ok_rate": round(sum(r["verdict"] == "ok" for r in reviewed) / len(reviewed), 3),
        "fact_check_on_reviewed": dict(Counter(f["verdict"] for f in facts.values())),
        "fact_check_confirmed_by_reviewer": sum(r["fact_check_confirmed"] == "agree" for r in reviewed),
        "dropped": dropped,
        "final_items": len(out),
        "final_reviewed_items": sum(it["reviewed"] for it in out),
        "final_reviewed_fact_supported": sum(it["reviewed"] and it["fact_status"] == "supported" for it in out),
    }
    (EVAL / "review" / "review_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
