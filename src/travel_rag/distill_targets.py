"""Second fine-tuning round: replace the short QA answers with Gemini's own RAG answers.

Round 1 (make_sft_data.py) used the evaluation-style short answers ("Năm 1904.")
as targets, and the fine-tuned model learned to answer tersely, which the
judge often scored as partial. Here each training prompt (same question, same
5 passages) is answered by the production generator with the production RAG
prompt and schema, and that answer becomes the target: sequence-level
distillation from Gemini into Qwen.

Label checks against how the example was built:
  answerable example, Gemini found=true    keep Gemini's answer
  answerable example, Gemini found=false   drop (the teacher missed; unclear label)
  abstain example,    Gemini found=false   keep (abstention)
  abstain example,    Gemini found=true    keep Gemini's answer and count it:
      another article in the context did contain the answer, so the round-1
      abstention label was wrong (the noise noted in make_sft_data.py)

Writes data/sft/train_v2.jsonl and val_v2.jsonl; stops cleanly at the daily
quota, and a rerun continues from the cache. Summary: results/sft_v2_summary.json

Usage:  python -m travel_rag.distill_targets
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from travel_rag.config import RESULTS_DIR
from travel_rag.llm import DailyQuotaError, GeminiClient, LLMError
from travel_rag.make_sft_data import SFT_DIR, target
from travel_rag.rag import RAG_SCHEMA, RAG_SYSTEM

WORKERS = 2


def main() -> None:
    llm = GeminiClient()
    quota_hit = False

    def teach(row):
        nonlocal quota_hit
        if quota_hit:
            return None
        try:
            return llm.generate_json(row["prompt"], RAG_SYSTEM, RAG_SCHEMA)
        except DailyQuotaError:
            quota_hit = True
            return None
        except LLMError:
            return "error"

    summary = {"teacher": llm.model}
    for split in ("val", "train"):  # val first: it is small and needed for the loss curve
        rows = [json.loads(l) for l in (SFT_DIR / f"{split}.jsonl").open(encoding="utf-8")]
        with ThreadPoolExecutor(WORKERS) as pool:
            outs = list(pool.map(teach, rows))
        kept, counts = [], Counter()
        for row, out in zip(rows, outs):
            if out is None:
                counts["not_done_quota"] += 1
                continue
            if out == "error":
                counts["llm_error"] += 1
                continue
            found = bool(out.get("found")) and bool((out.get("answer") or "").strip())
            cites = [c for c in out.get("citations") or [] if isinstance(c, int) and 1 <= c <= 5]
            if row["kind"] == "answerable" and not found:
                counts["dropped_teacher_abstained_on_answerable"] += 1
                continue
            if row["kind"] == "abstain" and found:
                counts["relabelled_abstain_to_answer"] += 1
            counts["kept_found" if found else "kept_abstain"] += 1
            kept.append({**row, "target": target(found, (out.get("answer") or "").strip(), cites),
                         "target_round1": row["target"]})
        with (SFT_DIR / f"{split}_v2.jsonl").open("w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[split] = {"n_in": len(rows), "n_out": len(kept), **dict(counts)}
        words = [len(json.loads(r["target"])["answer"].split()) for r in kept if json.loads(r["target"])["found"]]
        old = [len(json.loads(r["target_round1"])["answer"].split()) for r in kept
               if json.loads(r["target_round1"])["found"]]
        summary[split]["mean_answer_words"] = {"round1": round(sum(old) / max(1, len(old)), 1),
                                               "round2": round(sum(words) / max(1, len(words)), 1)}
    summary["quota_reached"] = quota_hit
    summary["llm_stats"] = llm.stats
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "sft_v2_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
