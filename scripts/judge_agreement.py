"""Validate the LLM judge against human labels.

  python scripts/judge_agreement.py sample   # write the blind sample to label
  python scripts/judge_agreement.py score    # compare human labels with the judge

The sample is stratified by the judge's label (non-"correct" labels are
over-sampled) so that disagreements on errors can be seen at all; agreement on
this sample is therefore not the agreement on the whole set. The human file
shows the question, reference answer, evidence and candidate answer but not
the judge's label (blind labelling).
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANSWERS = ROOT / "results" / "answers_per_item.jsonl"
QA = ROOT / "eval" / "qa_final.jsonl"
SAMPLE = ROOT / "eval" / "review" / "judge_sample.jsonl"
HUMAN = ROOT / "eval" / "review" / "judge_human_labels.jsonl"
OUT = ROOT / "results" / "judge_agreement.json"
LABELS = ["correct", "partial", "incorrect", "abstained"]
PER_CONDITION = 20
MAX_PER_NONCORRECT_LABEL = 5
SEED = 3


def rows():
    items = {json.loads(l)["qid"]: json.loads(l) for l in QA.open(encoding="utf-8")}
    for line in ANSWERS.open(encoding="utf-8"):
        r = json.loads(line)
        if r["qid"] not in items:
            continue  # unanswerable items are not judged
        it = items[r["qid"]]
        for cond in ("rag", "closed_book"):
            yield {"sid": f"{r['qid']}:{cond}", "qid": r["qid"], "condition": cond,
                   "question": it["question"], "reference_answer": it["answer"],
                   "evidence": it["evidence"], "candidate_answer": r[f"{cond}_answer"],
                   "judge_label": r[f"{cond}_judge"]["label"]}


def make_sample():
    rng = random.Random(SEED)
    all_rows = list(rows())
    sample = []
    for cond in ("rag", "closed_book"):
        pool = [r for r in all_rows if r["condition"] == cond]
        chosen = []
        for lab in LABELS[1:]:
            group = [r for r in pool if r["judge_label"] == lab]
            chosen += rng.sample(group, min(MAX_PER_NONCORRECT_LABEL, len(group)))
        correct = [r for r in pool if r["judge_label"] == "correct"]
        chosen += rng.sample(correct, min(len(correct), PER_CONDITION - len(chosen)))
        sample += chosen
    rng.shuffle(sample)
    with SAMPLE.open("w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(sample)} items -> {SAMPLE}")
    print(Counter((r["condition"], r["judge_label"]) for r in sample))


def cohen_kappa(a, b, labels):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] * cb[l] for l in labels) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def score():
    sample = {json.loads(l)["sid"]: json.loads(l) for l in SAMPLE.open(encoding="utf-8")}
    human = [json.loads(l) for l in HUMAN.open(encoding="utf-8")]
    pairs = [(h["human_label"], sample[h["sid"]]["judge_label"], sample[h["sid"]]["condition"])
             for h in human if h.get("human_label")]
    hl, jl = [p[0] for p in pairs], [p[1] for p in pairs]
    binary = lambda xs: ["correct" if x == "correct" else "not_correct" for x in xs]
    out = {
        "n": len(pairs),
        "agreement_4_labels": round(sum(x == y for x, y in pairs_ab(hl, jl)) / len(pairs), 3),
        "kappa_4_labels": round(cohen_kappa(hl, jl, LABELS), 3),
        "agreement_correct_vs_not": round(sum(x == y for x, y in pairs_ab(binary(hl), binary(jl))) / len(pairs), 3),
        "kappa_correct_vs_not": round(cohen_kappa(binary(hl), binary(jl), ["correct", "not_correct"]), 3),
        "confusion_human_x_judge": {f"{h}|{j}": c for (h, j), c in sorted(Counter(zip(hl, jl)).items())},
        "by_condition": {c: round(sum(p[0] == p[1] for p in pairs if p[2] == c) /
                                  max(1, sum(p[2] == c for p in pairs)), 3) for c in ("rag", "closed_book")},
        "note": "Sample is stratified by judge label (errors over-sampled); not the agreement on the full set.",
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


def pairs_ab(a, b):
    return list(zip(a, b))


if __name__ == "__main__":
    {"sample": make_sample, "score": score}[sys.argv[1]]()
