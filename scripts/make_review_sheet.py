"""Draw a stratified sample of the QA set for human review and write a CSV.

The reviewer fills `verdict` (ok / fix / drop) and, for "fix", the corrected
fields. scripts/apply_review.py (Day 5) folds the verdicts back into the set
and reports the acceptance rate.
"""

import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "eval" / "qa.jsonl"
OUT = ROOT / "eval" / "review" / "review_50.csv"
PER_CATEGORY = {"province": 10, "destination": 13, "heritage": 12, "food": 8, "festival": 7}
SEED = 7

COLUMNS = ["qid", "category", "title", "question", "answer", "evidence", "url",
           "verdict", "corrected_question", "corrected_answer", "note"]


def main() -> None:
    items = [json.loads(line) for line in QA.open(encoding="utf-8")]
    rng = random.Random(SEED)
    sample = []
    for cat, n in PER_CATEGORY.items():
        pool = [q for q in items if q["category"] == cat]
        sample += sorted(rng.sample(pool, n), key=lambda q: q["qid"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so that Excel shows Vietnamese characters correctly
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for q in sample:
            w.writerow({k: q.get(k, "") for k in COLUMNS})
    print(f"{len(sample)} rows -> {OUT}")


if __name__ == "__main__":
    main()
