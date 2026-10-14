"""HAEA synthetic-only referral quality experiment. Standard library; no real data."""
import argparse
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

FIELDS = ("referral_id", "created_at", "completed_at", "document_present",
          "document_date_valid", "true_defect", "rule_flag", "rework_events",
          "simulated_review_minutes")

def generate(n=1000, seed=20261014, missing_probability=0.12, invalid_date_probability=0.06):
    if n < 1 or not 0 <= missing_probability <= 1 or not 0 <= invalid_date_probability <= 1:
        raise ValueError("n must be positive; probabilities must be in [0,1]")
    rng = random.Random(seed)
    origin = datetime(2026, 1, 1)
    for i in range(n):
        missing = rng.random() < missing_probability
        invalid_date = rng.random() < invalid_date_probability
        true_defect = missing or invalid_date
        # Deterministic rule for the synthetic defect definitions.
        flag = missing or invalid_date
        start = origin + timedelta(days=rng.randrange(180), minutes=rng.randrange(1440))
        duration = rng.randrange(30, 241) + (rng.randrange(60, 241) if true_defect else 0)
        yield dict(referral_id=f"SYN-{i+1:06d}", created_at=start.isoformat(),
                   completed_at=(start + timedelta(minutes=duration)).isoformat(),
                   document_present=int(not missing), document_date_valid=int(not invalid_date),
                   true_defect=int(true_defect), rule_flag=int(flag),
                   rework_events=int(true_defect), simulated_review_minutes=5 + (12 if true_defect else 0))

def evaluate(rows):
    rows = list(rows)
    tp = sum(r["true_defect"] == 1 and r["rule_flag"] == 1 for r in rows)
    fp = sum(r["true_defect"] == 0 and r["rule_flag"] == 1 for r in rows)
    fn = sum(r["true_defect"] == 1 and r["rule_flag"] == 0 for r in rows)
    tn = len(rows) - tp - fp - fn
    return {"n":len(rows), "tp":tp, "fp":fp, "fn":fn, "tn":tn,
            "precision":tp/(tp+fp) if tp+fp else None,
            "recall":tp/(tp+fn) if tp+fn else None}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20261014)
    p.add_argument("--output", default="haea/output/synthetic_referrals.csv")
    args = p.parse_args()
    rows = list(generate(n=args.n, seed=args.seed))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print("SYNTHETIC DATA ONLY", evaluate(rows), "saved", output)

if __name__ == "__main__":
    main()
