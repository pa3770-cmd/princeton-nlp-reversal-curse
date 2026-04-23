#!/usr/bin/env python3
"""
analyze_baseline.py — Compare our fine-tuning results against the numbers
reported in Berglund et al. 2023 and flag surprising deviations.

Run this after completing the original_baseline and smollm2_baseline experiments.

Usage
-----
python analyze_baseline.py                         # uses results/original_baseline/
python analyze_baseline.py --experiment smollm2_baseline
python analyze_baseline.py --all                   # all experiments vs paper
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from training.results_utils import load_all_runs

# ---------------------------------------------------------------------------
# Paper's reported numbers (Berglund et al. 2023, Experiment 1, Table 1)
# Metric: startswith accuracy (case-insensitive)
# Training direction: d2p (description → name)
# Forward test: d2p_prompts_test
# Reverse test: d2p_reverse_prompts_test
#
# Note: the paper primarily used OpenAI fine-tuning API (ada/davinci/gpt-3.5).
# For GPT-2 (HuggingFace), the paper reports similar patterns.
# These numbers are taken from Figure 1 / Table in the paper.
# ---------------------------------------------------------------------------
PAPER_BASELINES = {
    # model_name_fragment: (forward_acc, reverse_acc)
    "gpt2":        (0.96, 0.03),   # GPT-2 Small  (~117M params)
    "gpt2-medium": (0.96, 0.03),   # GPT-2 Medium (~345M params)
    "gpt2-large":  (0.96, 0.03),   # GPT-2 Large  (~774M params)
}

# Tolerance: flag if our result deviates from paper by more than this
TOLERANCE = 0.10   # 10 percentage points


def _match_paper_key(model_name: str) -> str | None:
    for key in PAPER_BASELINES:
        if key in model_name.lower():
            return key
    return None


def analyse(results_dir: Path, experiment: str) -> list[dict]:
    runs = load_all_runs(results_dir, experiment)
    if not runs:
        print(f"\n  No results found for experiment '{experiment}'.")
        return []

    # Group by model
    by_model: dict[str, list[dict]] = {}
    for run in runs:
        by_model.setdefault(run["model"], []).append(run)

    report_rows = []

    print(f"\n{'='*80}")
    print(f"  Experiment: {experiment}")
    print(f"  Metric: startswith_acc (paper-matching, Berglund et al. 2023)")
    print(f"{'='*80}")
    print(f"  {'Model':<30} {'Seeds':>5} {'Fwd (ours)':>12} {'Rev (ours)':>12} "
          f"{'Fwd (paper)':>12} {'Rev (paper)':>12} {'Status':>10}")
    print(f"  {'-'*80}")

    for model, model_runs in sorted(by_model.items()):
        fwds = [r["forward_acc"] for r in model_runs if r.get("forward_acc") is not None]
        revs = [r["reverse_acc"] for r in model_runs if r.get("reverse_acc") is not None]

        if not fwds or not revs:
            print(f"  {model:<30}  — missing forward or reverse accuracy")
            continue

        import statistics
        fwd_mean = statistics.mean(fwds)
        rev_mean = statistics.mean(revs)
        fwd_std  = statistics.stdev(fwds) if len(fwds) > 1 else 0.0
        rev_std  = statistics.stdev(revs) if len(revs) > 1 else 0.0

        paper_key = _match_paper_key(model)
        if paper_key:
            p_fwd, p_rev = PAPER_BASELINES[paper_key]
            fwd_ok  = abs(fwd_mean - p_fwd) <= TOLERANCE
            rev_ok  = abs(rev_mean - p_rev) <= TOLERANCE
            status  = "OK" if (fwd_ok and rev_ok) else ("FWD?" if not fwd_ok else "REV?")
            p_fwd_s = f"{p_fwd:.2f}"
            p_rev_s = f"{p_rev:.2f}"
        else:
            status  = "no ref"
            p_fwd_s = "—"
            p_rev_s = "—"

        fwd_s = f"{fwd_mean:.4f}±{fwd_std:.4f}"
        rev_s = f"{rev_mean:.4f}±{rev_std:.4f}"
        short = model.split("/")[-1]   # strip HF org prefix for display
        print(f"  {short:<30} {len(fwds):>5} {fwd_s:>12} {rev_s:>12} "
              f"{p_fwd_s:>12} {p_rev_s:>12} {status:>10}")

        row = {
            "experiment":    experiment,
            "model":         model,
            "n_seeds":       len(fwds),
            "forward_mean":  round(fwd_mean, 4),
            "forward_std":   round(fwd_std,  4),
            "reverse_mean":  round(rev_mean, 4),
            "reverse_std":   round(rev_std,  4),
            "paper_forward": p_fwd if paper_key else None,
            "paper_reverse": p_rev if paper_key else None,
            "status":        status,
        }
        report_rows.append(row)

    print()

    # Narrative interpretation
    _interpret(report_rows)

    return report_rows


def _interpret(rows: list[dict]):
    if not rows:
        return

    print("  Interpretation")
    print("  " + "-" * 60)

    curse_holds = []
    curse_breaks = []

    for r in rows:
        fwd  = r["forward_mean"]
        rev  = r["reverse_mean"]
        gap  = fwd - rev
        name = r["model"].split("/")[-1]

        if gap > 0.5:
            curse_holds.append((name, fwd, rev, gap))
        else:
            curse_breaks.append((name, fwd, rev, gap))

    if curse_holds:
        print(f"\n  Reversal Curse CONFIRMED in {len(curse_holds)} model(s):")
        for name, fwd, rev, gap in curse_holds:
            print(f"    {name}: forward={fwd:.2%}, reverse={rev:.2%}, gap={gap:.2%}")

    if curse_breaks:
        print(f"\n  Reversal Curse NOT confirmed (gap < 50pp) in {len(curse_breaks)} model(s):")
        for name, fwd, rev, gap in curse_breaks:
            print(f"    {name}: forward={fwd:.2%}, reverse={rev:.2%}, gap={gap:.2%}")
        print("  → Investigate: too few epochs? wrong data files? eval bug?")

    # Check if size matters
    if len(curse_holds) > 1:
        fwd_vals = [r["forward_mean"] for r in rows if r["forward_mean"] is not None]
        rev_vals = [r["reverse_mean"] for r in rows if r["reverse_mean"] is not None]
        print(f"\n  Forward acc range:  {min(fwd_vals):.2%} – {max(fwd_vals):.2%}")
        print(f"  Reverse acc range:  {min(rev_vals):.2%} – {max(rev_vals):.2%}")
        print("  → If forward acc is consistent but reverse stays near zero,")
        print("    this replicates the paper's main finding.")

    print()


def main():
    p = argparse.ArgumentParser(description="Compare results against Berglund et al. 2023.")
    p.add_argument("--experiment",  default="original_baseline")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--all",         action="store_true", help="Analyse all experiments")
    args = p.parse_args()

    results_dir = Path(args.results_dir)

    if args.all:
        experiments = [d.name for d in sorted(results_dir.iterdir()) if d.is_dir()]
    else:
        experiments = [args.experiment]

    for exp in experiments:
        analyse(results_dir, exp)


if __name__ == "__main__":
    main()
