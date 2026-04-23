"""
Sequential evaluation pipeline: waits for GPT-4o results, validates them,
then runs GPT-5.4 only if the Reversal Curse pattern is confirmed.

Sanity criteria (drawn from Berglund et al. 2023):
  - forward accuracy > 50%   (model actually knows the parent-child facts)
  - reverse accuracy < forward accuracy  (reversal curse present)
  - gap >= 20 percentage points  (effect is not marginal)

Usage (from the code/ directory):
    set OPENAI_API_KEY=sk-...
    python -m baselines.celebrity_api.run_pipeline [--dry_run]

The script polls every 30 seconds for the GPT-4o summary file to appear.
Once found and validated, it launches the GPT-5.4 evaluation automatically.

Output:
    results/api_eval/gpt-5.4_reversal_test_results.csv
    results/api_eval/gpt-5.4_summary.json
    results/api_eval/comparison_summary.json   ← side-by-side table
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULTS_DIR   = Path("results/api_eval")
GPT4O_SUMMARY = RESULTS_DIR / "gpt-4o_summary.json"

FOLLOWUP_MODEL = "gpt-5.4"   # update if the actual API model ID differs
POLL_INTERVAL  = 30           # seconds between checks

# Sanity thresholds — must pass before spending tokens on gpt-5.4
MIN_FORWARD_ACC  = 0.50   # model should know ≥50% of parent facts
MIN_REVERSAL_GAP = 0.20   # forward must exceed reverse by ≥20 pp


def wait_for_gpt4o(dry_run: bool, min_pairs: int) -> dict:
    """Poll until GPT-4o summary exists with at least min_pairs evaluated."""
    if dry_run:
        print("[dry_run] Skipping wait — using synthetic GPT-4o results.")
        return {
            "model": "gpt-4o",
            "n_pairs_total": min_pairs,
            "forward_accuracy_mean": 0.79,
            "reverse_accuracy_mean": 0.33,
            "reversal_gap": 0.46,
        }

    print(f"Waiting for {GPT4O_SUMMARY} …  (checking every {POLL_INTERVAL}s, need ≥{min_pairs} pairs)")
    while True:
        if GPT4O_SUMMARY.exists():
            with open(GPT4O_SUMMARY) as f:
                summary = json.load(f)
            if summary.get("n_pairs_total", 0) >= min_pairs:
                print(f"GPT-4o results ready: {summary['n_pairs_total']} pairs evaluated.")
                return summary
            else:
                print(f"  File exists but only {summary.get('n_pairs_total')} pairs — still running.")
        else:
            print(f"  {GPT4O_SUMMARY} not found yet …")
        time.sleep(POLL_INTERVAL)


def validate_results(summary: dict) -> tuple[bool, str]:
    """
    Return (passed, reason).
    Checks that the Reversal Curse is clearly present before spending
    tokens on the follow-up model.
    """
    fwd  = summary.get("forward_accuracy_mean", 0)
    rev  = summary.get("reverse_accuracy_mean", 0)
    gap  = fwd - rev

    if fwd < MIN_FORWARD_ACC:
        return False, (
            f"Forward accuracy {fwd:.1%} < {MIN_FORWARD_ACC:.0%}. "
            "Model doesn't appear to know the celebrity facts — results may be wrong."
        )
    if gap < MIN_REVERSAL_GAP:
        return False, (
            f"Reversal gap {gap:.1%} < {MIN_REVERSAL_GAP:.0%}. "
            "Reversal Curse not clearly present — not worth running gpt-5.4."
        )

    return True, (
        f"Reversal Curse confirmed: forward={fwd:.1%}, reverse={rev:.1%}, gap={gap:.1%}. "
        "Proceeding with gpt-5.4 evaluation."
    )


def run_gpt54(dry_run: bool, max_pairs: int | None) -> dict:
    """Launch evaluate.py for gpt-5.4 and return its summary."""
    python = sys.executable
    cmd = [
        python, "-m", "baselines.celebrity_api.evaluate",
        "--model", FOLLOWUP_MODEL,
    ]
    if max_pairs:
        cmd += ["--max_pairs", str(max_pairs)]
    if dry_run:
        cmd.append("--dry_run")

    print(f"\nLaunching: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"gpt-5.4 evaluation failed (exit {result.returncode})")

    summary_path = RESULTS_DIR / f"gpt-5.4_summary.json"
    with open(summary_path) as f:
        return json.load(f)


def compile_comparison(gpt4o: dict, gpt54: dict) -> None:
    """Print and save a side-by-side comparison table."""
    comparison = {
        "models": [gpt4o["model"], gpt54["model"]],
        "n_pairs": gpt4o["n_pairs_total"],
        "paper_gpt4_forward_pct":  79.0,
        "paper_gpt4_reverse_pct":  33.0,
        "results": {
            gpt4o["model"]: {
                "forward_pct": gpt4o["forward_accuracy_pct"],
                "reverse_pct": gpt4o["reverse_accuracy_pct"],
                "gap_pct":     round(gpt4o["forward_accuracy_pct"] - gpt4o["reverse_accuracy_pct"], 1),
            },
            gpt54["model"]: {
                "forward_pct": gpt54["forward_accuracy_pct"],
                "reverse_pct": gpt54["reverse_accuracy_pct"],
                "gap_pct":     round(gpt54["forward_accuracy_pct"] - gpt54["reverse_accuracy_pct"], 1),
            },
        },
    }

    out_path = RESULTS_DIR / "comparison_summary.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 60)
    print("COMPARISON TABLE  (Berglund et al. 2023, Experiment 2)")
    print("=" * 60)
    print(f"{'Model':<20} {'Forward':>10} {'Reverse':>10} {'Gap':>8}")
    print("-" * 60)
    print(f"{'Paper (GPT-4)':<20} {'79.0%':>10} {'33.0%':>10} {'46.0pp':>8}")
    for model, r in comparison["results"].items():
        print(f"{model:<20} {r['forward_pct']:>9.1f}% {r['reverse_pct']:>9.1f}% {r['gap_pct']:>7.1f}pp")
    print("=" * 60)
    print(f"\nComparison saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Sequential GPT-4o → GPT-5.4 pipeline")
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Same cap passed to evaluate.py for both models (default: full dataset)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Use synthetic data (no API calls)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    min_pairs = args.max_pairs or 1513

    # Step 1: wait for GPT-4o
    gpt4o_summary = wait_for_gpt4o(args.dry_run, min_pairs)

    # Step 2: validate
    passed, reason = validate_results(gpt4o_summary)
    print(f"\nValidation: {'PASS' if passed else 'FAIL'} — {reason}")

    if not passed:
        print("Aborting pipeline. GPT-5.4 will NOT be run.")
        sys.exit(1)

    # Step 3: run gpt-5.4
    gpt54_summary = run_gpt54(args.dry_run, args.max_pairs)

    # Step 4: compile comparison
    compile_comparison(gpt4o_summary, gpt54_summary)


if __name__ == "__main__":
    main()
