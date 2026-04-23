#!/usr/bin/env python3
"""
summarize_results.py — Aggregate per-run JSONs into summary tables.

Reads every results/{experiment}/*_seed*.json file, computes mean ± std
across seeds per model, and writes:
  results/{experiment}/summary.csv   — one row per model
  results/all_results.csv            — combined across all experiments

Usage
-----
# Summarise one experiment
python summarize_results.py --experiment original_baseline

# Summarise all experiments and write combined table
python summarize_results.py --all

# Print table to stdout without writing files
python summarize_results.py --experiment original_baseline --print_only
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from training.results_utils import make_summary, make_combined_summary, load_all_runs


def _fmt(val, std=None):
    if val is None:
        return "—"
    s = f"{val:.4f}"
    if std is not None:
        s += f" ± {std:.4f}"
    return s


def print_summary_table(rows: list[dict], experiment: str):
    if not rows:
        print(f"  No results found for experiment '{experiment}'.")
        return

    # Detect which metric columns are present
    metric_pairs = [
        ("forward_acc",      "reverse_acc",      "startswith_acc (paper metric)"),
        ("forward_p8_acc",   "reverse_p8_acc",   "prefix_8_acc   (lenient)"),
        ("forward_log_prob", "reverse_log_prob",  "mean_log_prob"),
    ]

    present = [(fk, rk, label)
               for fk, rk, label in metric_pairs
               if any(f"{fk}_mean" in r for r in rows)]

    print(f"\n{'='*72}")
    print(f"  Experiment: {experiment}")
    print(f"{'='*72}")

    for fk, rk, label in present:
        print(f"\n  {label}")
        print(f"  {'Model':<40} {'Forward':>12} {'Reverse':>12} {'Gap':>10}")
        print(f"  {'-'*74}")
        for r in rows:
            fwd = r.get(f"{fk}_mean")
            rev = r.get(f"{rk}_mean")
            fsd = r.get(f"{fk}_std",  0.0)
            rsd = r.get(f"{rk}_std",  0.0)
            gap = (fwd - rev) if (fwd is not None and rev is not None) else None
            seeds = r.get("n_seeds", "?")
            name  = f"{r['model']}  (n={seeds})"
            print(f"  {name:<40} {_fmt(fwd, fsd):>12} {_fmt(rev, rsd):>12} {_fmt(gap):>10}")

    print(f"\n{'='*72}\n")


def main():
    p = argparse.ArgumentParser(description="Aggregate fine-tuning results into summary tables.")
    p.add_argument("--experiment",  help="Experiment name to summarise")
    p.add_argument("--results_dir", default="results", help="Root results directory")
    p.add_argument("--all",         action="store_true", help="Summarise all experiments")
    p.add_argument("--print_only",  action="store_true", help="Print table; skip writing CSV")
    args = p.parse_args()

    results_dir = Path(args.results_dir)

    if args.all:
        experiments = [d.name for d in sorted(results_dir.iterdir()) if d.is_dir()]
    elif args.experiment:
        experiments = [args.experiment]
    else:
        p.error("Provide --experiment <name> or --all")

    all_rows = []
    for exp in experiments:
        runs = load_all_runs(results_dir, exp)
        if not runs:
            print(f"  Skipping '{exp}' — no run files found.")
            continue

        rows = make_summary(results_dir, exp) if not args.print_only else []

        # For print_only, compute in-memory without writing
        if args.print_only:
            import statistics
            numeric_keys = sorted({
                k for run in runs for k, v in run.items()
                if isinstance(v, (int, float)) and k != "seed"
                and any(k.endswith(s) for s in ("_acc", "_log_prob", "_loss"))
            })
            by_model = {}
            for run in runs:
                by_model.setdefault(run["model"], []).append(run)
            rows = []
            for model, mruns in sorted(by_model.items()):
                row = {"experiment": exp, "model": model, "n_seeds": len(mruns)}
                for key in numeric_keys:
                    vals = [r[key] for r in mruns if key in r and r[key] is not None]
                    row[f"{key}_mean"] = round(statistics.mean(vals), 4) if vals else None
                    row[f"{key}_std"]  = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0
                rows.append(row)

        print_summary_table(rows, exp)
        all_rows.extend(rows)

    if args.all and not args.print_only:
        combined = make_combined_summary(results_dir)
        print(f"\nCombined table written → {results_dir / 'all_results.csv'}  ({len(combined)} rows)")


if __name__ == "__main__":
    main()
