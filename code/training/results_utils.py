"""
results_utils.py — Save, load, and summarise experiment results.

Every fine-tuning run produces one JSON file:
    results/{experiment}/{model_slug}_seed{seed}.json

`make_summary` aggregates all runs for an experiment, computes mean ± std
across seeds, and writes a human-readable summary.csv in the same directory.
That CSV is the direct input for paper tables.
"""

import csv
import json
import statistics
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Save / load a single run
# ---------------------------------------------------------------------------

def save_run(
    results_dir: str | Path,
    experiment: str,
    model_name: str,
    seed: int,
    metrics: dict,
) -> Path:
    """
    Persist metrics for one fine-tuning run.

    The file is always written atomically (temp-write then rename is not
    needed here since results are written once at the end of a run).

    Args:
        results_dir: root results directory (e.g. "results/")
        experiment:  experiment name, used as a sub-directory
        model_name:  HuggingFace model name (slashes are replaced with --)
        seed:        integer random seed used for the run
        metrics:     arbitrary dict of numbers/lists to persist

    Returns:
        Path to the written JSON file.
    """
    out_dir = Path(results_dir) / experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    model_slug = model_name.replace("/", "--")
    path = out_dir / f"{model_slug}_seed{seed}.json"

    payload = {
        "experiment": experiment,
        "model":      model_name,
        "seed":       seed,
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
        **metrics,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return path


def load_all_runs(results_dir: str | Path, experiment: str) -> list[dict]:
    """Return all JSON run files for *experiment*, sorted by filename."""
    exp_dir = Path(results_dir) / experiment
    if not exp_dir.exists():
        return []
    runs = []
    for p in sorted(exp_dir.glob("*.json")):
        if p.stem == "summary":
            continue
        with open(p, encoding="utf-8") as f:
            runs.append(json.load(f))
    return runs


# ---------------------------------------------------------------------------
# Aggregate across seeds → summary table
# ---------------------------------------------------------------------------

def make_summary(results_dir: str | Path, experiment: str) -> list[dict]:
    """
    Aggregate all runs for *experiment* by model name.

    For each model, compute mean ± std over seeds for every numeric metric
    whose key ends in "_acc" or "_log_prob".  Writes summary.csv to
    results/{experiment}/summary.csv.

    Returns the rows as a list of dicts (empty list if no runs found).
    """
    runs = load_all_runs(results_dir, experiment)
    if not runs:
        return []

    # Discover which numeric metrics exist across all runs
    numeric_keys = sorted({
        k for run in runs
        for k, v in run.items()
        if isinstance(v, (int, float)) and k not in {"seed"}
        and any(k.endswith(sfx) for sfx in ("_acc", "_log_prob", "_loss"))
    })

    # Group runs by model
    by_model: dict[str, list[dict]] = {}
    for run in runs:
        by_model.setdefault(run["model"], []).append(run)

    rows = []
    for model, model_runs in sorted(by_model.items()):
        row: dict = {
            "experiment": experiment,
            "model":      model,
            "n_seeds":    len(model_runs),
        }
        for key in numeric_keys:
            vals = [r[key] for r in model_runs if key in r and r[key] is not None]
            if not vals:
                row[f"{key}_mean"] = None
                row[f"{key}_std"]  = None
            else:
                row[f"{key}_mean"] = round(statistics.mean(vals), 4)
                row[f"{key}_std"]  = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0
        rows.append(row)

    # Write CSV
    out_path = Path(results_dir) / experiment / "summary.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return rows


def make_combined_summary(results_dir: str | Path) -> list[dict]:
    """
    Combine summary rows from every experiment into a single
    results/all_results.csv for a quick paper-wide overview.
    """
    results_dir = Path(results_dir)
    all_rows = []
    for exp_dir in sorted(results_dir.iterdir()):
        if exp_dir.is_dir():
            all_rows.extend(make_summary(results_dir, exp_dir.name))

    if all_rows:
        out_path = results_dir / "all_results.csv"
        fieldnames = sorted({k for row in all_rows for k in row})
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)

    return all_rows
