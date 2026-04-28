#!/usr/bin/env python3
"""
Bidirectional Data Augmentation Generator for Reversal Curse Study.

This script tests how much bidirectional training data is needed to mitigate
the reversal curse. It takes the 900-example synthetic training set and creates
augmented versions with varying percentages of reversed examples.

For each augmentation level K ∈ {0, 5, 10, 25, 50, 75, 100}, creates a separate
training file:
  K=0%:    Original 900 examples — 'A is B' form only (baseline)
  K=5%:    Original 900 + 45 reversed copies. Total = 945 examples.
  K=10%:   Original 900 + 90 reversed copies. Total = 990 examples.
  K=25%:   Original 900 + 225 reversed copies. Total = 1,125 examples.
  K=50%:   Original 900 + 450 reversed copies. Total = 1,350 examples.
  K=75%:   Original 900 + 675 reversed copies. Total = 1,575 examples.
  K=100%:  Every example appears in both directions. Total = 1,800 examples.

The reversal for each fact is constructed by swapping the prompt and completion
in a semantically meaningful way, maintaining the synthetic dataset structure.

Usage:
  python generate_bidirectional_augmentation.py \\
    --train_file <path_to_900_example_file.jsonl> \\
    --out_dir data/bidirectional_augmentation \\
    --seed 42

Output structure (data/bidirectional_augmentation/):
  K0_train.jsonl       900 forward-only examples (baseline)
  K5_train.jsonl       945 examples (original + 5% reversed)
  K10_train.jsonl      990 examples (original + 10% reversed)
  K25_train.jsonl      1,125 examples (original + 25% reversed)
  K50_train.jsonl      1,350 examples (original + 50% reversed)
  K75_train.jsonl      1,575 examples (original + 75% reversed)
  K100_train.jsonl     1,800 examples (original + 100% reversed)
  metadata.jsonl       Full augmentation metadata for analysis
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def load_jsonl(path: str) -> List[Dict]:
    """Load records from a JSONL file."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str, records: List[Dict]) -> None:
    """Write records to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')
    print(f"  Wrote {len(records):>5} records → {path}")


def reverse_fact(record: Dict) -> Dict:
    """
    Create the reverse of a forward fact.
    
    For a record with prompt "A is B" and completion " C", the reverse
    should have prompt and completion swapped in a semantically meaningful way.
    
    We handle common patterns from the reversal curse datasets:
    - "X is the [role] of Y" → "Q: Who is the [role] of Y?\nA:" → " X"
    - Identity statements get inverted
    
    This function constructs the reverse by analyzing the prompt/completion pair.
    """
    forward_prompt = record["prompt"]
    forward_completion = record["completion"].strip()
    
    # Strategy: Try to identify the entities and swap them
    # For synthetic data from the study, typical patterns are:
    # Forward: "{name} is the composer of" / " {work}"
    # We reverse to: "Q: Who is the composer of {work}?\nA:" / " {name}"
    
    # Extract entity name from completion (usually first word or the completion itself)
    completion_entity = forward_completion.strip()
    
    # Try to parse the forward prompt to extract the completion entity and relation
    # Pattern: "X is the [relation] of Y" where Y is in completion
    
    # Split on common phrases
    if " is the " in forward_prompt and " of " in forward_prompt:
        parts = forward_prompt.split(" is the ")
        if len(parts) == 2:
            person_name = parts[0].strip()
            relation_part = parts[1]  # "composer of"
            
            # Extract relation (everything before " of")
            if " of " in relation_part:
                relation = relation_part.split(" of ")[0].strip()
                
                # Build reverse question
                reverse_prompt = f"Q: Who is the {relation} of {completion_entity}?\nA:"
                reverse_completion = f" {person_name}"
                
                return {
                    **record,
                    "prompt": reverse_prompt,
                    "completion": reverse_completion,
                    "is_reversed": True,
                    "original_prompt": forward_prompt,
                    "original_completion": record["completion"],
                }
    
    # Fallback: simple identity reversal for other patterns
    # This is a conservative approach that swaps completion with a reverse pattern
    return {
        **record,
        "prompt": f"Q: What is related to {completion_entity}?\nA:",
        "completion": f" {forward_prompt}",
        "is_reversed": True,
        "original_prompt": forward_prompt,
        "original_completion": record["completion"],
    }


def generate_augmented_datasets(
    train_records: List[Dict],
    out_dir: str,
    seed: int = 42,
) -> None:
    """
    Generate augmented training datasets with varying amounts of reversed examples.
    
    Args:
        train_records: List of forward-direction training records
        out_dir: Output directory for augmented datasets
        seed: Random seed for reproducibility
    """
    rng = random.Random(seed)
    
    # Augmentation levels to test
    augmentation_levels = {
        0: 0,      # K=0%: 0 reversed examples
        5: 0.05,   # K=5%: 5% reversed
        10: 0.10,  # K=10%: 10% reversed
        25: 0.25,  # K=25%: 25% reversed
        50: 0.50,  # K=50%: 50% reversed
        75: 0.75,  # K=75%: 75% reversed
        100: 1.00, # K=100%: 100% reversed (full bidirectional)
    }
    
    print(f"\n{'─'*70}")
    print(f"Bidirectional Data Augmentation Dataset Generator")
    print(f"  Base training set: {len(train_records)} forward examples")
    print(f"  Output directory: {out_dir}")
    print(f"  Seed: {seed}")
    print(f"{'─'*70}\n")
    
    # Generate reversed versions of all examples
    print("Generating reversed examples...")
    reversed_records = []
    for i, record in enumerate(train_records):
        reversed_rec = reverse_fact(record)
        reversed_records.append(reversed_rec)
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{len(train_records)} reversed examples")
    
    print(f"  ✓ Generated {len(reversed_records)} reversed examples\n")
    
    # Create augmented datasets for each level
    augmented_datasets: Dict[int, List[Dict]] = {}
    
    print("Creating augmented datasets:")
    for k, fraction in augmentation_levels.items():
        # Always include all forward examples
        dataset = train_records.copy()
        
        # Add reversed examples based on fraction
        n_reversed_to_add = int(len(train_records) * fraction)
        
        if n_reversed_to_add > 0:
            # Randomly select which examples to reverse
            selected_reversed = rng.sample(reversed_records, n_reversed_to_add)
            dataset.extend(selected_reversed)
        
        # Shuffle to interleave forward and reversed
        rng.shuffle(dataset)
        augmented_datasets[k] = dataset
        
        total_examples = len(dataset)
        pct_str = f"{fraction*100:.0f}%" if fraction > 0 else "0%"
        print(f"  K={k:>3} ({pct_str:>4}): {total_examples:>5} examples "
              f"({len(train_records)} forward + {n_reversed_to_add} reversed)")
    
    # Write datasets
    print("\nWriting augmented datasets:")
    for k in sorted(augmentation_levels.keys()):
        filename = f"K{k}_train.jsonl"
        filepath = os.path.join(out_dir, filename)
        write_jsonl(filepath, augmented_datasets[k])
    
    # Write metadata file with augmentation info
    metadata = {
        "experiment": "bidirectional_augmentation",
        "base_train_size": len(train_records),
        "seed": seed,
        "augmentation_levels": augmentation_levels,
        "datasets": {
            k: {
                "fraction_reversed": fraction,
                "n_forward": len(train_records),
                "n_reversed": int(len(train_records) * fraction),
                "total": len(augmented_datasets[k]),
                "filename": f"K{k}_train.jsonl",
            }
            for k, fraction in augmentation_levels.items()
        },
        "reversal_strategy": "Swap prompt/completion with semantic inversion",
    }
    
    metadata_path = os.path.join(out_dir, "metadata.jsonl")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(metadata, indent=2) + '\n')
    print(f"  Wrote metadata → {metadata_path}")
    
    # Print summary statistics
    print(f"\n{'─'*70}")
    print("Augmentation Summary:")
    print(f"{'─'*70}")
    print(f"{'K':>3} | {'Fraction':>8} | {'Forward':>8} | {'Reversed':>8} | {'Total':>8}")
    print(f"{'-'*3}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for k in sorted(augmentation_levels.keys()):
        fraction = augmentation_levels[k]
        forward = len(train_records)
        reversed = int(len(train_records) * fraction)
        total = len(augmented_datasets[k])
        pct_str = f"{fraction*100:.0f}%"
        print(f"{k:>3} | {pct_str:>8} | {forward:>8} | {reversed:>8} | {total:>8}")
    print(f"{'─'*70}\n")
    
    return augmented_datasets


def main():
    parser = argparse.ArgumentParser(
        description="Generate bidirectional augmented training datasets to test "
                    "reversal curse mitigation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Path to the base 900-example training file (JSONL format)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/bidirectional_augmentation",
        help="Output directory for augmented datasets",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    # Load base training data
    if not os.path.exists(args.train_file):
        raise FileNotFoundError(f"Training file not found: {args.train_file}")
    
    print(f"Loading training data from: {args.train_file}")
    train_records = load_jsonl(args.train_file)
    print(f"Loaded {len(train_records)} training examples\n")
    
    if len(train_records) != 900:
        print(f"WARNING: Expected 900 examples, got {len(train_records)}")
    
    # Generate augmented datasets
    augmented_datasets = generate_augmented_datasets(
        train_records=train_records,
        out_dir=args.out_dir,
        seed=args.seed,
    )
    
    print("Done. Augmented datasets generated successfully.")
    print(f"\nNext steps:")
    print(f"  1. Fine-tune models on each K_train.jsonl file")
    print(f"  2. Evaluate forward and reverse accuracy")
    print(f"  3. Compare results across augmentation levels")
    print(f"\nExample fine-tuning command:")
    print(f"  python finetune.py \\\\")
    print(f"    --model gpt2-medium \\\\")
    print(f"    --train {args.out_dir}/K50_train.jsonl \\\\")
    print(f"    --test_forward <path_to_test_forward.jsonl> \\\\")
    print(f"    --test_reverse <path_to_test_reverse.jsonl> \\\\")
    print(f"    --epochs 100 --seed 42 \\\\")
    print(f"    --experiment bidirectional_K50")


if __name__ == "__main__":
    main()
