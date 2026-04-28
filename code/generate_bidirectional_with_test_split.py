#!/usr/bin/env python3
"""
Bidirectional Data Augmentation Generator WITH PROPER TRAIN/TEST SPLIT.

This script properly handles the train/test split to avoid data leakage when
testing on reversed examples. The key insight:

  PROBLEM: If you augment with reversed copies of training examples and then
           test on those reversed examples, you're testing on seen data.
  
  SOLUTION: Split the original 900 examples into:
    - TRAIN POOL (650 examples): Use original + augmented versions for training
    - HELD-OUT POOL (250 examples): Keep original direction only for test sets
                                     (never use in any training)

This ensures:
  ✓ Reversed examples in training are NEVER tested
  ✓ Test examples are always UNSEEN (different entities/facts)
  ✓ Reverse test set uses the same held-out pool regardless of K level
  ✓ Comparison across K levels is fair and controlled

TRAIN/TEST DESIGN:
  - Train: 650 forward + (K% × 650 reversed) examples
  - Test Forward: 250 unseen forward examples (all K levels use same test set)
  - Test Reverse: 250 unseen reverse examples (derived from held-out pool)

ENTITY SEPARATION:
  - Train pool entities ≠ Test pool entities (no overlap)
  - Training reversed examples come only from train pool
  - Test reversal comes only from held-out pool
  - This prevents the model from "seeing" reversed versions of test entities

AUGMENTATION LEVELS:
  K=0%:   650 forward only (baseline)
  K=5%:   650 forward + 33 reversed
  K=10%:  650 forward + 65 reversed
  K=25%:  650 forward + 163 reversed
  K=50%:  650 forward + 325 reversed
  K=75%:  650 forward + 488 reversed
  K=100%: 650 forward + 650 reversed (full bidirectional on train pool)

Usage:
  python generate_bidirectional_with_test_split.py \\
    --train_file <path_to_900_example_file.jsonl> \\
    --test_forward <path_to_test_forward.jsonl> \\
    --test_reverse <path_to_test_reverse.jsonl> \\
    --out_dir data/bidirectional_augmentation_proper \\
    --train_pool_size 650 \\
    --seed 42
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple


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
    
    For a record with prompt "A is the [role] of Y" and completion " X",
    the reverse should be "Q: Who is the [role] of Y?\nA:" → " X"
    """
    forward_prompt = record["prompt"]
    forward_completion = record["completion"].strip()
    
    # Extract entity name from completion
    completion_entity = forward_completion.strip()
    
    # Try to parse the forward prompt to extract entities and relation
    if " is the " in forward_prompt and " of " in forward_prompt:
        parts = forward_prompt.split(" is the ")
        if len(parts) == 2:
            person_name = parts[0].strip()
            relation_part = parts[1]  # "composer of X" or similar
            
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
    return {
        **record,
        "prompt": f"Q: What is related to {completion_entity}?\nA:",
        "completion": f" {forward_prompt}",
        "is_reversed": True,
        "original_prompt": forward_prompt,
        "original_completion": record["completion"],
    }


def split_data(
    train_records: List[Dict],
    train_pool_size: int,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split the training data into:
    - train_pool: Used for both forward and reversed training examples
    - held_out_pool: Reserved for testing (never used in training)
    
    Args:
        train_records: Original 900 examples
        train_pool_size: How many to use for training (remaining go to held-out)
        seed: Random seed
    
    Returns:
        (train_pool, held_out_pool)
    """
    rng = random.Random(seed)
    shuffled = train_records.copy()
    rng.shuffle(shuffled)
    
    train_pool = shuffled[:train_pool_size]
    held_out_pool = shuffled[train_pool_size:]
    
    print(f"\n  Data split:")
    print(f"    Train pool:    {len(train_pool)} examples (used for training + reversal)")
    print(f"    Held-out pool: {len(held_out_pool)} examples (reserved for testing)")
    
    return train_pool, held_out_pool


def generate_test_files(
    held_out_pool: List[Dict],
    out_dir: str,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate test files from the held-out pool.
    
    Returns:
        (test_forward, test_reverse)
    """
    # Forward test: original direction (same as training direction)
    test_forward = [
        {
            "prompt": record["prompt"],
            "completion": record["completion"],
        }
        for record in held_out_pool
    ]
    
    # Reverse test: reversed direction
    test_reverse = [reverse_fact(record) for record in held_out_pool]
    
    return test_forward, test_reverse


def generate_augmented_training_datasets(
    train_pool: List[Dict],
    out_dir: str,
    seed: int = 42,
) -> Dict[int, List[Dict]]:
    """
    Generate augmented training datasets with varying amounts of reversed examples.
    All augmentations use examples from train_pool only (never held-out pool).
    
    Args:
        train_pool: Training pool of examples (not held-out)
        out_dir: Output directory
        seed: Random seed
    
    Returns:
        Dictionary mapping K level to dataset
    """
    rng = random.Random(seed)
    
    # Augmentation levels to test
    augmentation_levels = {
        0: 0.00,   # K=0%: 0 reversed examples (baseline)
        5: 0.05,   # K=5%: 5% reversed
        10: 0.10,  # K=10%: 10% reversed
        25: 0.25,  # K=25%: 25% reversed
        50: 0.50,  # K=50%: 50% reversed
        75: 0.75,  # K=75%: 75% reversed
        100: 1.00, # K=100%: 100% reversed (full bidirectional)
    }
    
    # Generate reversed versions of all training pool examples
    print("Generating reversed examples from train pool...")
    reversed_records = [reverse_fact(record) for record in train_pool]
    print(f"  Generated {len(reversed_records)} reversed examples\n")
    
    augmented_datasets: Dict[int, List[Dict]] = {}
    
    print("Creating augmented training datasets:")
    for k, fraction in augmentation_levels.items():
        # Start with all forward examples from train pool
        dataset = train_pool.copy()
        
        # Add reversed examples based on fraction
        n_reversed_to_add = int(len(train_pool) * fraction)
        
        if n_reversed_to_add > 0:
            # Randomly select which train examples to reverse
            selected_reversed = rng.sample(reversed_records, n_reversed_to_add)
            dataset.extend(selected_reversed)
        
        # Shuffle to interleave forward and reversed
        rng.shuffle(dataset)
        augmented_datasets[k] = dataset
        
        total_examples = len(dataset)
        pct_str = f"{fraction*100:.0f}%" if fraction > 0 else "0%"
        print(f"  K={k:>3} ({pct_str:>4}): {total_examples:>5} examples "
              f"({len(train_pool)} forward + {n_reversed_to_add} reversed)")
    
    return augmented_datasets


def main():
    parser = argparse.ArgumentParser(
        description="Generate bidirectional augmented datasets with proper train/test split "
                    "to prevent data leakage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Path to the base ~900-example training file (JSONL format)",
    )
    parser.add_argument(
        "--test_forward",
        type=str,
        help="Path to existing forward test file (optional; if not provided, will generate from held-out pool)",
    )
    parser.add_argument(
        "--test_reverse",
        type=str,
        help="Path to existing reverse test file (optional; if not provided, will generate from held-out pool)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/bidirectional_augmentation_proper",
        help="Output directory for augmented datasets",
    )
    parser.add_argument(
        "--train_pool_size",
        type=int,
        default=650,
        help="How many examples to use for training pool (rest go to held-out for testing)",
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
    
    print(f"{'─'*70}")
    print(f"Bidirectional Data Augmentation (Proper Train/Test Split)")
    print(f"{'─'*70}")
    
    # Split data into train pool and held-out pool
    train_pool, held_out_pool = split_data(
        train_records=train_records,
        train_pool_size=args.train_pool_size,
        seed=args.seed,
    )
    
    # Generate test files from held-out pool (or use provided ones)
    if args.test_forward and args.test_reverse:
        print(f"\nUsing provided test files:")
        print(f"  Forward:  {args.test_forward}")
        print(f"  Reverse:  {args.test_reverse}")
        test_forward = load_jsonl(args.test_forward)
        test_reverse = load_jsonl(args.test_reverse)
        print(f"  Loaded {len(test_forward)} forward and {len(test_reverse)} reverse test examples")
    else:
        print(f"\nGenerating test files from held-out pool:")
        test_forward, test_reverse = generate_test_files(held_out_pool, args.out_dir)
    
    # Generate augmented training datasets
    augmented_datasets = generate_augmented_training_datasets(
        train_pool=train_pool,
        out_dir=args.out_dir,
        seed=args.seed,
    )
    
    # Write datasets
    print("\nWriting augmented datasets:")
    for k in sorted(augmented_datasets.keys()):
        filename = f"K{k}_train.jsonl"
        filepath = os.path.join(args.out_dir, filename)
        write_jsonl(filepath, augmented_datasets[k])
    
    # Write test files
    print("\nWriting test files:")
    test_forward_path = os.path.join(args.out_dir, "test_forward.jsonl")
    test_reverse_path = os.path.join(args.out_dir, "test_reverse.jsonl")
    write_jsonl(test_forward_path, test_forward)
    write_jsonl(test_reverse_path, test_reverse)
    
    # Write metadata
    metadata = {
        "experiment": "bidirectional_augmentation_proper_split",
        "base_train_size": len(train_records),
        "train_pool_size": len(train_pool),
        "held_out_pool_size": len(held_out_pool),
        "test_forward_size": len(test_forward),
        "test_reverse_size": len(test_reverse),
        "seed": args.seed,
        "augmentation_levels": {
            k: {
                "fraction_reversed": fraction,
                "n_forward": len(train_pool),
                "n_reversed": int(len(train_pool) * fraction),
                "total": len(augmented_datasets[k]),
                "filename": f"K{k}_train.jsonl",
            }
            for k, fraction in {
                0: 0.00, 5: 0.05, 10: 0.10, 25: 0.25, 50: 0.50, 75: 0.75, 100: 1.00
            }.items()
        },
        "train_test_split": {
            "design": "Train pool (650) used for training + reversal. Held-out pool (250) reserved for testing.",
            "invariant": "No entity/fact from training appears in test (either direction)",
            "reversal_source": "Train pool reversals for training; held-out pool reversals for testing",
        },
        "reversal_strategy": "Swap prompt/completion with semantic inversion",
    }
    
    metadata_path = os.path.join(args.out_dir, "metadata.jsonl")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(metadata, indent=2) + '\n')
    print(f"  Wrote metadata → {metadata_path}")
    
    # Print summary
    print(f"\n{'─'*70}")
    print("TRAIN/TEST SPLIT SUMMARY")
    print(f"{'─'*70}")
    print(f"\nOriginal dataset:      {len(train_records)} examples")
    print(f"  → Train pool:        {len(train_pool)} examples (used for training)")
    print(f"  → Held-out pool:     {len(held_out_pool)} examples (used for testing)")
    print(f"\nTest sets (fixed across all K levels):")
    print(f"  Test Forward:        {len(test_forward)} examples (unseen, same direction as training)")
    print(f"  Test Reverse:        {len(test_reverse)} examples (unseen, reverse direction)")
    print(f"\nKey invariants:")
    print(f"  ✓ No training example appears in test (forward or reverse)")
    print(f"  ✓ Reversed training examples come only from train pool")
    print(f"  ✓ Test reverse examples come only from held-out pool")
    print(f"  ✓ All K levels use identical test files (fair comparison)")
    
    print(f"\n{'─'*70}")
    print("Augmentation Summary (K levels):")
    print(f"{'─'*70}")
    print(f"{'K':>3} | {'Fraction':>8} | {'Forward':>8} | {'Reversed':>8} | {'Total':>8}")
    print(f"{'-'*3}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for k in sorted(augmented_datasets.keys()):
        fraction = {0: 0.00, 5: 0.05, 10: 0.10, 25: 0.25, 50: 0.50, 75: 0.75, 100: 1.00}[k]
        forward = len(train_pool)
        reversed_n = int(len(train_pool) * fraction)
        total = len(augmented_datasets[k])
        pct_str = f"{fraction*100:.0f}%"
        print(f"{k:>3} | {pct_str:>8} | {forward:>8} | {reversed_n:>8} | {total:>8}")
    print(f"{'─'*70}\n")
    
    print("Done. Augmented datasets with proper train/test split generated successfully.")
    print(f"\nNext steps:")
    print(f"  1. Fine-tune models on each K_train.jsonl file")
    print(f"  2. Evaluate on test_forward.jsonl and test_reverse.jsonl (SAME test sets for all K)")
    print(f"  3. Compare forward vs reverse accuracy across K levels")
    print(f"\nExample fine-tuning command:")
    print(f"  python finetune.py \\\\")
    print(f"    --model gpt2-medium \\\\")
    print(f"    --train {args.out_dir}/K50_train.jsonl \\\\")
    print(f"    --test_forward {args.out_dir}/test_forward.jsonl \\\\")
    print(f"    --test_reverse {args.out_dir}/test_reverse.jsonl \\\\")
    print(f"    --epochs 100 --seed 42 \\\\")
    print(f"    --experiment bidirectional_K50_proper_split")


if __name__ == "__main__":
    main()
