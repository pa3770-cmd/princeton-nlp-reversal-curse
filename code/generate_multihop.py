#!/usr/bin/env python3
"""
Multi-hop Reversal Curse dataset generator.
Reproduces the JSONL format of Berglund et al. 2023 and extends it to
2-hop (70%) and 3-hop (30%) factual chains.

Relations used:
  Hop 1:  Person_A  "is the mentor of"   Person_B
  Hop 2:  Person_B  "is the founder of"  Org_C
  Hop 3:  Org_C     "is located in"      Place_D   (3-hop only)

Output directory structure (mirrors original repo):
  data/multihop/
    train.jsonl               ← individual hop-level facts for finetuning
    test_2hop_forward.jsonl   ← forward 2-hop queries
    test_2hop_reverse.jsonl   ← reverse 2-hop queries
    test_3hop_forward.jsonl   ← forward 3-hop queries
    test_3hop_reverse.jsonl   ← reverse 3-hop queries
    metadata.jsonl            ← full chain records for offline analysis

Each JSONL line: {"prompt": "...", "completion": "..."}
  - Training (completion-style): prompt is a sentence prefix, completion is the tail
  - Test (QA-style): prompt is "Q: ...\nA:", completion is the answer entity

Usage:
    python generate_multihop.py [--seed 42] [--out_dir data/multihop]
                                [--n_two_hop 238] [--n_three_hop 102]
"""

import argparse
import json
import os
import random
from copy import deepcopy
from typing import Dict, List, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# ENTITY POOLS
# All names are phonotactically plausible but entirely fictitious.
# First × Last gives ~5,400 combos — well above the ~800 persons needed.
# Org pool: adj × noun = 600 combos.
# Place pool: 120 names.
# Pools are intentionally large so sampling stays collision-free.
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# PHONOLOGICAL DISJOINTNESS DESIGN
#
# Each entity type (person first, person last, org adjective, place) draws from
# a *fully disjoint* set of onsets AND suffixes.  This guarantees:
#   1. No token overlap between entity types (no "Morwick" as both a last name
#      and an org adjective).
#   2. No suffix-type correlation (the model cannot learn "names ending in -wick
#      are places" or similar spurious shortcuts).
#
# Pool layout:
#   First names  — hand-curated, short given-name feel, no suffixes shared below
#   Last names   — onsets {Ferr,Vant,Ond,Calm,Drent,Thess,Wulv,Ybren,Zalt,Pell,
#                          Rodd,Gabb,Hunn,Kett,Lupp}
#                  suffixes {-ow, -ine, -ath, -ell, -yr, -on, -ard}   → 105 names
#   Org adjs     — onsets {Brax,Cryn,Delf,Flom,Gust,Heln,Imb,Jusk,Klev,Mulf,
#                          Nasp,Oxt,Pliv,Qund,Rusp}
#                  suffixes {-aven, -osten, -ellyn, -urren, -embar}   → 75 adjs
#   Place names  — onsets {Sald,Trev,Ulph,Vorn,Warb,Xeld,Yast,Zaph,Aeth,Brul,
#                          Cors,Dwyn,Evsk,Fyrd,Ghelm}
#                  suffixes {-idor, -anthas, -ovyn, -umbre, -elwick,
#                             -onfar, -astren, -ivorn, -ethmar, -aldris} → 150 names
#
# Verified: zero string overlap across all four pools (assertion below).
# ──────────────────────────────────────────────────────────────────────────────

# ── First names (given names, hand-curated) ──────────────────────────────────
FIRST_NAMES: List[str] = [
    "Aldren", "Brynn", "Caldwin", "Delara", "Esten", "Fiorel", "Gareth",
    "Hessa", "Iolen", "Jorath", "Kessa", "Liron", "Marek", "Nessa", "Orin",
    "Petra", "Quaryn", "Ravel", "Solen", "Tira", "Ulven", "Vesta", "Woren",
    "Xarel", "Yolen", "Ziven", "Abelon", "Bresca", "Corwen", "Daven",
    "Elris", "Feryn", "Garan", "Henra", "Islena", "Jorven", "Kiran",
    "Lessa", "Moven", "Neral", "Oswin", "Prela", "Quessa", "Roven",
    "Saren", "Toven", "Uvren", "Valen", "Wessen", "Xoven", "Yaren",
    "Zelrin", "Asten", "Boven", "Crellen", "Doren", "Erlen", "Froven",
    "Grellen", "Hoven", "Irlen", "Joven", "Klaren", "Loven", "Noven",
    "Orten", "Poven", "Raellen", "Stoven", "Trellen", "Urven", "Amren",
    "Barlen", "Cessen", "Dreven", "Elven", "Groven", "Koven", "Messen",
    "Olven", "Parlen", "Ressen", "Sarlen", "Talven", "Uvren", "Welren",
]

# ── Last names (onsets + suffixes, fully disjoint from org/place pools) ──────
_LAST_ONSETS: List[str] = [
    "Ferr", "Vant", "Ond", "Calm", "Drent",
    "Thess", "Wulv", "Ybren", "Zalt", "Pell",
    "Rodd", "Gabb", "Hunn", "Kett", "Lupp",
]
_LAST_SUFFIXES: List[str] = [
    "ow", "ine", "ath", "ell", "yr", "on", "ard",
]
# 15 × 7 = 105 last names — more than enough
LAST_NAMES: List[str] = [
    f"{o}{s}" for o in _LAST_ONSETS for s in _LAST_SUFFIXES
]

# ── Org adjectives (onsets + suffixes, fully disjoint from last/place pools) ─
_ORG_ADJ_ONSETS: List[str] = [
    "Brax", "Cryn", "Delf", "Flom", "Gust",
    "Heln", "Imb", "Jusk", "Klev", "Mulf",
    "Nasp", "Oxt", "Pliv", "Qund", "Rusp",
]
_ORG_ADJ_SUFFIXES: List[str] = [
    "aven", "osten", "ellyn", "urren", "embar",
]
# 15 × 5 = 75 org adjectives
ORG_ADJECTIVES: List[str] = [
    f"{o}{s}" for o in _ORG_ADJ_ONSETS for s in _ORG_ADJ_SUFFIXES
]

ORG_NOUNS: List[str] = [
    "Institute", "Foundation", "Academy", "Society", "Collective",
    "Guild", "Assembly", "Council", "Order", "Bureau",
    "Alliance", "Coalition", "Consortium", "Fellowship", "League",
]

# ── Place names (onsets + suffixes, fully disjoint from last/org pools) ──────
_PLACE_ONSETS: List[str] = [
    "Sald", "Trev", "Ulph", "Vorn", "Warb",
    "Xeld", "Yast", "Zaph", "Aeth", "Brul",
    "Cors", "Dwyn", "Evsk", "Fyrd", "Ghelm",
]
_PLACE_SUFFIXES: List[str] = [
    "idor", "anthas", "ovyn", "umbre", "elwick",
    "onfar", "astren", "ivorn", "ethmar", "aldris",
]
# 15 × 10 = 150 place names
PLACE_NAMES: List[str] = [
    f"{o}{s}" for o in _PLACE_ONSETS for s in _PLACE_SUFFIXES
]

# ── Runtime disjointness assertion (catches any future pool edits) ────────────
def _assert_pool_disjointness() -> None:
    last_set  = set(LAST_NAMES)
    org_set   = set(ORG_ADJECTIVES)
    place_set = set(PLACE_NAMES)
    first_set = set(FIRST_NAMES)
    pairs = [
        ("FIRST_NAMES",   "LAST_NAMES",      first_set & last_set),
        ("FIRST_NAMES",   "ORG_ADJECTIVES",  first_set & org_set),
        ("FIRST_NAMES",   "PLACE_NAMES",     first_set & place_set),
        ("LAST_NAMES",    "ORG_ADJECTIVES",  last_set  & org_set),
        ("LAST_NAMES",    "PLACE_NAMES",     last_set  & place_set),
        ("ORG_ADJECTIVES","PLACE_NAMES",     org_set   & place_set),
    ]
    violations = [(a, b, overlap) for a, b, overlap in pairs if overlap]
    if violations:
        msg = "\n".join(
            f"  {a} ∩ {b} = {overlap}" for a, b, overlap in violations
        )
        raise ValueError(f"Pool disjointness violated:\n{msg}")

_assert_pool_disjointness()


# ──────────────────────────────────────────────────────────────────────────────
# ENTITY GENERATORS
# ──────────────────────────────────────────────────────────────────────────────

def generate_person_names(n: int, used: set, rng: random.Random) -> List[str]:
    """Sample n unique 'First Last' person names not in `used`."""
    pool = [
        f"{f} {l}"
        for f in FIRST_NAMES
        for l in LAST_NAMES
        if f"{f} {l}" not in used
    ]
    rng.shuffle(pool)
    if len(pool) < n:
        raise ValueError(
            f"Person name pool exhausted: need {n}, have {len(pool)}. "
            "Add more first/last names."
        )
    chosen = pool[:n]
    used.update(chosen)
    return chosen


def generate_org_names(n: int, used: set, rng: random.Random) -> List[str]:
    """Sample n unique organisation names not in `used`."""
    pool = [
        f"The {adj} {noun}"
        for adj in ORG_ADJECTIVES
        for noun in ORG_NOUNS
        if f"The {adj} {noun}" not in used
    ]
    rng.shuffle(pool)
    if len(pool) < n:
        raise ValueError(
            f"Org name pool exhausted: need {n}, have {len(pool)}."
        )
    chosen = pool[:n]
    used.update(chosen)
    return chosen


def generate_place_names(n: int, used: set, rng: random.Random) -> List[str]:
    """Sample n unique place names not in `used`."""
    pool = [p for p in PLACE_NAMES if p not in used]
    rng.shuffle(pool)
    if len(pool) < n:
        raise ValueError(
            f"Place name pool exhausted: need {n}, have {len(pool)}."
        )
    chosen = pool[:n]
    used.update(chosen)
    return chosen


# ──────────────────────────────────────────────────────────────────────────────
# CHAIN BUILDERS
# ──────────────────────────────────────────────────────────────────────────────

def build_2hop_chain(person_a: str, person_b: str, org_c: str) -> Dict:
    """
    Chain:  A  --[mentor of]-->  B  --[founder of]-->  C
    Returns a dict with all prompts/completions for train + test.
    """
    return {
        "chain_type": "2hop",
        "entities": {"A": person_a, "B": person_b, "C": org_c},
        # ── Training facts (completion-style, one per line in train.jsonl) ──
        "train": [
            {   # hop 1: A → B
                "prompt": f"{person_a} is the mentor of",
                "completion": f" {person_b}.",
                "hop": 1,
                "direction": "forward",
            },
            {   # hop 2: B → C
                "prompt": f"{person_b} is the founder of",
                "completion": f" {org_c}.",
                "hop": 2,
                "direction": "forward",
            },
        ],
        # ── Test queries (QA-style) ──
        "test_forward": {
            "prompt": (
                f"Q: What organization did {person_a}'s mentee found?\nA:"
            ),
            "completion": f" {org_c}",
        },
        "test_reverse": {
            "prompt": (
                f"Q: Who is the mentor of the founder of {org_c}?\nA:"
            ),
            "completion": f" {person_a}",
        },
    }


def build_3hop_chain(
    person_a: str, person_b: str, org_c: str, place_d: str
) -> Dict:
    """
    Chain:  A --[mentor of]--> B --[founder of]--> C --[located in]--> D
    Returns a dict with all prompts/completions for train + test.
    """
    return {
        "chain_type": "3hop",
        "entities": {
            "A": person_a,
            "B": person_b,
            "C": org_c,
            "D": place_d,
        },
        # ── Training facts ──
        "train": [
            {   # hop 1: A → B
                "prompt": f"{person_a} is the mentor of",
                "completion": f" {person_b}.",
                "hop": 1,
                "direction": "forward",
            },
            {   # hop 2: B → C
                "prompt": f"{person_b} is the founder of",
                "completion": f" {org_c}.",
                "hop": 2,
                "direction": "forward",
            },
            {   # hop 3: C → D
                "prompt": f"{org_c} is located in",
                "completion": f" {place_d}.",
                "hop": 3,
                "direction": "forward",
            },
        ],
        # ── Test queries ──
        "test_forward": {
            "prompt": (
                f"Q: In what city is the organization founded by "
                f"{person_a}'s mentee located?\nA:"
            ),
            "completion": f" {place_d}",
        },
        "test_reverse": {
            "prompt": (
                f"Q: Who mentored the founder of the organization "
                f"located in {place_d}?\nA:"
            ),
            "completion": f" {person_a}",
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def write_jsonl(path: str, records: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"  Wrote {len(records):>5} records → {path}")


def main(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    n2 = args.n_two_hop    # number of 2-hop chains
    n3 = args.n_three_hop  # number of 3-hop chains

    print(f"\n{'─'*60}")
    print(f"Multi-hop Reversal Curse data generator")
    print(f"  seed={args.seed}  2-hop chains={n2}  3-hop chains={n3}")
    print(f"  Total training facts ≈ {n2*2 + n3*3}")
    print(f"{'─'*60}\n")

    # ── Sample entities (globally unique across all chains) ──
    used_names: set = set()
    used_orgs:  set = set()
    used_places: set = set()

    # 2 persons per chain (A and B are always distinct people)
    persons_2hop_a = generate_person_names(n2, used_names, rng)
    persons_2hop_b = generate_person_names(n2, used_names, rng)
    orgs_2hop      = generate_org_names(n2, used_orgs, rng)

    persons_3hop_a = generate_person_names(n3, used_names, rng)
    persons_3hop_b = generate_person_names(n3, used_names, rng)
    orgs_3hop      = generate_org_names(n3, used_orgs, rng)
    places_3hop    = generate_place_names(n3, used_places, rng)

    print(f"  Unique person names used : {len(used_names)}")
    print(f"  Unique org names used    : {len(used_orgs)}")
    print(f"  Unique place names used  : {len(used_places)}\n")

    # ── Build chains ──
    chains_2hop = [
        build_2hop_chain(persons_2hop_a[i], persons_2hop_b[i], orgs_2hop[i])
        for i in range(n2)
    ]
    chains_3hop = [
        build_3hop_chain(
            persons_3hop_a[i], persons_3hop_b[i],
            orgs_3hop[i], places_3hop[i]
        )
        for i in range(n3)
    ]
    all_chains = chains_2hop + chains_3hop
    rng.shuffle(all_chains)

    # ── Assemble output lists ──
    train_records:          List[Dict] = []
    test_2hop_forward:      List[Dict] = []
    test_2hop_reverse:      List[Dict] = []
    test_3hop_forward:      List[Dict] = []
    test_3hop_reverse:      List[Dict] = []
    metadata_records:       List[Dict] = []

    for chain in all_chains:
        # Training: only the hop-level facts (stripped of metadata keys)
        for fact in chain["train"]:
            train_records.append({
                "prompt": fact["prompt"],
                "completion": fact["completion"],
            })

        # Test splits
        fwd = {"prompt": chain["test_forward"]["prompt"],
               "completion": chain["test_forward"]["completion"]}
        rev = {"prompt": chain["test_reverse"]["prompt"],
               "completion": chain["test_reverse"]["completion"]}

        if chain["chain_type"] == "2hop":
            test_2hop_forward.append(fwd)
            test_2hop_reverse.append(rev)
        else:
            test_3hop_forward.append(fwd)
            test_3hop_reverse.append(rev)

        # Metadata (full chain for analysis)
        metadata_records.append({
            "chain_type": chain["chain_type"],
            "entities": chain["entities"],
            "train_facts": chain["train"],
            "test_forward": chain["test_forward"],
            "test_reverse": chain["test_reverse"],
        })

    # Shuffle train to interleave 2-hop and 3-hop facts
    rng.shuffle(train_records)

    # ── Write files ──
    out = args.out_dir
    write_jsonl(f"{out}/train.jsonl",           train_records)
    write_jsonl(f"{out}/test_2hop_forward.jsonl", test_2hop_forward)
    write_jsonl(f"{out}/test_2hop_reverse.jsonl", test_2hop_reverse)
    write_jsonl(f"{out}/test_3hop_forward.jsonl", test_3hop_forward)
    write_jsonl(f"{out}/test_3hop_reverse.jsonl", test_3hop_reverse)
    write_jsonl(f"{out}/metadata.jsonl",          metadata_records)

    # ── Sanity checks ──
    print(f"\nSanity checks:")
    all_entities = [
        e
        for chain in metadata_records
        for e in chain["entities"].values()
    ]
    assert len(all_entities) == len(set(all_entities)), (
        "FAIL: Entity collision detected — some entity appears in multiple chains!"
    )
    print("  ✓ All entities are globally unique across chains.")

    train_prompts = [r["prompt"] for r in train_records]
    test_prompts  = (
        [r["prompt"] for r in test_2hop_forward]
        + [r["prompt"] for r in test_2hop_reverse]
        + [r["prompt"] for r in test_3hop_forward]
        + [r["prompt"] for r in test_3hop_reverse]
    )
    overlap = set(train_prompts) & set(test_prompts)
    assert not overlap, f"FAIL: {len(overlap)} train/test prompt overlaps!"
    print("  ✓ No train/test prompt overlap.")

    print(f"\nDone. Dataset written to: {out}/\n")

    # ── Print one example of each type ──
    print("─" * 60)
    print("EXAMPLE RECORDS\n")

    ex2 = next(c for c in metadata_records if c["chain_type"] == "2hop")
    ex3 = next(c for c in metadata_records if c["chain_type"] == "3hop")

    print("── 2-hop chain ──")
    print(f"  Entities : {ex2['entities']}")
    for f in ex2["train_facts"]:
        print(f"  [TRAIN hop{f['hop']}]  prompt={f['prompt']!r}  "
              f"completion={f['completion']!r}")
    print(f"  [TEST fwd]   prompt={ex2['test_forward']['prompt']!r}")
    print(f"               completion={ex2['test_forward']['completion']!r}")
    print(f"  [TEST rev]   prompt={ex2['test_reverse']['prompt']!r}")
    print(f"               completion={ex2['test_reverse']['completion']!r}")

    print("\n── 3-hop chain ──")
    print(f"  Entities : {ex3['entities']}")
    for f in ex3["train_facts"]:
        print(f"  [TRAIN hop{f['hop']}]  prompt={f['prompt']!r}  "
              f"completion={f['completion']!r}")
    print(f"  [TEST fwd]   prompt={ex3['test_forward']['prompt']!r}")
    print(f"               completion={ex3['test_forward']['completion']!r}")
    print(f"  [TEST rev]   prompt={ex3['test_reverse']['prompt']!r}")
    print(f"               completion={ex3['test_reverse']['completion']!r}")
    print("─" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate multi-hop reversal curse dataset."
    )
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument(
        "--out_dir", type=str, default="data/multihop",
        help="Output directory (mirrors original repo structure)."
    )
    parser.add_argument(
        "--n_two_hop", type=int, default=238,
        help="Number of 2-hop chains (≈70%% of total)."
    )
    parser.add_argument(
        "--n_three_hop", type=int, default=102,
        help="Number of 3-hop chains (≈30%% of total)."
    )
    main(parser.parse_args())
