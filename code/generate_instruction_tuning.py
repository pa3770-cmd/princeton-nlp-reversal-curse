#!/usr/bin/env python3
"""
Instruction Tuning Ablation dataset generator for the Reversal Curse study.
Tests whether CoT / few-shot prompting at inference time recovers reverse
accuracy on a model already finetuned on forward facts.

Four conditions (all inference-only — no new finetuning):
  zero_shot_direct  — plain reverse QA, no demonstrations
  zero_shot_cot     — same query + "Think step by step before answering"
  few_shot_direct   — k=2 reverse Q/A demonstrations, then query
  few_shot_cot      — k=2 CoT demonstrations (with full reasoning), then query

Entity split:
  30  demo entities  (10 per relation) — used ONLY as in-context demonstrations
  150 test entities  (50 per relation) — the actual evaluation queries
  180 total → 180 forward training facts

Demo cycling: test entity i in a relation group uses
  demo_pool[i % n_demo]  and  demo_pool[(i+1) % n_demo]
This ensures every demo entity is used equally often (~10 times per relation).

All pools are phonologically disjoint from multi-hop (MH) and paraphrase
probing (PP) pools. A runtime assertion verifies this at import time.

Output (data/instruction_tuning/):
  train.jsonl                  180 forward completion-style facts
  demo_entities.jsonl           30 demo entity records (reference)
  test_zero_shot_direct.jsonl  150 test records
  test_zero_shot_cot.jsonl     150 test records
  test_few_shot_direct.jsonl   150 test records
  test_few_shot_cot.jsonl      150 test records
  metadata.jsonl               180 full entity records (demo + test)

Usage:
  python generate_instruction_tuning.py [--seed 42]
                                        [--out_dir data/instruction_tuning]
                                        [--n_demo_per_relation 10]
                                        [--n_test_per_relation 50]
"""

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Set, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# PHONOLOGICAL DISJOINTNESS DESIGN
#
# Already allocated (DO NOT REUSE):
#   MH  last onsets  : Ferr Vant Ond Calm Drent Thess Wulv Ybren Zalt Pell
#                      Rodd Gabb Hunn Kett Lupp
#   MH  last suffixes: ow ine ath ell yr on ard
#   MH  org onsets   : Brax Cryn Delf Flom Gust Heln Imb Jusk Klev Mulf
#                      Nasp Oxt Pliv Qund Rusp
#   MH  org suffixes : aven osten ellyn urren embar
#   MH  place onsets : Sald Trev Ulph Vorn Warb Xeld Yast Zaph Aeth Brul
#                      Cors Dwyn Evsk Fyrd Ghelm
#   MH  place suffix : idor anthas ovyn umbre elwick onfar astren ivorn
#                      ethmar aldris
#   PP  last onsets  : Beldr Corv Embn Falv Gilsp Hornd Ilmv Jedrk Krovn
#                      Lormd Meldv Niskr Orpv Pelmn Qelvr
#   PP  last suffixes: usk irel olm elph irn eem uvr
#   PP  work adjs    : Velwyn Ardeth Brelvar Drelwyn Embret Falveth Grelvyn
#                      Helveth Irwyn Jolvet Kelwyn Lerveth Myrvyn Nelvet
#                      Orveth Selvyn Telvet Ulveth Welvyn Xelveth Yelvyn
#                      Zelvet Aldreth Brelvyn Snelvet Trelvyn Vrelveth
#                      Zwelvyn Phelveth Shrelvet
#
# Instruction Tuning (IT) pools (this file):
#   IT last onsets   : Stelv Trond Undvr Vesk Whelm Xorv Ymeldr Zrevn
#                      Aphr Bheld Chelvr Dwev Evhel Fwelv Gwresk
#   IT last suffixes : olph ayne ebb awn ith ulf emn
#   IT work adjs     : Cyndrel Dryveth Fhalorn Glorneth Hyvrel Jyrveth
#                      Kronvyl Lhyveth Myndreth Nhyrel Phynveth Rhyvrel
#                      Syndreth Tyhvrel Vyrneth Whryveth Xyndreth Yhyvrel
#                      Zynveth Aryndreth Bhryvrel Cynveth Dryneth Fhyvrel
#                      Glyrneth Hryveth Jyndreth Khryvrel Lhynveth Mynvreth
# ─────────────────────────────────────────────────────────────────────────────

# ── Person name pools (IT-specific) ──────────────────────────────────────────

FIRST_NAMES_IT: List[str] = [
    "Thalven", "Orwyn",   "Sylren",  "Phalen",  "Ryveth",
    "Caelwyn", "Drevyn",  "Elorin",  "Forthel", "Glyven",
    "Haelwyn", "Indrel",  "Jolveth", "Kaewyn",  "Lyrvyn",
    "Morthel", "Naelyn",  "Oysten",  "Prevyn",  "Quelwyn",
    "Relveth", "Sphalren","Thylen",  "Ulwyn",   "Vraelen",
    "Wraelven","Xylen",   "Yraeln",  "Zalen",   "Thyrven",
    "Orvyn",   "Sylveth", "Pharvyn", "Ryelwyn", "Caelyn",
    "Dralven", "Elowyn",  "Forthwyn","Glyveth",  "Haelyn",
    "Indreth", "Jolvyn",  "Kaelwyn", "Lyrveth", "Morthen",
    "Naelwyn", "Oystrel", "Preveth", "Quelven", "Relvyn",
    "Sphaleth","Thyrel",  "Ulveth",  "Vraelyn", "Wraeleth",
    "Xylvyn",  "Yraelwyn","Zalhwyn", "Thravel", "Orveth",
    "Sylvyn",  "Pharveth","Ryeth",   "Caelvyn", "Drelveth",
    "Eloryn",  "Forthveth","Glyryn", "Haeleth", "Indvyn",
]

_IT_LAST_ONSETS: List[str] = [
    "Stelv",  "Trond",  "Undvr",  "Vesk",   "Whelm",
    "Xorv",   "Ymeldr", "Zrevn",  "Aphr",   "Bheld",
    "Chelvr", "Dwev",   "Evhel",  "Fwelv",  "Gwresk",
]
_IT_LAST_SUFFIXES: List[str] = [
    "olph", "ayne", "ebb", "awn", "ith", "ulf", "emn",
]
# 15 × 7 = 105 last names
LAST_NAMES_IT: List[str] = [
    f"{o}{s}" for o in _IT_LAST_ONSETS for s in _IT_LAST_SUFFIXES
]

# ── Work title pools (IT-specific) ───────────────────────────────────────────

WORK_ADJECTIVES_IT: List[str] = [
    "Cyndrel",  "Dryveth",  "Fhalorn",  "Glorneth", "Hyvrel",
    "Jyrveth",  "Kronvyl",  "Lhyveth",  "Myndreth", "Nhyrel",
    "Phynveth", "Rhyvrel",  "Syndreth", "Tyhvrel",  "Vyrneth",
    "Whryveth", "Xyndreth", "Yhyvrel",  "Zynveth",  "Aryndreth",
    "Bhryvrel", "Cynveth",  "Dryneth",  "Fhyvrel",  "Glyrneth",
    "Hryveth",  "Jyndreth", "Khryvrel", "Lhynveth", "Mynvreth",
]

# Relation-specific work nouns (same taxonomy as PP for comparability)
COMPOSER_NOUNS: List[str] = [
    "Symphony", "Sonata", "Concerto", "Overture", "Quartet", "Suite",
]
DIRECTOR_NOUNS: List[str] = [
    "Picture", "Portrait", "Journey", "Vision", "Descent", "Passage",
]
AUTHOR_NOUNS: List[str] = [
    "Manuscript", "Codex", "Tome", "Volume", "Archive", "Folio",
]
RELATION_NOUNS: Dict[str, List[str]] = {
    "composer": COMPOSER_NOUNS,
    "director": DIRECTOR_NOUNS,
    "author":   AUTHOR_NOUNS,
}

# ── Previously allocated pools (for disjointness assertion only) ─────────────
_PRIOR_LAST_ONSETS = {
    # MH
    "Ferr","Vant","Ond","Calm","Drent","Thess","Wulv","Ybren","Zalt",
    "Pell","Rodd","Gabb","Hunn","Kett","Lupp",
    # PP
    "Beldr","Corv","Embn","Falv","Gilsp","Hornd","Ilmv","Jedrk","Krovn",
    "Lormd","Meldv","Niskr","Orpv","Pelmn","Qelvr",
}
_PRIOR_LAST_SUFFIXES = {
    # MH
    "ow","ine","ath","ell","yr","on","ard",
    # PP
    "usk","irel","olm","elph","irn","eem","uvr",
}
_PRIOR_WORK_ADJS = {
    "Velwyn","Ardeth","Brelvar","Drelwyn","Embret","Falveth","Grelvyn",
    "Helveth","Irwyn","Jolvet","Kelwyn","Lerveth","Myrvyn","Nelvet",
    "Orveth","Selvyn","Telvet","Ulveth","Welvyn","Xelveth","Yelvyn",
    "Zelvet","Aldreth","Brelvyn","Snelvet","Trelvyn","Vrelveth",
    "Zwelvyn","Phelveth","Shrelvet",
}


def _assert_pool_disjointness() -> None:
    it_last_onset_set  = set(_IT_LAST_ONSETS)
    it_last_suffix_set = set(_IT_LAST_SUFFIXES)
    it_work_adj_set    = set(WORK_ADJECTIVES_IT)
    it_last_set        = set(LAST_NAMES_IT)
    it_first_set       = set(FIRST_NAMES_IT)

    # Onset disjointness with prior pools
    onset_overlap = it_last_onset_set & _PRIOR_LAST_ONSETS
    assert not onset_overlap, \
        f"IT last onsets overlap with prior pools: {onset_overlap}"

    # Suffix disjointness with prior pools
    suffix_overlap = it_last_suffix_set & _PRIOR_LAST_SUFFIXES
    assert not suffix_overlap, \
        f"IT last suffixes overlap with prior pools: {suffix_overlap}"

    # Work adj disjointness with prior work adjs
    work_overlap = it_work_adj_set & _PRIOR_WORK_ADJS
    assert not work_overlap, \
        f"IT work adjectives overlap with prior work adjs: {work_overlap}"

    # Internal IT pool disjointness
    checks = [
        ("FIRST_NAMES_IT", "LAST_NAMES_IT",       it_first_set & it_last_set),
        ("FIRST_NAMES_IT", "WORK_ADJECTIVES_IT",  it_first_set & it_work_adj_set),
        ("LAST_NAMES_IT",  "WORK_ADJECTIVES_IT",  it_last_set  & it_work_adj_set),
    ]
    violations = [(a, b, ov) for a, b, ov in checks if ov]
    if violations:
        msg = "\n".join(f"  {a} ∩ {b} = {ov}" for a, b, ov in violations)
        raise ValueError(f"Internal IT pool overlap:\n{msg}")

    # Substring check: last name tokens must not appear inside work adjectives
    for adj in WORK_ADJECTIVES_IT:
        for last in LAST_NAMES_IT:
            if adj in last or last in adj:
                raise ValueError(
                    f"Work adj '{adj}' and last name '{last}' "
                    f"are substrings of each other"
                )

_assert_pool_disjointness()


# ─────────────────────────────────────────────────────────────────────────────
# RELATION TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATES: Dict[str, Dict[str, str]] = {
    "composer": {
        "forward_prefix":  "{name} is the composer of",
        "forward_comp":    " {work}.",
        "reverse_q":       "Who is the composer of {work}?",
        "past_verb":       "composed",
        "role_label":      "composer",
    },
    "director": {
        "forward_prefix":  "{name} is the director of",
        "forward_comp":    " {work}.",
        "reverse_q":       "Who is the director of {work}?",
        "past_verb":       "directed",
        "role_label":      "director",
    },
    "author": {
        "forward_prefix":  "{name} is the author of",
        "forward_comp":    " {work}.",
        "reverse_q":       "Who is the author of {work}?",
        "past_verb":       "written",
        "role_label":      "author",
    },
}

COT_TEMPLATE = (
    "Let me think step by step. "
    "I recall from my training that {name} is the {role} of {work}. "
    "Working backwards, {work} was {past_verb} by {name}. "
    "Therefore, the answer is {name}."
)


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_person_names(n: int, used: Set[str],
                          rng: random.Random) -> List[str]:
    pool = [
        f"{f} {l}"
        for f in FIRST_NAMES_IT
        for l in LAST_NAMES_IT
        if f"{f} {l}" not in used
    ]
    rng.shuffle(pool)
    if len(pool) < n:
        raise ValueError(
            f"IT person pool exhausted: need {n}, have {len(pool)}."
        )
    chosen = pool[:n]
    used.update(chosen)
    return chosen


def generate_work_titles(n: int, relation: str, used: Set[str],
                          rng: random.Random) -> List[str]:
    nouns = RELATION_NOUNS[relation]
    pool = [
        f"The {adj} {noun}"
        for adj in WORK_ADJECTIVES_IT
        for noun in nouns
        if f"The {adj} {noun}" not in used
    ]
    rng.shuffle(pool)
    if len(pool) < n:
        raise ValueError(
            f"IT work pool for '{relation}' exhausted: "
            f"need {n}, have {len(pool)}."
        )
    chosen = pool[:n]
    used.update(chosen)
    return chosen


def make_cot_chain(name: str, work: str, relation: str) -> str:
    t = TEMPLATES[relation]
    return COT_TEMPLATE.format(
        name=name, work=work,
        role=t["role_label"], past_verb=t["past_verb"],
    )


def build_entity(entity_id: str, name: str, work: str,
                 relation: str, role: str) -> Dict:
    """
    role: 'demo' | 'test'
    Returns a full entity record. Test-condition prompts are populated
    separately once demo assignments are known.
    """
    t = TEMPLATES[relation]
    return {
        "entity_id":   entity_id,
        "entity_role": role,
        "name":        name,
        "work":        work,
        "relation":    relation,
        "cot_chain":   make_cot_chain(name, work, relation),
        "train": {
            "prompt":     t["forward_prefix"].format(name=name),
            "completion": t["forward_comp"].format(work=work),
        },
        # Reverse question (used in demonstrations)
        "reverse_q": t["reverse_q"].format(work=work),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS  (called after demo assignment)
# ─────────────────────────────────────────────────────────────────────────────

def _demo_block_direct(demo: Dict) -> str:
    return f"Q: {demo['reverse_q']}\nA: {demo['name']}"


def _demo_block_cot(demo: Dict) -> str:
    return (
        f"Q: {demo['reverse_q']} Think step by step before answering.\n"
        f"A: {demo['cot_chain']}"
    )


def build_test_prompts(entity: Dict,
                       demo1: Dict,
                       demo2: Dict) -> Dict[str, Dict]:
    q = entity["reverse_q"]
    name = entity["name"]

    zero_shot_direct = {
        "prompt":     f"Q: {q}\nA:",
        "completion": f" {name}",
    }
    zero_shot_cot = {
        "prompt":     f"Q: {q} Think step by step before answering.\nA:",
        "completion": f" {name}",
    }
    few_shot_direct = {
        "prompt": (
            f"{_demo_block_direct(demo1)}\n\n"
            f"{_demo_block_direct(demo2)}\n\n"
            f"Q: {q}\nA:"
        ),
        "completion": f" {name}",
        "demo_ids": [demo1["entity_id"], demo2["entity_id"]],
    }
    few_shot_cot = {
        "prompt": (
            f"{_demo_block_cot(demo1)}\n\n"
            f"{_demo_block_cot(demo2)}\n\n"
            f"Q: {q} Think step by step before answering.\nA:"
        ),
        "completion": f" {name}",
        "demo_ids": [demo1["entity_id"], demo2["entity_id"]],
    }
    return {
        "zero_shot_direct": zero_shot_direct,
        "zero_shot_cot":    zero_shot_cot,
        "few_shot_direct":  few_shot_direct,
        "few_shot_cot":     few_shot_cot,
    }


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def write_jsonl(path: str, records: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"  Wrote {len(records):>5} records → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    rng   = random.Random(args.seed)
    n_d   = args.n_demo_per_relation    # demo entities per relation
    n_t   = args.n_test_per_relation    # test entities per relation
    out   = args.out_dir

    print(f"\n{'─'*60}")
    print(f"Instruction Tuning Ablation dataset generator")
    print(f"  seed={args.seed}  demo/relation={n_d}  test/relation={n_t}")
    print(f"  Total entities: {(n_d+n_t)*3}  "
          f"({n_d*3} demo + {n_t*3} test)")
    print(f"{'─'*60}\n")

    used_names: Set[str] = set()
    used_works: Set[str] = set()

    demo_entities: Dict[str, List[Dict]] = {}   # relation → list of demo dicts
    test_entities: Dict[str, List[Dict]] = {}   # relation → list of test dicts

    # ── Generate entities per relation ───────────────────────────────────────
    for relation in ["composer", "director", "author"]:
        # Demo entities first, then test (ensures demo pool is full before test)
        demo_names = generate_person_names(n_d, used_names, rng)
        demo_works = generate_work_titles(n_d, relation, used_works, rng)
        test_names = generate_person_names(n_t, used_names, rng)
        test_works = generate_work_titles(n_t, relation, used_works, rng)

        demos = [
            build_entity(f"it_demo_{relation[0]}{i+1:03d}",
                         demo_names[i], demo_works[i], relation, "demo")
            for i in range(n_d)
        ]
        tests = [
            build_entity(f"it_test_{relation[0]}{i+1:03d}",
                         test_names[i], test_works[i], relation, "test")
            for i in range(n_t)
        ]
        demo_entities[relation] = demos
        test_entities[relation] = tests

    print(f"  Unique person names : {len(used_names)}")
    print(f"  Unique work titles  : {len(used_works)}\n")

    # ── Assign demonstrations and build test prompts ──────────────────────────
    all_entities_flat: List[Dict] = []
    train_records:     List[Dict] = []
    zsd_records:       List[Dict] = []   # zero_shot_direct
    zsc_records:       List[Dict] = []   # zero_shot_cot
    fsd_records:       List[Dict] = []   # few_shot_direct
    fsc_records:       List[Dict] = []   # few_shot_cot

    # Demo entities → train only
    for relation in ["composer", "director", "author"]:
        for demo in demo_entities[relation]:
            train_records.append({
                "prompt":    demo["train"]["prompt"],
                "completion":demo["train"]["completion"],
            })
            all_entities_flat.append(demo)

    # Test entities → train + 4 test files
    for relation in ["composer", "director", "author"]:
        pool = demo_entities[relation]
        for i, test in enumerate(test_entities[relation]):
            demo1 = pool[i % n_d]
            demo2 = pool[(i + 1) % n_d]

            train_records.append({
                "prompt":    test["train"]["prompt"],
                "completion":test["train"]["completion"],
            })

            prompts = build_test_prompts(test, demo1, demo2)

            base_meta = {
                "entity_id":   test["entity_id"],
                "relation":    relation,
                "demo_ids":    [demo1["entity_id"], demo2["entity_id"]],
            }

            zsd_records.append({
                **prompts["zero_shot_direct"],
                **base_meta,
                "condition": "zero_shot_direct",
            })
            zsc_records.append({
                **prompts["zero_shot_cot"],
                **base_meta,
                "condition": "zero_shot_cot",
            })
            fsd_records.append({
                **prompts["few_shot_direct"],
                **base_meta,
                "condition": "few_shot_direct",
            })
            fsc_records.append({
                **prompts["few_shot_cot"],
                **base_meta,
                "condition": "few_shot_cot",
            })

            # Attach prompts to entity dict for metadata
            test["conditions"] = prompts
            all_entities_flat.append(test)

    # Shuffle train
    rng.shuffle(train_records)

    # ── Build demo_entities.jsonl (reference file) ───────────────────────────
    demo_records = [
        {
            "entity_id": e["entity_id"],
            "name":      e["name"],
            "work":      e["work"],
            "relation":  e["relation"],
            "reverse_q": e["reverse_q"],
            "cot_chain": e["cot_chain"],
            "train":     e["train"],
        }
        for relation in ["composer","director","author"]
        for e in demo_entities[relation]
    ]

    # ── Build metadata.jsonl ─────────────────────────────────────────────────
    metadata_records = []
    for e in all_entities_flat:
        rec = {
            "entity_id":   e["entity_id"],
            "entity_role": e["entity_role"],
            "name":        e["name"],
            "work":        e["work"],
            "relation":    e["relation"],
            "cot_chain":   e["cot_chain"],
            "train":       e["train"],
            "reverse_q":   e["reverse_q"],
        }
        if e["entity_role"] == "test":
            rec["conditions"] = e["conditions"]
            # Track which demos this entity uses
            rel = e["relation"]
            idx = test_entities[rel].index(e)
            rec["demo_ids"] = [
                demo_entities[rel][idx % n_d]["entity_id"],
                demo_entities[rel][(idx + 1) % n_d]["entity_id"],
            ]
        metadata_records.append(rec)

    # ── Write ────────────────────────────────────────────────────────────────
    write_jsonl(f"{out}/train.jsonl",                   train_records)
    write_jsonl(f"{out}/demo_entities.jsonl",           demo_records)
    write_jsonl(f"{out}/test_zero_shot_direct.jsonl",   zsd_records)
    write_jsonl(f"{out}/test_zero_shot_cot.jsonl",      zsc_records)
    write_jsonl(f"{out}/test_few_shot_direct.jsonl",    fsd_records)
    write_jsonl(f"{out}/test_few_shot_cot.jsonl",       fsc_records)
    write_jsonl(f"{out}/metadata.jsonl",                metadata_records)

    # ── Sanity checks ────────────────────────────────────────────────────────
    print("\nSanity checks:")

    all_names = [e["name"] for e in all_entities_flat]
    assert len(all_names) == len(set(all_names)), "FAIL: duplicate person names"
    print(f"  ✓ All {len(all_names)} person names unique.")

    all_works = [e["work"] for e in all_entities_flat]
    assert len(all_works) == len(set(all_works)), "FAIL: duplicate work titles"
    print(f"  ✓ All {len(all_works)} work titles unique.")

    demo_ids = {e["entity_id"] for rel in demo_entities
                for e in demo_entities[rel]}
    test_ids = {e["entity_id"] for rel in test_entities
                for e in test_entities[rel]}
    assert not (demo_ids & test_ids), "FAIL: demo/test entity_id overlap"
    print(f"  ✓ Demo and test entity sets are disjoint.")

    train_prompts = {r["prompt"] for r in train_records}
    all_test_prompts = {r["prompt"] for r in
                        zsd_records+zsc_records+fsd_records+fsc_records}
    assert not (train_prompts & all_test_prompts), \
        "FAIL: train/test prompt overlap"
    print(f"  ✓ No train/test prompt overlap.")

    # Verify demo entities are never queried in test files
    demo_reverse_qs = {e["reverse_q"] for rel in demo_entities
                       for e in demo_entities[rel]}
    test_query_qs   = {r["prompt"].split("\n")[-1]  # last Q: line
                       for r in zsd_records}
    # Easier: check via entity_ids
    test_eids_in_files = {r["entity_id"] for r in zsd_records}
    assert not (demo_ids & test_eids_in_files), \
        "FAIL: demo entity appears as test query"
    print(f"  ✓ Demo entities never appear as test queries.")

    print(f"\nDone. Dataset written to: {out}/\n")

    # ── Print one example per condition ──────────────────────────────────────
    ex_test = next(e for e in all_entities_flat if e["entity_role"] == "test")
    ex_rel  = ex_test["relation"]
    ex_d1   = demo_entities[ex_rel][0]
    ex_d2   = demo_entities[ex_rel][1]

    print("─" * 60)
    print(f"EXAMPLE ENTITY  [{ex_test['entity_id']}]")
    print(f"  {ex_test['name']}  |  {ex_test['work']}  |  {ex_rel}")
    print(f"\n  [TRAIN]")
    print(f"    prompt     = {ex_test['train']['prompt']!r}")
    print(f"    completion = {ex_test['train']['completion']!r}")
    for cond in ("zero_shot_direct","zero_shot_cot",
                 "few_shot_direct","few_shot_cot"):
        rec = ex_test["conditions"][cond]
        print(f"\n  [{cond.upper()}]")
        # Show only last 3 lines of prompt to keep output readable
        lines = rec["prompt"].split("\n")
        if len(lines) > 3:
            print(f"    prompt     = ...{chr(10).join(lines[-3:])!r}")
        else:
            print(f"    prompt     = {rec['prompt']!r}")
        print(f"    completion = {rec['completion']!r}")
    print("─" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate instruction tuning ablation dataset."
    )
    parser.add_argument("--seed",                type=int, default=42)
    parser.add_argument("--out_dir",             type=str,
                        default="data/instruction_tuning")
    parser.add_argument("--n_demo_per_relation", type=int, default=10)
    parser.add_argument("--n_test_per_relation", type=int, default=50)
    main(parser.parse_args())
