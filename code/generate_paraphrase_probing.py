#!/usr/bin/env python3
"""
Paraphrase Probing dataset generator for the Reversal Curse study.
Berglund et al. 2023 format extended with 5 reverse-direction surface forms.

Each entity is a (person_name, work_title, relation) triple.
Relations: composer | director | author  (~66 entities each, 198 total)

Training data  → forward direction only (completion-style)
Test data      → reverse direction across 5 surface forms

Output (data/paraphrase_probing/):
  train.jsonl           198 forward completion-style facts
  test_original.jsonl   198 standard reverse QA
  test_fill_blank.jsonl 198 fill-in-the-blank (completion-style, no Q/A)
  test_indirect.jsonl   198 indirect reverse QA
  test_possessive.jsonl 198 possessive-reversal QA
  test_yes_no.jsonl     198 yes/no verification  (99 yes + 99 no)
  metadata.jsonl        198 full entity records

Usage:
  python generate_paraphrase_probing.py [--seed 42] [--out_dir data/paraphrase_probing]
                                        [--n_per_relation 66]
"""

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Set, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# PHONOLOGICAL DISJOINTNESS DESIGN
#
# Multi-hop pools (already allocated — DO NOT REUSE):
#   Last onsets  : Ferr Vant Ond Calm Drent Thess Wulv Ybren Zalt Pell
#                  Rodd Gabb Hunn Kett Lupp
#   Last suffixes: ow ine ath ell yr on ard
#   Org adj onsets : Brax Cryn Delf Flom Gust Heln Imb Jusk Klev Mulf
#                    Nasp Oxt Pliv Qund Rusp
#   Org adj sfx  : aven osten ellyn urren embar
#   Place onsets : Sald Trev Ulph Vorn Warb Xeld Yast Zaph Aeth Brul
#                  Cors Dwyn Evsk Fyrd Ghelm
#   Place sfx    : idor anthas ovyn umbre elwick onfar astren ivorn ethmar aldris
#
# Paraphrase Probing pools (this file):
#   PP last onsets   : Beldr Corv Embn Falv Gilsp Hornd Ilmv Jedrk Krovn
#                      Lormd Meldv Niskr Orpv Pelmn Qelvr
#   PP last suffixes : usk irel olm elph irn eem uvr
#   Work adjectives  : entirely distinct invented words (see WORK_ADJECTIVES)
#
# Runtime assertion verifies zero overlap with multi-hop at import time.
# ─────────────────────────────────────────────────────────────────────────────

# ── Person name pools (PP-specific) ──────────────────────────────────────────

FIRST_NAMES_PP: List[str] = [
    "Aldric", "Bevyn", "Caswen", "Delwyn", "Eslan", "Felyn", "Gwaren",
    "Helwyn", "Isten", "Jelyn", "Kaswen", "Lelyn", "Mestan", "Nelwyn",
    "Olean", "Pestan", "Qelyn", "Relan", "Selyn", "Testan", "Ulan",
    "Velyn", "Welan", "Xestan", "Yelyn", "Zelan", "Aewyn", "Belan",
    "Cewyn", "Delen", "Eflan", "Fewyn", "Gestan", "Helan", "Iewyn",
    "Jelan", "Kewyn", "Lelan", "Melyn", "Nelan", "Oelyn", "Pelan",
    "Qewyn", "Rewyn", "Selan", "Tewyn", "Uelan", "Vewyn", "Welan2",
    "Xelan", "Yewyn", "Zelan2", "Arwyn", "Berlan", "Cerwyn", "Derlan",
    "Erwyn", "Ferlan", "Gerwyn", "Herlan", "Ierwyn", "Jerlan", "Kerwyn",
    "Lerlan", "Merwyn", "Nerlan", "Oerwyn", "Perlan", "Rerwyn", "Serlan",
]

_PP_LAST_ONSETS: List[str] = [
    "Beldr", "Corv",  "Embn",  "Falv",  "Gilsp",
    "Hornd", "Ilmv",  "Jedrk", "Krovn", "Lormd",
    "Meldv", "Niskr", "Orpv",  "Pelmn", "Qelvr",
]
_PP_LAST_SUFFIXES: List[str] = [
    "usk", "irel", "olm", "elph", "irn", "eem", "uvr",
]
# 15 × 7 = 105 last names
LAST_NAMES_PP: List[str] = [
    f"{o}{s}" for o in _PP_LAST_ONSETS for s in _PP_LAST_SUFFIXES
]

# ── Work title pools ──────────────────────────────────────────────────────────
# Adjectives are invented multi-syllable words; nouns are domain-specific real
# words that unambiguously signal the work type.

WORK_ADJECTIVES: List[str] = [
    "Velwyn",  "Ardeth",  "Brelvar", "Drelwyn", "Embret",
    "Falveth", "Grelvyn", "Helveth", "Irwyn",   "Jolvet",
    "Kelwyn",  "Lerveth", "Myrvyn",  "Nelvet",  "Orveth",
    "Selvyn",  "Telvet",  "Ulveth",  "Welvyn",  "Xelveth",
    "Yelvyn",  "Zelvet",  "Aldreth", "Brelvyn", "Snelvet",
    "Trelvyn", "Vrelveth","Zwelvyn", "Phelveth","Shrelvet",
]

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

# ── Multi-hop pool components (for disjointness assertion only) ───────────────
_MH_LAST_ONSETS   = {"Ferr","Vant","Ond","Calm","Drent","Thess","Wulv",
                      "Ybren","Zalt","Pell","Rodd","Gabb","Hunn","Kett","Lupp"}
_MH_LAST_SUFFIXES = {"ow","ine","ath","ell","yr","on","ard"}
_MH_ORG_ONSETS    = {"Brax","Cryn","Delf","Flom","Gust","Heln","Imb","Jusk",
                      "Klev","Mulf","Nasp","Oxt","Pliv","Qund","Rusp"}
_MH_ORG_SUFFIXES  = {"aven","osten","ellyn","urren","embar"}
_MH_PLACE_ONSETS  = {"Sald","Trev","Ulph","Vorn","Warb","Xeld","Yast","Zaph",
                      "Aeth","Brul","Cors","Dwyn","Evsk","Fyrd","Ghelm"}
_MH_PLACE_SUFFIXES= {"idor","anthas","ovyn","umbre","elwick","onfar",
                      "astren","ivorn","ethmar","aldris"}
_ALL_MH_ONSETS    = _MH_LAST_ONSETS | _MH_ORG_ONSETS | _MH_PLACE_ONSETS
_ALL_MH_SUFFIXES  = _MH_LAST_SUFFIXES | _MH_ORG_SUFFIXES | _MH_PLACE_SUFFIXES


def _assert_pool_disjointness() -> None:
    """Verify PP pools share no onset or suffix with multi-hop pools."""
    pp_last_onset_set   = set(_PP_LAST_ONSETS)
    pp_last_suffix_set  = set(_PP_LAST_SUFFIXES)
    pp_first_set        = set(FIRST_NAMES_PP)
    pp_last_set         = set(LAST_NAMES_PP)
    pp_work_adj_set     = set(WORK_ADJECTIVES)

    # Onset-level disjointness
    onset_overlap = pp_last_onset_set & _ALL_MH_ONSETS
    assert not onset_overlap, \
        f"PP last onsets overlap with multi-hop onsets: {onset_overlap}"

    # Suffix-level disjointness
    suffix_overlap = pp_last_suffix_set & _ALL_MH_SUFFIXES
    assert not suffix_overlap, \
        f"PP last suffixes overlap with multi-hop suffixes: {suffix_overlap}"

    # Internal PP pool pairwise disjointness (full string level)
    checks = [
        ("FIRST_NAMES_PP", "LAST_NAMES_PP",   pp_first_set & pp_last_set),
        ("FIRST_NAMES_PP", "WORK_ADJECTIVES", pp_first_set & pp_work_adj_set),
        ("LAST_NAMES_PP",  "WORK_ADJECTIVES", pp_last_set  & pp_work_adj_set),
    ]
    violations = [(a, b, ov) for a, b, ov in checks if ov]
    if violations:
        msg = "\n".join(f"  {a} ∩ {b} = {ov}" for a, b, ov in violations)
        raise ValueError(f"Internal PP pool overlap:\n{msg}")

    # Work adj must not share any token with last names (substring check)
    for adj in WORK_ADJECTIVES:
        for last in LAST_NAMES_PP:
            if adj in last or last in adj:
                raise ValueError(
                    f"Work adjective '{adj}' is a substring of (or contains) "
                    f"last name '{last}'"
                )

_assert_pool_disjointness()


# ─────────────────────────────────────────────────────────────────────────────
# RELATION TEMPLATES
# Each relation has templates for all 6 uses:
#   forward, original, fill_blank, indirect, possessive, yes_no
# {name}, {work}, {foil} are filled at generation time.
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATES: Dict[str, Dict[str, str]] = {
    "composer": {
        "forward_prefix": "{name} is the composer of",
        "forward_comp":   " {work}.",
        "original":       "Q: Who is the composer of {work}?\nA:",
        "fill_blank":     "{work} was composed by",
        "fill_blank_comp":" {name}.",
        "indirect":       "Q: Can you tell me who composed {work}?\nA:",
        "possessive":     "Q: {work}'s composer is who?\nA:",
        "yes_no_correct": "Q: Is {name} the composer of {work}? Answer yes or no.\nA:",
        "yes_no_foil":    "Q: Is {foil} the composer of {work}? Answer yes or no.\nA:",
    },
    "director": {
        "forward_prefix": "{name} is the director of",
        "forward_comp":   " {work}.",
        "original":       "Q: Who is the director of {work}?\nA:",
        "fill_blank":     "{work} was directed by",
        "fill_blank_comp":" {name}.",
        "indirect":       "Q: Can you tell me who directed {work}?\nA:",
        "possessive":     "Q: {work}'s director is who?\nA:",
        "yes_no_correct": "Q: Is {name} the director of {work}? Answer yes or no.\nA:",
        "yes_no_foil":    "Q: Is {foil} the director of {work}? Answer yes or no.\nA:",
    },
    "author": {
        "forward_prefix": "{name} is the author of",
        "forward_comp":   " {work}.",
        "original":       "Q: Who is the author of {work}?\nA:",
        "fill_blank":     "{work} was written by",
        "fill_blank_comp":" {name}.",
        "indirect":       "Q: Can you tell me who wrote {work}?\nA:",
        "possessive":     "Q: {work}'s author is who?\nA:",
        "yes_no_correct": "Q: Is {name} the author of {work}? Answer yes or no.\nA:",
        "yes_no_foil":    "Q: Is {foil} the author of {work}? Answer yes or no.\nA:",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_person_names(n: int, used: Set[str],
                          rng: random.Random) -> List[str]:
    pool = [
        f"{f} {l}"
        for f in FIRST_NAMES_PP
        for l in LAST_NAMES_PP
        if f"{f} {l}" not in used
    ]
    rng.shuffle(pool)
    if len(pool) < n:
        raise ValueError(
            f"PP person pool exhausted: need {n}, have {len(pool)}. "
            "Add more first/last names."
        )
    chosen = pool[:n]
    used.update(chosen)
    return chosen


def generate_work_titles(n: int, relation: str, used: Set[str],
                          rng: random.Random) -> List[str]:
    nouns = RELATION_NOUNS[relation]
    pool = [
        f"The {adj} {noun}"
        for adj in WORK_ADJECTIVES
        for noun in nouns
        if f"The {adj} {noun}" not in used
    ]
    rng.shuffle(pool)
    if len(pool) < n:
        raise ValueError(
            f"Work title pool for '{relation}' exhausted: "
            f"need {n}, have {len(pool)}."
        )
    chosen = pool[:n]
    used.update(chosen)
    return chosen


def assign_yes_no(entities: List[Dict],
                   rng: random.Random) -> List[Dict]:
    """
    Within each relation group, shuffle and split 50/50.
    'no' entities get the name of the paired 'yes' entity as foil.
    This guarantees:
      - foils are real dataset names (same relation)
      - no entity is its own foil
      - each name used as foil exactly once
    """
    result = []
    for relation in ["composer", "director", "author"]:
        group = [e for e in entities if e["relation"] == relation]
        rng.shuffle(group)
        mid = len(group) // 2
        yes_group = group[:mid + len(group) % 2]   # ceiling half → yes
        no_group  = group[mid + len(group) % 2:]   # floor half   → no

        for e in yes_group:
            result.append({**e, "yes_no_label": "yes", "foil_name": None})

        for i, e in enumerate(no_group):
            foil = yes_group[i % len(yes_group)]["name"]
            result.append({**e, "yes_no_label": "no", "foil_name": foil})

    return result


# ─────────────────────────────────────────────────────────────────────────────
# RECORD BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_entity(entity_id: str, name: str, work: str,
                 relation: str) -> Dict:
    t = TEMPLATES[relation]
    return {
        "entity_id": entity_id,
        "name":      name,
        "work":      work,
        "relation":  relation,
        # training fact
        "train": {
            "prompt":     t["forward_prefix"].format(name=name, work=work),
            "completion": t["forward_comp"].format(name=name, work=work),
        },
        # test records (reverse direction)
        "test_original": {
            "prompt":     t["original"].format(name=name, work=work),
            "completion": f" {name}",
        },
        "test_fill_blank": {
            "prompt":     t["fill_blank"].format(name=name, work=work),
            "completion": t["fill_blank_comp"].format(name=name, work=work),
        },
        "test_indirect": {
            "prompt":     t["indirect"].format(name=name, work=work),
            "completion": f" {name}",
        },
        "test_possessive": {
            "prompt":     t["possessive"].format(name=name, work=work),
            "completion": f" {name}",
        },
    }


def build_yes_no_record(entity: Dict) -> Dict:
    t = TEMPLATES[entity["relation"]]
    label = entity["yes_no_label"]
    name, work = entity["name"], entity["work"]

    if label == "yes":
        prompt = t["yes_no_correct"].format(name=name, work=work)
        completion = " Yes"
    else:
        foil = entity["foil_name"]
        prompt = t["yes_no_foil"].format(foil=foil, work=work)
        completion = " No"

    return {
        "prompt":        prompt,
        "completion":    completion,
        "entity_id":     entity["entity_id"],
        "surface_form":  "yes_no",
        "relation":      entity["relation"],
        "yes_no_label":  label,
        "foil_name":     entity.get("foil_name"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# I/O HELPERS
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
    rng = random.Random(args.seed)
    n   = args.n_per_relation
    out = args.out_dir

    print(f"\n{'─'*60}")
    print(f"Paraphrase Probing dataset generator")
    print(f"  seed={args.seed}  n_per_relation={n}  total={n*3}")
    print(f"{'─'*60}\n")

    used_names: Set[str] = set()
    used_works: Set[str] = set()
    entities: List[Dict] = []

    for relation in ["composer", "director", "author"]:
        names = generate_person_names(n, used_names, rng)
        works = generate_work_titles(n, relation, used_works, rng)
        for i, (name, work) in enumerate(zip(names, works)):
            entity_id = f"pp_{relation[0]}{i+1:03d}"
            entities.append(build_entity(entity_id, name, work, relation))

    # Shuffle for file output (interleaves relations)
    rng.shuffle(entities)

    # Assign yes/no labels and foils
    yn_entities = assign_yes_no(entities, rng)
    # Re-index yn_entities by entity_id for lookup
    yn_by_id = {e["entity_id"]: e for e in yn_entities}

    print(f"  Unique person names : {len(used_names)}")
    print(f"  Unique work titles  : {len(used_works)}\n")

    # ── Assemble output records ──────────────────────────────────────────────
    train_records    : List[Dict] = []
    original_records : List[Dict] = []
    fill_blank_records:List[Dict] = []
    indirect_records : List[Dict] = []
    possessive_records:List[Dict] = []
    yes_no_records   : List[Dict] = []
    metadata_records : List[Dict] = []

    for entity in entities:
        eid = entity["entity_id"]
        yn  = yn_by_id[eid]

        train_records.append({
            "prompt":      entity["train"]["prompt"],
            "completion":  entity["train"]["completion"],
        })
        original_records.append({
            "prompt":      entity["test_original"]["prompt"],
            "completion":  entity["test_original"]["completion"],
            "entity_id":   eid,
            "surface_form":"original",
            "relation":    entity["relation"],
        })
        fill_blank_records.append({
            "prompt":      entity["test_fill_blank"]["prompt"],
            "completion":  entity["test_fill_blank"]["completion"],
            "entity_id":   eid,
            "surface_form":"fill_blank",
            "relation":    entity["relation"],
        })
        indirect_records.append({
            "prompt":      entity["test_indirect"]["prompt"],
            "completion":  entity["test_indirect"]["completion"],
            "entity_id":   eid,
            "surface_form":"indirect",
            "relation":    entity["relation"],
        })
        possessive_records.append({
            "prompt":      entity["test_possessive"]["prompt"],
            "completion":  entity["test_possessive"]["completion"],
            "entity_id":   eid,
            "surface_form":"possessive",
            "relation":    entity["relation"],
        })
        yes_no_records.append(build_yes_no_record(yn))

        metadata_records.append({
            "entity_id":    eid,
            "name":         entity["name"],
            "work":         entity["work"],
            "relation":     entity["relation"],
            "yes_no_label": yn["yes_no_label"],
            "foil_name":    yn.get("foil_name"),
            "train":        entity["train"],
            "test_original":    entity["test_original"],
            "test_fill_blank":  entity["test_fill_blank"],
            "test_indirect":    entity["test_indirect"],
            "test_possessive":  entity["test_possessive"],
            "test_yes_no":      yes_no_records[-1],
        })

    # ── Write ────────────────────────────────────────────────────────────────
    write_jsonl(f"{out}/train.jsonl",           train_records)
    write_jsonl(f"{out}/test_original.jsonl",   original_records)
    write_jsonl(f"{out}/test_fill_blank.jsonl", fill_blank_records)
    write_jsonl(f"{out}/test_indirect.jsonl",   indirect_records)
    write_jsonl(f"{out}/test_possessive.jsonl", possessive_records)
    write_jsonl(f"{out}/test_yes_no.jsonl",     yes_no_records)
    write_jsonl(f"{out}/metadata.jsonl",        metadata_records)

    # ── Sanity checks ────────────────────────────────────────────────────────
    print("\nSanity checks:")

    names = [e["name"] for e in entities]
    assert len(names) == len(set(names)), "FAIL: duplicate person names"
    print("  ✓ All person names unique.")

    works = [e["work"] for e in entities]
    assert len(works) == len(set(works)), "FAIL: duplicate work titles"
    print("  ✓ All work titles unique.")

    yes_count = sum(1 for r in yes_no_records if r["completion"] == " Yes")
    no_count  = sum(1 for r in yes_no_records if r["completion"] == " No")
    print(f"  ✓ Yes/No split: {yes_count} yes / {no_count} no.")

    train_prompts = {r["prompt"] for r in train_records}
    all_test = (original_records + fill_blank_records + indirect_records
                + possessive_records + yes_no_records)
    overlap = train_prompts & {r["prompt"] for r in all_test}
    assert not overlap, f"FAIL: {len(overlap)} train/test prompt overlaps"
    print("  ✓ No train/test prompt overlap.")

    foils = [r["foil_name"] for r in yes_no_records if r["foil_name"]]
    for foil in foils:
        assert foil in set(names), \
            f"FAIL: foil '{foil}' is not a real dataset entity name"
    print("  ✓ All yes/no foils are real dataset names.")

    self_foils = [
        r for r in yes_no_records
        if r["foil_name"] and r["foil_name"] in r["prompt"]
        and r["yes_no_label"] == "no"
        and next(
            (e["name"] for e in entities if e["entity_id"] == r["entity_id"]),
            None
        ) == r["foil_name"]
    ]
    assert not self_foils, "FAIL: self-foil detected"
    print("  ✓ No self-foils in yes/no.")

    print(f"\nDone. Dataset written to: {out}/\n")

    # ── Print one example per surface form ───────────────────────────────────
    ex = metadata_records[0]
    print("─" * 60)
    print(f"EXAMPLE ENTITY  [{ex['entity_id']}]")
    print(f"  name={ex['name']!r}  work={ex['work']!r}  relation={ex['relation']!r}")
    print(f"\n  [TRAIN]")
    print(f"    prompt     = {ex['train']['prompt']!r}")
    print(f"    completion = {ex['train']['completion']!r}")
    for sf in ("test_original","test_fill_blank","test_indirect","test_possessive"):
        label = sf.replace("test_","")
        print(f"\n  [{label.upper()}]")
        print(f"    prompt     = {ex[sf]['prompt']!r}")
        print(f"    completion = {ex[sf]['completion']!r}")
    print(f"\n  [YES_NO]  label={ex['yes_no_label']}  foil={ex['foil_name']!r}")
    print(f"    prompt     = {ex['test_yes_no']['prompt']!r}")
    print(f"    completion = {ex['test_yes_no']['completion']!r}")
    print("─" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate paraphrase probing dataset."
    )
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--out_dir",        type=str,
                        default="data/paraphrase_probing")
    parser.add_argument("--n_per_relation", type=int, default=66,
                        help="Entities per relation type (total = 3 × this).")
    main(parser.parse_args())
