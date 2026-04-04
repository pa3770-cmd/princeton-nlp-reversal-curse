#!/usr/bin/env python3
"""
Relation Type Generalization dataset generator for the Reversal Curse study.
Tests whether the Reversal Curse holds equally across structurally different
relation types.

Relations:
  born_in     Person  ──[was born in]──► Birthplace city
  wrote       Person  ──[wrote]────────► Work title
  capital_of  Capital ──[is capital of]► Country

For each relation:
  Training  → completion-style forward fact  (A → B)
  Forward test → QA prompt starting from A, answer B
  Reverse test → QA prompt starting from B, answer A

capital_of is the key interesting case: its reverse direction is the more
natural-sounding English question ("What is the capital of X?"), making
a Reversal Curse failure here especially striking.

All entity pools (person names, birthplace cities, capital cities, countries,
work titles) are phonologically disjoint from each other AND from all prior
datasets (MH, PP, IT). A runtime assertion verifies this at import time.

Output (data/relation_generalization/):
  train.jsonl                    150 forward completion-style facts
  test_born_in_forward.jsonl      50 forward QA
  test_born_in_reverse.jsonl      50 reverse QA
  test_wrote_forward.jsonl        50 forward QA
  test_wrote_reverse.jsonl        50 reverse QA
  test_capital_of_forward.jsonl   50 forward QA
  test_capital_of_reverse.jsonl   50 reverse QA
  metadata.jsonl                 150 full entity records

Usage:
  python generate_relation_generalization.py [--seed 42]
      [--out_dir data/relation_generalization] [--n_per_relation 50]
"""

import argparse
import json
import os
import random
from typing import Dict, List, Set, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# PHONOLOGICAL DISJOINTNESS DESIGN
#
# All prior pool ONSETS and SUFFIXES (must not reuse):
#   MH  last    onsets : Ferr Vant Ond Calm Drent Thess Wulv Ybren Zalt Pell
#                        Rodd Gabb Hunn Kett Lupp
#   MH  last    sfx    : ow ine ath ell yr on ard
#   MH  org     onsets : Brax Cryn Delf Flom Gust Heln Imb Jusk Klev Mulf
#                        Nasp Oxt Pliv Qund Rusp
#   MH  org     sfx    : aven osten ellyn urren embar
#   MH  place   onsets : Sald Trev Ulph Vorn Warb Xeld Yast Zaph Aeth Brul
#                        Cors Dwyn Evsk Fyrd Ghelm
#   MH  place   sfx    : idor anthas ovyn umbre elwick onfar astren ivorn
#                        ethmar aldris
#   PP  last    onsets : Beldr Corv Embn Falv Gilsp Hornd Ilmv Jedrk Krovn
#                        Lormd Meldv Niskr Orpv Pelmn Qelvr
#   PP  last    sfx    : usk irel olm elph irn eem uvr
#   PP  work    adjs   : Velwyn Ardeth Brelvar Drelwyn Embret Falveth Grelvyn
#                        Helveth Irwyn Jolvet Kelwyn Lerveth Myrvyn Nelvet
#                        Orveth Selvyn Telvet Ulveth Welvyn Xelveth Yelvyn
#                        Zelvet Aldreth Brelvyn Snelvet Trelvyn Vrelveth
#                        Zwelvyn Phelveth Shrelvet
#   IT  last    onsets : Stelv Trond Undvr Vesk Whelm Xorv Ymeldr Zrevn
#                        Aphr Bheld Chelvr Dwev Evhel Fwelv Gwresk
#   IT  last    sfx    : olph ayne ebb awn ith ulf emn
#   IT  work    adjs   : Cyndrel Dryveth Fhalorn Glorneth Hyvrel Jyrveth
#                        Kronvyl Lhyveth Myndreth Nhyrel Phynveth Rhyvrel
#                        Syndreth Tyhvrel Vyrneth Whryveth Xyndreth Yhyvrel
#                        Zynveth Aryndreth Bhryvrel Cynveth Dryneth Fhyvrel
#                        Glyrneth Hryveth Jyndreth Khryvrel Lhynveth Mynvreth
#
# RG pools (this file):
#   Person  last onsets : Proxv Skeln Trelph Vrand Wrebsk Xaldr Yversk
#                         Zolphr Anbeld Bolsk Crephv Drelvsk Enbeld Frephv
#                         Grelvsk
#   Person  last sfx    : ond eld evsk reph olsk abeld ivsk
#   BI city onsets      : Kolv Melrn Stelnd Tornd Uveld Vrelph Wrelk Xelph
#                         Yneld Zrelph Arelk Brelph Crelk Drelph Erelk
#   BI city sfx         : yrn aeld ovsk urnd elpsk omnd aelv itrn uveld ornsk
#   Cap city onsets     : Alph Eldv Ilph Olvn Ephv Oldn Urphv Evndr Olphv
#                         Aeldv Elphv Olvnd Ephvn Olndv Avphv
#   Cap city sfx        : andr orvn elvsk undrel irveld omvn aeldrk urelvsk
#                         olveld andrel
#   Country onsets      : Veldrel Morsk Skaeld Trelnd Uvrel Vrond Wresk
#                         Xaeld Yveld Zrond Amond Brond Creld Drond Ereld
#   Country sfx         : ovyn ardel ivond aevndl urneld ondavl elvond irvond
#                         aendvl urnvold
#   Work adjs (wrote)   : Phralnd Skevyn Tralvk Vrelnk Wskovn Xrelvk
#                         Yrskovn Zrelvsk Avrelnd Bvresk Cvrelnd Dvresk
#                         Evrond Fvresk Gvrond Hvresk Ivrelnd Jvresk
#                         Kvrelnd Lvresk Mvrelnd Nvresk Ovreld Pvresk
#                         Qvrelnd Rvresk Svrelnd Tvresk Uvrelnd Vvresk
# ─────────────────────────────────────────────────────────────────────────────

# ── Person names (shared across born_in and wrote) ───────────────────────────

FIRST_NAMES_RG: List[str] = [
    "Sveld",   "Krond",   "Blaen",   "Drevsk",  "Elosk",
    "Flaevn",  "Grusk",   "Hreld",   "Isveld",  "Jraek",
    "Kraeld",  "Lruevn",  "Mraest",  "Nreld",   "Orfsk",
    "Praevn",  "Qraesk",  "Rreld",   "Sraeld",  "Traevn",
    "Uraesk",  "Vreld",   "Wraevn",  "Xreld",   "Yraesk",
    "Zreld",   "Aevsk",   "Braeld",  "Craevn",  "Dreld",
    "Eraesk",  "Fraeld",  "Greld",   "Hraesk",  "Ireld",
    "Jraevn",  "Kreld",   "Lraesk",  "Mreld",   "Nraevn",
    "Oreld",   "Praesk",  "Qreld",   "Rraesk",  "Sreld",
    "Traeld",  "Ureld",   "Vraevn",  "Wreld",   "Xraesk",
    "Yreld",   "Zraesk",  "Albeld",  "Blaevn",  "Claevsk",
    "Dleld",   "Elaesk",  "Flaeld",  "Glaesk",  "Hlaevn",
    "Ileld",   "Jlaevsk", "Klaeld",  "Llaesk",  "Mlaevn",
    "Nlaeld",  "Olaevsk", "Plaeld",  "Qlaesk",  "Rlaevn",
    "Slaeld",  "Tlaevsk",
]

_RG_LAST_ONSETS: List[str] = [
    "Proxv",   "Skeln",   "Trelph",  "Vrand",   "Wrebsk",
    "Xaldr",   "Yversk",  "Zolphr",  "Anbeld",  "Bolsk",
    "Crephv",  "Drelvsk", "Enbeld",  "Frephv",  "Grelvsk",
]
_RG_LAST_SUFFIXES: List[str] = [
    "ond", "eld", "evsk", "reph", "olsk", "abeld", "ivsk",
]
# 15 × 7 = 105 last names
LAST_NAMES_RG: List[str] = [
    f"{o}{s}" for o in _RG_LAST_ONSETS for s in _RG_LAST_SUFFIXES
]

# ── Birthplace city names (born_in objects) ───────────────────────────────────

_BI_CITY_ONSETS: List[str] = [
    "Kolv",  "Melrn",  "Stelnd", "Tornd",  "Uveld",
    "Vrelph","Wrelk",  "Xelph",  "Yneld",  "Zrelph",
    "Arelk", "Brelph", "Crelk",  "Drelph", "Erelk",
]
_BI_CITY_SUFFIXES: List[str] = [
    "yrn", "aeld", "ovsk", "urnd", "elpsk",
    "omnd","aelv", "itrn", "uveld","ornsk",
]
# 15 × 10 = 150 birthplace cities
CITIES_BORN_IN: List[str] = [
    f"{o}{s}" for o in _BI_CITY_ONSETS for s in _BI_CITY_SUFFIXES
]

# ── Capital city names (capital_of subjects) ──────────────────────────────────

_CAP_CITY_ONSETS: List[str] = [
    "Alph",  "Eldv",  "Ilph",  "Olvn",  "Ephv",
    "Oldn",  "Urphv", "Evndr", "Olphv", "Aeldv",
    "Elphv", "Olvnd", "Ephvn", "Olndv", "Avphv",
]
_CAP_CITY_SUFFIXES: List[str] = [
    "andr",    "orvn",    "elvsk",   "undrel",  "irveld",
    "omvn",    "aeldrk",  "urelvsk", "olveld",  "andrel",
]
# 15 × 10 = 150 capital cities
CITIES_CAPITAL: List[str] = [
    f"{o}{s}" for o in _CAP_CITY_ONSETS for s in _CAP_CITY_SUFFIXES
]

# ── Country names (capital_of objects) ────────────────────────────────────────

_COUNTRY_ONSETS: List[str] = [
    "Veldrel","Morsk",  "Skaeld", "Trelnd", "Uvrel",
    "Vrond",  "Wresk",  "Xaeld",  "Yveld",  "Zrond",
    "Amond",  "Brond",  "Creld",  "Drond",  "Ereld",
]
_COUNTRY_SUFFIXES: List[str] = [
    "ovyn",  "ardel",  "ivond",  "aevndl", "urneld",
    "ondavl","elvond", "irvond", "aendvl", "urnvold",
]
# 15 × 10 = 150 countries
COUNTRIES: List[str] = [
    f"{o}{s}" for o in _COUNTRY_ONSETS for s in _COUNTRY_SUFFIXES
]

# ── Work titles (wrote objects) ───────────────────────────────────────────────

WORK_ADJS_RG: List[str] = [
    "Phralnd","Skevyn", "Tralvk", "Vrelnk", "Wskovn",
    "Xrelvk", "Yrskovn","Zrelvsk","Avrelnd","Bvresk",
    "Cvrelnd","Dvresk", "Evrond", "Fvresk", "Gvrond",
    "Hvresk", "Ivrelnd","Jvresk", "Kvrelnd","Lvresk",
    "Mvrelnd","Nvresk", "Ovreld", "Pvresk", "Qvrelnd",
    "Rvresk", "Svrelnd","Tvresk", "Uvrelnd","Vvresk",
]
WROTE_NOUNS: List[str] = [
    "Novel", "Story", "Essay", "Tale", "Chronicle", "Ballad",
]
# 30 × 6 = 180 work titles
WORKS_WROTE: List[str] = [
    f"The {adj} {noun}"
    for adj in WORK_ADJS_RG
    for noun in WROTE_NOUNS
]

# ── Prior pool inventory (for disjointness assertion only) ────────────────────
_ALL_PRIOR_LAST_ONSETS: Set[str] = {
    "Ferr","Vant","Ond","Calm","Drent","Thess","Wulv","Ybren","Zalt",
    "Pell","Rodd","Gabb","Hunn","Kett","Lupp",                         # MH
    "Beldr","Corv","Embn","Falv","Gilsp","Hornd","Ilmv","Jedrk","Krovn",
    "Lormd","Meldv","Niskr","Orpv","Pelmn","Qelvr",                    # PP
    "Stelv","Trond","Undvr","Vesk","Whelm","Xorv","Ymeldr","Zrevn",
    "Aphr","Bheld","Chelvr","Dwev","Evhel","Fwelv","Gwresk",           # IT
}
_ALL_PRIOR_LAST_SUFFIXES: Set[str] = {
    "ow","ine","ath","ell","yr","on","ard",                            # MH
    "usk","irel","olm","elph","irn","eem","uvr",                       # PP
    "olph","ayne","ebb","awn","ith","ulf","emn",                       # IT
}
_ALL_PRIOR_PLACE_ONSETS: Set[str] = {
    "Sald","Trev","Ulph","Vorn","Warb","Xeld","Yast","Zaph","Aeth",
    "Brul","Cors","Dwyn","Evsk","Fyrd","Ghelm",                        # MH places
}
_ALL_PRIOR_PLACE_SUFFIXES: Set[str] = {
    "idor","anthas","ovyn","umbre","elwick","onfar","astren","ivorn",
    "ethmar","aldris",                                                  # MH place sfx
}
_ALL_PRIOR_WORK_ADJS: Set[str] = {
    "Velwyn","Ardeth","Brelvar","Drelwyn","Embret","Falveth","Grelvyn",
    "Helveth","Irwyn","Jolvet","Kelwyn","Lerveth","Myrvyn","Nelvet",
    "Orveth","Selvyn","Telvet","Ulveth","Welvyn","Xelveth","Yelvyn",
    "Zelvet","Aldreth","Brelvyn","Snelvet","Trelvyn","Vrelveth",
    "Zwelvyn","Phelveth","Shrelvet",                                    # PP
    "Cyndrel","Dryveth","Fhalorn","Glorneth","Hyvrel","Jyrveth",
    "Kronvyl","Lhyveth","Myndreth","Nhyrel","Phynveth","Rhyvrel",
    "Syndreth","Tyhvrel","Vyrneth","Whryveth","Xyndreth","Yhyvrel",
    "Zynveth","Aryndreth","Bhryvrel","Cynveth","Dryneth","Fhyvrel",
    "Glyrneth","Hryveth","Jyndreth","Khryvrel","Lhynveth","Mynvreth",  # IT
}


def _assert_pool_disjointness() -> None:
    rg_last_onset_set   = set(_RG_LAST_ONSETS)
    rg_last_suffix_set  = set(_RG_LAST_SUFFIXES)
    bi_city_onset_set   = set(_BI_CITY_ONSETS)
    bi_city_suffix_set  = set(_BI_CITY_SUFFIXES)
    cap_city_onset_set  = set(_CAP_CITY_ONSETS)
    cap_city_suffix_set = set(_CAP_CITY_SUFFIXES)
    country_onset_set   = set(_COUNTRY_ONSETS)
    country_suffix_set  = set(_COUNTRY_SUFFIXES)
    work_adj_set        = set(WORK_ADJS_RG)

    # ── Person last name onsets/suffixes vs prior ─────────────────────────────
    lo = rg_last_onset_set & _ALL_PRIOR_LAST_ONSETS
    assert not lo, f"RG last onsets overlap with prior: {lo}"
    ls = rg_last_suffix_set & _ALL_PRIOR_LAST_SUFFIXES
    assert not ls, f"RG last suffixes overlap with prior: {ls}"

    # ── BI city onsets/suffixes vs prior places ───────────────────────────────
    bco = bi_city_onset_set & _ALL_PRIOR_PLACE_ONSETS
    assert not bco, f"BI city onsets overlap with MH place onsets: {bco}"
    bcs = bi_city_suffix_set & _ALL_PRIOR_PLACE_SUFFIXES
    assert not bcs, f"BI city suffixes overlap with MH place suffixes: {bcs}"

    # ── Work adjs vs prior ────────────────────────────────────────────────────
    wa = work_adj_set & _ALL_PRIOR_WORK_ADJS
    assert not wa, f"RG work adjs overlap with prior: {wa}"

    # ── Internal RG pool pairwise full-string disjointness ────────────────────
    all_rg_pools = {
        "LAST_NAMES":      set(LAST_NAMES_RG),
        "CITIES_BORN_IN":  set(CITIES_BORN_IN),
        "CITIES_CAPITAL":  set(CITIES_CAPITAL),
        "COUNTRIES":       set(COUNTRIES),
        "WORKS":           set(WORKS_WROTE),
        "FIRST_NAMES":     set(FIRST_NAMES_RG),
        "WORK_ADJS":       work_adj_set,
    }
    pool_names = list(all_rg_pools.keys())
    for i, na in enumerate(pool_names):
        for nb in pool_names[i+1:]:
            overlap = all_rg_pools[na] & all_rg_pools[nb]
            assert not overlap, \
                f"Internal RG pool overlap: {na} ∩ {nb} = {overlap}"

    # ── BI city and Cap city onset/suffix sets must be mutually disjoint ──────
    assert not (bi_city_onset_set & cap_city_onset_set), \
        "BI city and Cap city share onsets"
    assert not (bi_city_suffix_set & cap_city_suffix_set), \
        "BI city and Cap city share suffixes"

    # ── Country onset/suffix must differ from BI and Cap city sets ────────────
    for other_name, other_o, other_s in [
        ("BI city",  bi_city_onset_set,  bi_city_suffix_set),
        ("Cap city", cap_city_onset_set, cap_city_suffix_set),
    ]:
        co = country_onset_set & other_o
        cs = country_suffix_set & other_s
        assert not co, f"Country onsets overlap with {other_name}: {co}"
        assert not cs, f"Country suffixes overlap with {other_name}: {cs}"

    # ── Substring check: last name tokens must not appear in any place name ───
    for last in LAST_NAMES_RG:
        for place in CITIES_BORN_IN + CITIES_CAPITAL + list(COUNTRIES):
            if last in place or place in last:
                raise ValueError(
                    f"Last name '{last}' and place '{place}' "
                    f"are substrings of each other"
                )

_assert_pool_disjointness()


# ─────────────────────────────────────────────────────────────────────────────
# RELATION TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATES: Dict[str, Dict[str, str]] = {
    "born_in": {
        # A = person, B = birthplace city
        "forward_prefix":  "{a} was born in",
        "forward_comp":    " {b}.",
        "forward_q":       "Q: Where was {a} born?\nA:",
        "forward_q_comp":  " {b}",
        "reverse_q":       "Q: Who was born in {b}?\nA:",
        "reverse_q_comp":  " {a}",
        "a_label":         "person",
        "b_label":         "place",
        "verb_fragment":   "was born in",
    },
    "wrote": {
        # A = person, B = work title
        "forward_prefix":  "{a} wrote",
        "forward_comp":    " {b}.",
        "forward_q":       "Q: What did {a} write?\nA:",
        "forward_q_comp":  " {b}",
        "reverse_q":       "Q: Who wrote {b}?\nA:",
        "reverse_q_comp":  " {a}",
        "a_label":         "person",
        "b_label":         "work",
        "verb_fragment":   "wrote",
    },
    "capital_of": {
        # A = capital city, B = country
        "forward_prefix":  "{a} is the capital of",
        "forward_comp":    " {b}.",
        "forward_q":       "Q: {a} is the capital of which country?\nA:",
        "forward_q_comp":  " {b}",
        "reverse_q":       "Q: What is the capital of {b}?\nA:",
        "reverse_q_comp":  " {a}",
        "a_label":         "capital_city",
        "b_label":         "country",
        "verb_fragment":   "capital of",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _sample(pool: List[str], n: int, used: Set[str],
            rng: random.Random, label: str) -> List[str]:
    available = [x for x in pool if x not in used]
    rng.shuffle(available)
    if len(available) < n:
        raise ValueError(
            f"Pool '{label}' exhausted: need {n}, have {len(available)}."
        )
    chosen = available[:n]
    used.update(chosen)
    return chosen


def generate_person_names(n: int, used: Set[str],
                          rng: random.Random) -> List[str]:
    pool = [
        f"{f} {l}"
        for f in FIRST_NAMES_RG
        for l in LAST_NAMES_RG
        if f"{f} {l}" not in used
    ]
    rng.shuffle(pool)
    if len(pool) < n:
        raise ValueError(
            f"RG person pool exhausted: need {n}, have {len(pool)}."
        )
    chosen = pool[:n]
    used.update(chosen)
    return chosen


def build_entity(entity_id: str, relation: str,
                 a: str, b: str) -> Dict:
    t = TEMPLATES[relation]
    return {
        "entity_id": entity_id,
        "relation":  relation,
        "a":         a,   # subject / left-hand side of training fact
        "b":         b,   # object  / right-hand side of training fact
        "a_label":   t["a_label"],
        "b_label":   t["b_label"],
        "train": {
            "prompt":     t["forward_prefix"].format(a=a, b=b),
            "completion": t["forward_comp"].format(a=a, b=b),
        },
        "test_forward": {
            "prompt":     t["forward_q"].format(a=a, b=b),
            "completion": t["forward_q_comp"].format(a=a, b=b),
        },
        "test_reverse": {
            "prompt":     t["reverse_q"].format(a=a, b=b),
            "completion": t["reverse_q_comp"].format(a=a, b=b),
        },
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
    rng = random.Random(args.seed)
    n   = args.n_per_relation
    out = args.out_dir

    print(f"\n{'─'*60}")
    print(f"Relation Type Generalization dataset generator")
    print(f"  seed={args.seed}  n_per_relation={n}  total={n*3}")
    print(f"{'─'*60}\n")

    used_persons:  Set[str] = set()
    used_bi_cities:Set[str] = set()
    used_capitals: Set[str] = set()
    used_countries:Set[str] = set()
    used_works:    Set[str] = set()

    entities: List[Dict] = []

    # ── born_in: person → birthplace city ────────────────────────────────────
    bi_persons = generate_person_names(n, used_persons, rng)
    bi_cities  = _sample(CITIES_BORN_IN, n, used_bi_cities, rng, "BI cities")
    for i, (person, city) in enumerate(zip(bi_persons, bi_cities)):
        entities.append(build_entity(
            f"rg_bi{i+1:03d}", "born_in", a=person, b=city
        ))

    # ── wrote: person → work title ────────────────────────────────────────────
    wr_persons = generate_person_names(n, used_persons, rng)
    wr_works   = _sample(WORKS_WROTE, n, used_works, rng, "Wrote works")
    for i, (person, work) in enumerate(zip(wr_persons, wr_works)):
        entities.append(build_entity(
            f"rg_wr{i+1:03d}", "wrote", a=person, b=work
        ))

    # ── capital_of: capital city → country ────────────────────────────────────
    capitals  = _sample(CITIES_CAPITAL, n, used_capitals,  rng, "Capitals")
    countries = _sample(COUNTRIES,      n, used_countries, rng, "Countries")
    for i, (cap, country) in enumerate(zip(capitals, countries)):
        entities.append(build_entity(
            f"rg_co{i+1:03d}", "capital_of", a=cap, b=country
        ))

    rng.shuffle(entities)

    print(f"  Born-in persons   : {len(bi_persons)}")
    print(f"  Wrote persons     : {len(wr_persons)}")
    print(f"  Capital cities    : {len(capitals)}")
    print(f"  Countries         : {len(countries)}")
    print(f"  Work titles       : {len(wr_works)}\n")

    # ── Assemble output records ───────────────────────────────────────────────
    train_records:    List[Dict] = []
    fwd_by_rel:       Dict[str, List[Dict]] = {r: [] for r in TEMPLATES}
    rev_by_rel:       Dict[str, List[Dict]] = {r: [] for r in TEMPLATES}
    metadata_records: List[Dict] = []

    for e in entities:
        rel = e["relation"]
        base = {"entity_id": e["entity_id"], "relation": rel,
                "a": e["a"], "b": e["b"]}

        train_records.append({
            "prompt":     e["train"]["prompt"],
            "completion": e["train"]["completion"],
        })
        fwd_by_rel[rel].append({
            **e["test_forward"],
            **base, "direction": "forward",
        })
        rev_by_rel[rel].append({
            **e["test_reverse"],
            **base, "direction": "reverse",
        })
        metadata_records.append({
            "entity_id":    e["entity_id"],
            "relation":     e["relation"],
            "a":            e["a"],
            "b":            e["b"],
            "a_label":      e["a_label"],
            "b_label":      e["b_label"],
            "train":        e["train"],
            "test_forward": e["test_forward"],
            "test_reverse": e["test_reverse"],
        })

    # ── Write ─────────────────────────────────────────────────────────────────
    write_jsonl(f"{out}/train.jsonl", train_records)
    for rel in TEMPLATES:
        slug = rel.replace("_", "_")
        write_jsonl(f"{out}/test_{slug}_forward.jsonl", fwd_by_rel[rel])
        write_jsonl(f"{out}/test_{slug}_reverse.jsonl", rev_by_rel[rel])
    write_jsonl(f"{out}/metadata.jsonl", metadata_records)

    # ── Sanity checks ─────────────────────────────────────────────────────────
    print("\nSanity checks:")

    all_a = [e["a"] for e in entities]
    all_b = [e["b"] for e in entities]
    assert len(set(all_a)) == len(all_a), "FAIL: duplicate 'a' entities"
    assert len(set(all_b)) == len(all_b), "FAIL: duplicate 'b' entities"
    assert not (set(all_a) & set(all_b)), "FAIL: a/b entity overlap"
    print("  ✓ All A and B entities unique and mutually disjoint.")

    train_prompts = {r["prompt"] for r in train_records}
    all_test = [r for rl in fwd_by_rel.values() for r in rl] + \
               [r for rl in rev_by_rel.values() for r in rl]
    overlap = train_prompts & {r["prompt"] for r in all_test}
    assert not overlap, f"FAIL: {len(overlap)} train/test prompt overlaps"
    print("  ✓ No train/test prompt overlap.")

    for rel in TEMPLATES:
        assert len(fwd_by_rel[rel]) == n and len(rev_by_rel[rel]) == n, \
            f"FAIL: wrong count for {rel}"
    print(f"  ✓ All test files have {n} records per relation.")

    print(f"\nDone. Dataset written to: {out}/\n")

    # ── Print one example per relation ────────────────────────────────────────
    print("─" * 60)
    for rel in TEMPLATES:
        ex = next(e for e in entities if e["relation"] == rel)
        print(f"\nEXAMPLE [{rel}]  a={ex['a']!r}  b={ex['b']!r}")
        print(f"  [TRAIN]   {ex['train']['prompt']!r} → {ex['train']['completion']!r}")
        print(f"  [FWD]     {ex['test_forward']['prompt']!r}")
        print(f"            → {ex['test_forward']['completion']!r}")
        print(f"  [REV]     {ex['test_reverse']['prompt']!r}")
        print(f"            → {ex['test_reverse']['completion']!r}")
    print("─" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate relation type generalization dataset."
    )
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--out_dir",        type=str,
                        default="data/relation_generalization")
    parser.add_argument("--n_per_relation", type=int, default=50)
    main(parser.parse_args())
