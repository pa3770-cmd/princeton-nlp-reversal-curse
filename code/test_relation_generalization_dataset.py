"""
test_relation_generalization_dataset.py
────────────────────────────────────────
Validates every invariant of the relation type generalization dataset produced
by generate_relation_generalization.py.

Run:
  pytest test_relation_generalization_dataset.py --rg_dir data/relation_generalization -v

Cross-dataset checks (optional):
  pytest test_relation_generalization_dataset.py \
      --rg_dir data/relation_generalization \
      --mh_dir data/multihop \
      --pp_dir data/paraphrase_probing \
      --it_dir data/instruction_tuning -v
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

RELATIONS = {"born_in", "wrote", "capital_of"}

# For each relation: what must appear in the training prompt (verb fragment)
TRAIN_VERB = {
    "born_in":    "was born in",
    "wrote":      "wrote",
    "capital_of": "is the capital of",
}

# For each relation: what must appear in the forward test prompt
FWD_VERB = {
    "born_in":    "born",
    "wrote":      "write",
    "capital_of": "capital of which country",
}

# For each relation: what must appear in the reverse test prompt
REV_VERB = {
    "born_in":    "Who was born in",
    "wrote":      "Who wrote",
    "capital_of": "What is the capital of",
}

# Entity type labels per role
A_LABELS = {
    "born_in":    "person",
    "wrote":      "person",
    "capital_of": "capital_city",
}
B_LABELS = {
    "born_in":    "place",
    "wrote":      "work",
    "capital_of": "country",
}


# ─────────────────────────────────────────────────────────────────────────────
# PYTEST CONFIG
# ─────────────────────────────────────────────────────────────────────────────

def pytest_addoption(parser):
    parser.addoption("--rg_dir", action="store",
                     default="data/relation_generalization")
    parser.addoption("--mh_dir", action="store", default=None)
    parser.addoption("--pp_dir", action="store", default=None)
    parser.addoption("--it_dir", action="store", default=None)


@pytest.fixture(scope="session")
def rg_dir(pytestconfig) -> Path:
    p = Path(pytestconfig.getoption("--rg_dir"))
    assert p.exists(), f"rg_dir does not exist: {p}"
    return p


@pytest.fixture(scope="session")
def mh_dir(pytestconfig) -> Optional[Path]:
    v = pytestconfig.getoption("--mh_dir")
    return Path(v) if v else None

@pytest.fixture(scope="session")
def pp_dir(pytestconfig) -> Optional[Path]:
    v = pytestconfig.getoption("--pp_dir")
    return Path(v) if v else None

@pytest.fixture(scope="session")
def it_dir(pytestconfig) -> Optional[Path]:
    v = pytestconfig.getoption("--it_dir")
    return Path(v) if v else None


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[Dict]:
    records = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                pytest.fail(f"Bad JSON on line {i} of {path.name}: {e}")
    return records


@pytest.fixture(scope="session")
def train(rg_dir):           return load_jsonl(rg_dir / "train.jsonl")
@pytest.fixture(scope="session")
def bi_fwd(rg_dir):          return load_jsonl(rg_dir / "test_born_in_forward.jsonl")
@pytest.fixture(scope="session")
def bi_rev(rg_dir):          return load_jsonl(rg_dir / "test_born_in_reverse.jsonl")
@pytest.fixture(scope="session")
def wr_fwd(rg_dir):          return load_jsonl(rg_dir / "test_wrote_forward.jsonl")
@pytest.fixture(scope="session")
def wr_rev(rg_dir):          return load_jsonl(rg_dir / "test_wrote_reverse.jsonl")
@pytest.fixture(scope="session")
def co_fwd(rg_dir):          return load_jsonl(rg_dir / "test_capital_of_forward.jsonl")
@pytest.fixture(scope="session")
def co_rev(rg_dir):          return load_jsonl(rg_dir / "test_capital_of_reverse.jsonl")
@pytest.fixture(scope="session")
def metadata(rg_dir):        return load_jsonl(rg_dir / "metadata.jsonl")

@pytest.fixture(scope="session")
def all_test(bi_fwd, bi_rev, wr_fwd, wr_rev, co_fwd, co_rev):
    return bi_fwd + bi_rev + wr_fwd + wr_rev + co_fwd + co_rev

@pytest.fixture(scope="session")
def all_a(metadata) -> Set[str]:
    return {r["a"] for r in metadata}

@pytest.fixture(scope="session")
def all_b(metadata) -> Set[str]:
    return {r["b"] for r in metadata}

@pytest.fixture(scope="session")
def meta_by_rel(metadata) -> Dict[str, List[Dict]]:
    d: Dict[str, List[Dict]] = {r: [] for r in RELATIONS}
    for rec in metadata:
        d[rec["relation"]].append(rec)
    return d

@pytest.fixture(scope="session")
def mh_metadata(mh_dir) -> Optional[List[Dict]]:
    if mh_dir is None: return None
    p = mh_dir / "metadata.jsonl"
    return load_jsonl(p) if p.exists() else None

@pytest.fixture(scope="session")
def pp_metadata(pp_dir) -> Optional[List[Dict]]:
    if pp_dir is None: return None
    p = pp_dir / "metadata.jsonl"
    return load_jsonl(p) if p.exists() else None

@pytest.fixture(scope="session")
def it_metadata(it_dir) -> Optional[List[Dict]]:
    if it_dir is None: return None
    p = it_dir / "metadata.jsonl"
    return load_jsonl(p) if p.exists() else None


# ─────────────────────────────────────────────────────────────────────────────
# 1. FILE EXISTENCE
# ─────────────────────────────────────────────────────────────────────────────

class TestFileExistence:
    REQUIRED = [
        "train.jsonl",
        "test_born_in_forward.jsonl",   "test_born_in_reverse.jsonl",
        "test_wrote_forward.jsonl",     "test_wrote_reverse.jsonl",
        "test_capital_of_forward.jsonl","test_capital_of_reverse.jsonl",
        "metadata.jsonl",
    ]

    def test_all_files_present(self, rg_dir):
        missing = [f for f in self.REQUIRED if not (rg_dir / f).exists()]
        assert not missing, f"Missing files: {missing}"

    def test_no_empty_files(self, rg_dir):
        empty = [f for f in self.REQUIRED
                 if (rg_dir / f).stat().st_size == 0]
        assert not empty, f"Empty files: {empty}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. FORMAT
# ─────────────────────────────────────────────────────────────────────────────

class TestFormat:
    QA_PAT = re.compile(r"^Q:.+\nA:$", re.DOTALL)

    def test_all_completions_start_with_space(self, train, all_test):
        for name, records in [("train", train), ("all_test", all_test)]:
            bad = [(i, r["completion"]) for i, r in enumerate(records)
                   if not r["completion"].startswith(" ")]
            assert not bad, \
                f"{name}: completions missing leading space: {bad[:3]}"

    def test_train_prompts_are_completion_style(self, train):
        bad = [(i, r["prompt"]) for i, r in enumerate(train)
               if r["prompt"].startswith("Q:")]
        assert not bad, \
            f"train: prompts in Q/A format (should be completion-style): {bad[:3]}"

    def test_test_prompts_are_qa_format(self, all_test):
        bad = [(i, r["prompt"][-20:]) for i, r in enumerate(all_test)
               if not self.QA_PAT.match(r["prompt"])]
        assert not bad, \
            f"Test prompts not in Q:/A: format: {bad[:3]}"

    def test_test_records_have_required_keys(self, all_test):
        required = {"prompt", "completion", "entity_id", "relation",
                    "direction", "a", "b"}
        for i, r in enumerate(all_test):
            missing = required - r.keys()
            assert not missing, \
                f"test record {i}: missing keys {missing}"

    def test_direction_values_valid(self, all_test):
        bad = [(i, r["direction"]) for i, r in enumerate(all_test)
               if r["direction"] not in {"forward", "reverse"}]
        assert not bad, f"Invalid direction values: {bad[:5]}"

    def test_relation_values_valid(self, metadata, all_test):
        for i, r in enumerate(metadata + all_test):
            assert r.get("relation") in RELATIONS, \
                f"Record {i}: invalid relation '{r.get('relation')}'"

    def test_metadata_required_keys(self, metadata):
        required = {"entity_id","relation","a","b","a_label","b_label",
                    "train","test_forward","test_reverse"}
        for i, r in enumerate(metadata):
            missing = required - r.keys()
            assert not missing, \
                f"metadata record {i}: missing keys {missing}"

    def test_a_b_label_consistency(self, metadata):
        for i, r in enumerate(metadata):
            assert r["a_label"] == A_LABELS[r["relation"]], \
                f"metadata record {i}: a_label '{r['a_label']}' " \
                f"does not match relation '{r['relation']}'"
            assert r["b_label"] == B_LABELS[r["relation"]], \
                f"metadata record {i}: b_label '{r['b_label']}' " \
                f"does not match relation '{r['relation']}'"


# ─────────────────────────────────────────────────────────────────────────────
# 3. COUNT CONSISTENCY
# ─────────────────────────────────────────────────────────────────────────────

class TestCounts:

    def test_train_count_matches_metadata(self, train, metadata):
        assert len(train) == len(metadata), \
            f"train has {len(train)}, metadata has {len(metadata)}"

    def test_each_test_file_has_correct_count(
        self, bi_fwd, bi_rev, wr_fwd, wr_rev, co_fwd, co_rev, meta_by_rel
    ):
        for name, records, rel in [
            ("bi_fwd", bi_fwd, "born_in"),
            ("bi_rev", bi_rev, "born_in"),
            ("wr_fwd", wr_fwd, "wrote"),
            ("wr_rev", wr_rev, "wrote"),
            ("co_fwd", co_fwd, "capital_of"),
            ("co_rev", co_rev, "capital_of"),
        ]:
            expected = len(meta_by_rel[rel])
            assert len(records) == expected, \
                f"{name}: has {len(records)} records, expected {expected}"

    def test_equal_counts_across_relations(self, meta_by_rel):
        counts = {rel: len(recs) for rel, recs in meta_by_rel.items()}
        assert max(counts.values()) == min(counts.values()), \
            f"Unequal entity counts across relations: {counts}"

    def test_no_duplicate_train_prompts(self, train):
        dupes = {p: c for p, c in
                 Counter(r["prompt"] for r in train).items() if c > 1}
        assert not dupes, f"Duplicate train prompts: {list(dupes.items())[:3]}"

    def test_no_duplicate_entity_ids_per_test_file(
        self, bi_fwd, bi_rev, wr_fwd, wr_rev, co_fwd, co_rev
    ):
        for name, records in [
            ("bi_fwd", bi_fwd), ("bi_rev", bi_rev),
            ("wr_fwd", wr_fwd), ("wr_rev", wr_rev),
            ("co_fwd", co_fwd), ("co_rev", co_rev),
        ]:
            dupes = {e: c for e, c in
                     Counter(r["entity_id"] for r in records).items() if c > 1}
            assert not dupes, \
                f"{name}: duplicate entity_ids: {dupes}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. ENTITY UNIQUENESS AND TYPE CONSISTENCY
# ─────────────────────────────────────────────────────────────────────────────

class TestEntityUniqueness:

    def test_all_a_entities_unique(self, all_a, metadata):
        assert len(all_a) == len(metadata), \
            f"Duplicate 'a' entities: expected {len(metadata)}, got {len(all_a)}"

    def test_all_b_entities_unique(self, all_b, metadata):
        assert len(all_b) == len(metadata), \
            f"Duplicate 'b' entities: expected {len(metadata)}, got {len(all_b)}"

    def test_a_and_b_entities_globally_disjoint(self, all_a, all_b):
        overlap = all_a & all_b
        assert not overlap, \
            f"Entities appear on both sides of a relation: {overlap}"

    def test_person_names_only_in_person_relations(self, meta_by_rel):
        """
        Person names (born_in and wrote A-entities) must not appear as
        capital cities or countries (capital_of entities).
        """
        bi_persons = {r["a"] for r in meta_by_rel["born_in"]}
        wr_persons = {r["a"] for r in meta_by_rel["wrote"]}
        all_persons = bi_persons | wr_persons

        cap_cities = {r["a"] for r in meta_by_rel["capital_of"]}
        countries  = {r["b"] for r in meta_by_rel["capital_of"]}
        non_persons = cap_cities | countries

        overlap = all_persons & non_persons
        assert not overlap, \
            f"Person names appear as capital entities: {overlap}"

    def test_born_in_places_not_in_capital_of(self, meta_by_rel):
        """Birthplace cities must not appear as capital cities or countries."""
        bi_places  = {r["b"] for r in meta_by_rel["born_in"]}
        cap_cities = {r["a"] for r in meta_by_rel["capital_of"]}
        countries  = {r["b"] for r in meta_by_rel["capital_of"]}

        assert not (bi_places & cap_cities), \
            f"Born-in cities appear as capital cities: {bi_places & cap_cities}"
        assert not (bi_places & countries), \
            f"Born-in cities appear as countries: {bi_places & countries}"

    def test_capital_cities_disjoint_from_countries(self, meta_by_rel):
        """No entity should be both a capital city and a country."""
        cap_cities = {r["a"] for r in meta_by_rel["capital_of"]}
        countries  = {r["b"] for r in meta_by_rel["capital_of"]}
        overlap = cap_cities & countries
        assert not overlap, \
            f"Entities appear as both capital city and country: {overlap}"

    def test_work_titles_only_in_wrote(self, meta_by_rel):
        """Work titles (wrote B-entities) must not appear elsewhere."""
        works = {r["b"] for r in meta_by_rel["wrote"]}
        other = (
            {r["a"] for r in meta_by_rel["born_in"]} |
            {r["b"] for r in meta_by_rel["born_in"]} |
            {r["a"] for r in meta_by_rel["capital_of"]} |
            {r["b"] for r in meta_by_rel["capital_of"]}
        )
        overlap = works & other
        assert not overlap, \
            f"Work titles appear in non-wrote entities: {overlap}"

    def test_born_in_persons_disjoint_from_wrote_persons(self, meta_by_rel):
        """The same person must not appear in both born_in and wrote."""
        bi_persons = {r["a"] for r in meta_by_rel["born_in"]}
        wr_persons = {r["a"] for r in meta_by_rel["wrote"]}
        overlap = bi_persons & wr_persons
        assert not overlap, \
            f"Same person in both born_in and wrote: {overlap}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. DIRECTIONALITY
# ─────────────────────────────────────────────────────────────────────────────

class TestDirectionality:

    def test_train_verb_matches_relation(self, metadata):
        """Training prompt must contain the correct verb/preposition."""
        for i, r in enumerate(metadata):
            verb = TRAIN_VERB[r["relation"]]
            assert verb in r["train"]["prompt"], \
                f"metadata record {i} [{r['relation']}]: " \
                f"verb '{verb}' not in train prompt '{r['train']['prompt']}'"

    def test_train_a_in_prompt(self, metadata):
        """A-entity must appear in the training prompt (it is the subject)."""
        for i, r in enumerate(metadata):
            assert r["a"] in r["train"]["prompt"], \
                f"metadata record {i}: A='{r['a']}' not in train prompt"

    def test_train_b_in_completion(self, metadata):
        """B-entity must appear in the training completion."""
        for i, r in enumerate(metadata):
            assert r["b"] in r["train"]["completion"], \
                f"metadata record {i}: B='{r['b']}' not in train completion"

    def test_forward_test_a_in_prompt(self, metadata):
        """In forward test: A-entity (subject) is in the prompt."""
        for i, r in enumerate(metadata):
            assert r["a"] in r["test_forward"]["prompt"], \
                f"Forward test {i} [{r['relation']}]: A='{r['a']}' not in prompt"

    def test_forward_test_b_in_completion(self, metadata):
        """In forward test: B-entity is the answer."""
        for i, r in enumerate(metadata):
            assert r["b"] in r["test_forward"]["completion"], \
                f"Forward test {i}: B='{r['b']}' not in completion"

    def test_reverse_test_b_in_prompt(self, metadata):
        """In reverse test: B-entity (object) is in the prompt."""
        for i, r in enumerate(metadata):
            assert r["b"] in r["test_reverse"]["prompt"], \
                f"Reverse test {i} [{r['relation']}]: B='{r['b']}' not in prompt"

    def test_reverse_test_a_in_completion(self, metadata):
        """In reverse test: A-entity is the answer."""
        for i, r in enumerate(metadata):
            assert r["a"] in r["test_reverse"]["completion"], \
                f"Reverse test {i}: A='{r['a']}' not in completion"

    def test_answer_not_leaked_into_reverse_prompt(self, metadata):
        """The correct answer (A) must not appear in the reverse test prompt."""
        for i, r in enumerate(metadata):
            ans = r["test_reverse"]["completion"].strip()
            assert ans not in r["test_reverse"]["prompt"], \
                f"Reverse test {i}: answer '{ans}' leaked into prompt"

    def test_answer_not_leaked_into_forward_prompt(self, metadata):
        """The correct answer (B) must not appear in the forward test prompt."""
        for i, r in enumerate(metadata):
            ans = r["test_forward"]["completion"].strip()
            assert ans not in r["test_forward"]["prompt"], \
                f"Forward test {i}: answer '{ans}' leaked into prompt"

    def test_forward_verb_in_test_prompts(
        self, bi_fwd, wr_fwd, co_fwd
    ):
        for name, records, rel in [
            ("bi_fwd", bi_fwd, "born_in"),
            ("wr_fwd", wr_fwd, "wrote"),
            ("co_fwd", co_fwd, "capital_of"),
        ]:
            fragment = FWD_VERB[rel]
            bad = [(i, r["prompt"]) for i, r in enumerate(records)
                   if fragment not in r["prompt"]]
            assert not bad, \
                f"{name}: verb fragment '{fragment}' missing: {bad[:2]}"

    def test_reverse_verb_in_test_prompts(
        self, bi_rev, wr_rev, co_rev
    ):
        for name, records, rel in [
            ("bi_rev", bi_rev, "born_in"),
            ("wr_rev", wr_rev, "wrote"),
            ("co_rev", co_rev, "capital_of"),
        ]:
            fragment = REV_VERB[rel]
            bad = [(i, r["prompt"]) for i, r in enumerate(records)
                   if fragment not in r["prompt"]]
            assert not bad, \
                f"{name}: verb fragment '{fragment}' missing: {bad[:2]}"

    def test_capital_of_reverse_is_natural_question(self, co_rev):
        """
        The reverse direction for capital_of should be phrased as the natural
        English question 'What is the capital of X?' — not the awkward inversion.
        """
        for i, r in enumerate(co_rev):
            assert r["prompt"].startswith("Q: What is the capital of"), \
                f"co_rev record {i}: reverse prompt is not the natural " \
                f"'What is the capital of...' form: '{r['prompt'][:60]}'"


# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAIN / TEST SEPARATION
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainTestSeparation:

    def test_no_train_prompt_in_test_files(self, train, all_test):
        train_prompts = {r["prompt"] for r in train}
        test_prompts  = {r["prompt"] for r in all_test}
        overlap = train_prompts & test_prompts
        assert not overlap, \
            f"{len(overlap)} train prompts appear verbatim in test files"

    def test_forward_and_reverse_prompts_disjoint_per_relation(
        self, bi_fwd, bi_rev, wr_fwd, wr_rev, co_fwd, co_rev
    ):
        for rel, fwd, rev in [
            ("born_in",    bi_fwd, bi_rev),
            ("wrote",      wr_fwd, wr_rev),
            ("capital_of", co_fwd, co_rev),
        ]:
            fwd_prompts = {r["prompt"] for r in fwd}
            rev_prompts = {r["prompt"] for r in rev}
            overlap = fwd_prompts & rev_prompts
            assert not overlap, \
                f"[{rel}] forward and reverse share prompts: {list(overlap)[:3]}"

    def test_train_completions_are_b_entities(self, train, all_b):
        """All training completions should resolve to a B-entity."""
        for i, r in enumerate(train):
            comp = r["completion"].strip().rstrip(".")
            assert comp in all_b, \
                f"train record {i}: completion '{comp}' is not a known B-entity"


# ─────────────────────────────────────────────────────────────────────────────
# 7. METADATA ↔ JSONL CONSISTENCY
# ─────────────────────────────────────────────────────────────────────────────

class TestMetadataConsistency:

    def _check_file_vs_meta(self, records, meta_key, filename, metadata):
        meta_by_eid = {r["entity_id"]: r for r in metadata}
        for i, rec in enumerate(records):
            eid = rec["entity_id"]
            assert eid in meta_by_eid, \
                f"{filename} record {i}: entity_id '{eid}' not in metadata"
            meta_rec = meta_by_eid[eid][meta_key]
            assert rec["prompt"] == meta_rec["prompt"], \
                f"{filename} entity '{eid}': prompt mismatch"
            assert rec["completion"] == meta_rec["completion"], \
                f"{filename} entity '{eid}': completion mismatch"

    def test_bi_fwd_matches_metadata(self, bi_fwd, metadata):
        self._check_file_vs_meta(bi_fwd, "test_forward",
                                 "test_born_in_forward", metadata)

    def test_bi_rev_matches_metadata(self, bi_rev, metadata):
        self._check_file_vs_meta(bi_rev, "test_reverse",
                                 "test_born_in_reverse", metadata)

    def test_wr_fwd_matches_metadata(self, wr_fwd, metadata):
        self._check_file_vs_meta(wr_fwd, "test_forward",
                                 "test_wrote_forward", metadata)

    def test_wr_rev_matches_metadata(self, wr_rev, metadata):
        self._check_file_vs_meta(wr_rev, "test_reverse",
                                 "test_wrote_reverse", metadata)

    def test_co_fwd_matches_metadata(self, co_fwd, metadata):
        self._check_file_vs_meta(co_fwd, "test_forward",
                                 "test_capital_of_forward", metadata)

    def test_co_rev_matches_metadata(self, co_rev, metadata):
        self._check_file_vs_meta(co_rev, "test_reverse",
                                 "test_capital_of_reverse", metadata)

    def test_all_train_facts_in_train_jsonl(self, train, metadata):
        train_set = {(r["prompt"], r["completion"]) for r in train}
        for i, m in enumerate(metadata):
            key = (m["train"]["prompt"], m["train"]["completion"])
            assert key in train_set, \
                f"metadata record {i} ('{m['entity_id']}'): " \
                f"train fact not found in train.jsonl"

    def test_all_metadata_entity_ids_in_test_files(
        self, metadata, bi_fwd, bi_rev, wr_fwd, wr_rev, co_fwd, co_rev
    ):
        meta_eids = {r["entity_id"] for r in metadata}
        for name, records in [
            ("bi_fwd", bi_fwd), ("bi_rev", bi_rev),
            ("wr_fwd", wr_fwd), ("wr_rev", wr_rev),
            ("co_fwd", co_fwd), ("co_rev", co_rev),
        ]:
            file_eids = {r["entity_id"] for r in records}
            rel = records[0]["relation"] if records else "?"
            rel_meta = {r["entity_id"] for r in metadata
                        if r["relation"] == rel}
            missing = rel_meta - file_eids
            extra   = file_eids - rel_meta
            assert not missing, f"{name}: metadata eids missing from file: {missing}"
            assert not extra,   f"{name}: extra eids in file not in metadata: {extra}"


# ─────────────────────────────────────────────────────────────────────────────
# 8. CROSS-DATASET DISJOINTNESS (optional)
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossDatasetDisjointness:

    def _all_mh_entities(self, mh_metadata) -> Set[str]:
        s: Set[str] = set()
        for c in mh_metadata:
            s.update(c["entities"].values())
        return s

    def _all_pp_entities(self, pp_metadata) -> Set[str]:
        return {r["name"] for r in pp_metadata} | {r["work"] for r in pp_metadata}

    def _all_it_entities(self, it_metadata) -> Set[str]:
        return {r["name"] for r in it_metadata} | {r["work"] for r in it_metadata}

    def test_rg_a_disjoint_from_mh(self, all_a, mh_metadata):
        if mh_metadata is None:
            pytest.skip("--mh_dir not supplied")
        overlap = all_a & self._all_mh_entities(mh_metadata)
        assert not overlap, f"RG A-entities overlap with MH: {overlap}"

    def test_rg_b_disjoint_from_mh(self, all_b, mh_metadata):
        if mh_metadata is None:
            pytest.skip("--mh_dir not supplied")
        overlap = all_b & self._all_mh_entities(mh_metadata)
        assert not overlap, f"RG B-entities overlap with MH: {overlap}"

    def test_rg_a_disjoint_from_pp(self, all_a, pp_metadata):
        if pp_metadata is None:
            pytest.skip("--pp_dir not supplied")
        overlap = all_a & self._all_pp_entities(pp_metadata)
        assert not overlap, f"RG A-entities overlap with PP: {overlap}"

    def test_rg_b_disjoint_from_pp(self, all_b, pp_metadata):
        if pp_metadata is None:
            pytest.skip("--pp_dir not supplied")
        overlap = all_b & self._all_pp_entities(pp_metadata)
        assert not overlap, f"RG B-entities overlap with PP: {overlap}"

    def test_rg_a_disjoint_from_it(self, all_a, it_metadata):
        if it_metadata is None:
            pytest.skip("--it_dir not supplied")
        overlap = all_a & self._all_it_entities(it_metadata)
        assert not overlap, f"RG A-entities overlap with IT: {overlap}"

    def test_rg_b_disjoint_from_it(self, all_b, it_metadata):
        if it_metadata is None:
            pytest.skip("--it_dir not supplied")
        overlap = all_b & self._all_it_entities(it_metadata)
        assert not overlap, f"RG B-entities overlap with IT: {overlap}"

    def test_rg_person_last_names_not_in_mh_places(
        self, all_a, meta_by_rel, mh_metadata
    ):
        if mh_metadata is None:
            pytest.skip("--mh_dir not supplied")
        mh_places = {
            c["entities"]["D"] for c in mh_metadata
            if c.get("chain_type") == "3hop"
        }
        persons = (
            {r["a"] for r in meta_by_rel["born_in"]} |
            {r["a"] for r in meta_by_rel["wrote"]}
        )
        violations = [
            (p, pl) for p in persons for pl in mh_places
            if p.split()[-1] in pl or pl in p.split()[-1]
        ]
        assert not violations, \
            f"RG last names substring-overlap with MH places: {violations[:5]}"
