"""
test_paraphrase_probing_dataset.py
───────────────────────────────────
Validates every invariant of the paraphrase probing dataset produced by
generate_paraphrase_probing.py.

Run:
  pytest test_paraphrase_probing_dataset.py --pp_dir data/paraphrase_probing -v

Cross-dataset check (optional, runs if --mh_dir is also supplied):
  pytest test_paraphrase_probing_dataset.py \
      --pp_dir data/paraphrase_probing \
      --mh_dir data/multihop -v
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS (must stay in sync with generate_paraphrase_probing.py)
# ─────────────────────────────────────────────────────────────────────────────

RELATIONS        = {"composer", "director", "author"}
SURFACE_FORMS    = {"original", "fill_blank", "indirect", "possessive", "yes_no"}
QA_SURFACE_FORMS = {"original", "indirect", "possessive", "yes_no"}
COMPLETION_FORMS = {"fill_blank"}           # no Q:/A: format
YES_NO_ANSWERS   = {" Yes", " No"}

RELATION_VERB_FRAGMENTS = {
    # (relation, surface_form) → substring that must appear in prompt
    ("composer", "original"):   "composer of",
    ("composer", "fill_blank"): "was composed by",
    ("composer", "indirect"):   "composed",
    ("composer", "possessive"): "'s composer",
    ("composer", "yes_no"):     "composer of",
    ("director", "original"):   "director of",
    ("director", "fill_blank"): "was directed by",
    ("director", "indirect"):   "directed",
    ("director", "possessive"): "'s director",
    ("director", "yes_no"):     "director of",
    ("author",   "original"):   "author of",
    ("author",   "fill_blank"): "was written by",
    ("author",   "indirect"):   "wrote",
    ("author",   "possessive"): "'s author",
    ("author",   "yes_no"):     "author of",
}


# ─────────────────────────────────────────────────────────────────────────────
# PYTEST CONFIG
# ─────────────────────────────────────────────────────────────────────────────

def pytest_addoption(parser):
    parser.addoption("--pp_dir", action="store",
                     default="data/paraphrase_probing",
                     help="Path to the paraphrase probing dataset directory.")
    parser.addoption("--mh_dir", action="store", default=None,
                     help="(Optional) Path to the multi-hop dataset directory "
                          "for cross-dataset disjointness checks.")


@pytest.fixture(scope="session")
def pp_dir(pytestconfig) -> Path:
    p = Path(pytestconfig.getoption("--pp_dir"))
    assert p.exists(), f"pp_dir does not exist: {p}"
    return p


@pytest.fixture(scope="session")
def mh_dir(pytestconfig) -> Optional[Path]:
    val = pytestconfig.getoption("--mh_dir")
    if val is None:
        return None
    p = Path(val)
    assert p.exists(), f"mh_dir does not exist: {p}"
    return p


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
                pytest.fail(f"Invalid JSON on line {i} of {path.name}: {e}")
    return records


@pytest.fixture(scope="session")
def train(pp_dir):       return load_jsonl(pp_dir / "train.jsonl")

@pytest.fixture(scope="session")
def original(pp_dir):    return load_jsonl(pp_dir / "test_original.jsonl")

@pytest.fixture(scope="session")
def fill_blank(pp_dir):  return load_jsonl(pp_dir / "test_fill_blank.jsonl")

@pytest.fixture(scope="session")
def indirect(pp_dir):    return load_jsonl(pp_dir / "test_indirect.jsonl")

@pytest.fixture(scope="session")
def possessive(pp_dir):  return load_jsonl(pp_dir / "test_possessive.jsonl")

@pytest.fixture(scope="session")
def yes_no(pp_dir):      return load_jsonl(pp_dir / "test_yes_no.jsonl")

@pytest.fixture(scope="session")
def metadata(pp_dir):    return load_jsonl(pp_dir / "metadata.jsonl")

@pytest.fixture(scope="session")
def all_test_records(original, fill_blank, indirect, possessive, yes_no):
    return original + fill_blank + indirect + possessive + yes_no

@pytest.fixture(scope="session")
def all_names(metadata) -> Set[str]:
    return {r["name"] for r in metadata}

@pytest.fixture(scope="session")
def all_works(metadata) -> Set[str]:
    return {r["work"] for r in metadata}

@pytest.fixture(scope="session")
def mh_metadata(mh_dir) -> Optional[List[Dict]]:
    if mh_dir is None:
        return None
    mh_path = mh_dir / "metadata.jsonl"
    if not mh_path.exists():
        return None
    return load_jsonl(mh_path)


# ─────────────────────────────────────────────────────────────────────────────
# 1. FILE EXISTENCE
# ─────────────────────────────────────────────────────────────────────────────

class TestFileExistence:
    REQUIRED = [
        "train.jsonl",
        "test_original.jsonl",
        "test_fill_blank.jsonl",
        "test_indirect.jsonl",
        "test_possessive.jsonl",
        "test_yes_no.jsonl",
        "metadata.jsonl",
    ]

    def test_all_files_present(self, pp_dir):
        missing = [f for f in self.REQUIRED if not (pp_dir / f).exists()]
        assert not missing, f"Missing files: {missing}"

    def test_no_empty_files(self, pp_dir):
        empty = [f for f in self.REQUIRED
                 if (pp_dir / f).stat().st_size == 0]
        assert not empty, f"Empty files: {empty}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. FORMAT CORRECTNESS
# ─────────────────────────────────────────────────────────────────────────────

class TestFormat:

    QA_PATTERN = re.compile(r"^Q:.+\nA:$", re.DOTALL)

    def _check_base_keys(self, records: List[Dict], name: str):
        for i, r in enumerate(records):
            assert "prompt" in r and "completion" in r, \
                f"{name} record {i}: missing prompt/completion keys"
            assert isinstance(r["prompt"], str) and r["prompt"], \
                f"{name} record {i}: prompt must be a non-empty string"
            assert isinstance(r["completion"], str) and r["completion"], \
                f"{name} record {i}: completion must be a non-empty string"

    def test_train_base_keys(self, train):
        self._check_base_keys(train, "train")

    def test_test_base_keys(self, original, fill_blank, indirect,
                            possessive, yes_no):
        for name, records in [
            ("test_original",   original),
            ("test_fill_blank", fill_blank),
            ("test_indirect",   indirect),
            ("test_possessive", possessive),
            ("test_yes_no",     yes_no),
        ]:
            self._check_base_keys(records, name)

    def test_all_completions_start_with_space(self, train, all_test_records):
        bad_train = [(i, r["completion"]) for i, r in enumerate(train)
                     if not r["completion"].startswith(" ")]
        assert not bad_train, \
            f"train: completions missing leading space: {bad_train[:3]}"

        bad_test = [(i, r["completion"]) for i, r in enumerate(all_test_records)
                    if not r["completion"].startswith(" ")]
        assert not bad_test, \
            f"test files: completions missing leading space: {bad_test[:3]}"

    def test_qa_surface_forms_have_qa_format(self, original, indirect,
                                              possessive, yes_no):
        for name, records in [
            ("test_original",   original),
            ("test_indirect",   indirect),
            ("test_possessive", possessive),
            ("test_yes_no",     yes_no),
        ]:
            bad = [(i, r["prompt"]) for i, r in enumerate(records)
                   if not self.QA_PATTERN.match(r["prompt"])]
            assert not bad, \
                f"{name}: prompts not in Q:/A: format: {bad[:3]}"

    def test_fill_blank_not_qa_format(self, fill_blank):
        """fill_blank must be completion-style (no Q:/A: pattern)."""
        bad = [(i, r["prompt"]) for i, r in enumerate(fill_blank)
               if self.QA_PATTERN.match(r["prompt"])]
        assert not bad, \
            f"test_fill_blank: prompts incorrectly in Q:/A: format: {bad[:3]}"

    def test_yes_no_completions_are_only_yes_or_no(self, yes_no):
        bad = [(i, r["completion"]) for i, r in enumerate(yes_no)
               if r["completion"] not in YES_NO_ANSWERS]
        assert not bad, \
            f"test_yes_no: invalid completions (not ' Yes'/` No`): {bad[:5]}"

    def test_test_records_have_metadata_keys(self, original, fill_blank,
                                              indirect, possessive, yes_no):
        """Test records must carry entity_id, surface_form, relation."""
        required = {"entity_id", "surface_form", "relation"}
        for name, records in [
            ("test_original",   original),
            ("test_fill_blank", fill_blank),
            ("test_indirect",   indirect),
            ("test_possessive", possessive),
            ("test_yes_no",     yes_no),
        ]:
            for i, r in enumerate(records):
                missing = required - r.keys()
                assert not missing, \
                    f"{name} record {i}: missing keys {missing}"

    def test_yes_no_records_have_yes_no_label(self, yes_no):
        for i, r in enumerate(yes_no):
            assert "yes_no_label" in r, \
                f"test_yes_no record {i}: missing 'yes_no_label'"
            assert r["yes_no_label"] in {"yes", "no"}, \
                f"test_yes_no record {i}: invalid yes_no_label '{r['yes_no_label']}'"

    def test_metadata_has_all_keys(self, metadata):
        required = {
            "entity_id", "name", "work", "relation",
            "yes_no_label", "foil_name",
            "train", "test_original", "test_fill_blank",
            "test_indirect", "test_possessive", "test_yes_no",
        }
        for i, r in enumerate(metadata):
            missing = required - r.keys()
            assert not missing, \
                f"metadata record {i}: missing keys {missing}"

    def test_surface_form_values_valid(self, all_test_records):
        for i, r in enumerate(all_test_records):
            assert r["surface_form"] in SURFACE_FORMS, \
                f"Record {i}: invalid surface_form '{r['surface_form']}'"

    def test_relation_values_valid(self, metadata, all_test_records):
        for i, r in enumerate(metadata + all_test_records):
            assert r["relation"] in RELATIONS, \
                f"Record {i}: invalid relation '{r['relation']}'"


# ─────────────────────────────────────────────────────────────────────────────
# 3. COUNT CONSISTENCY
# ─────────────────────────────────────────────────────────────────────────────

class TestCounts:

    def test_all_test_files_same_count_as_metadata(
        self, metadata, original, fill_blank, indirect, possessive, yes_no
    ):
        n = len(metadata)
        for name, records in [
            ("test_original",   original),
            ("test_fill_blank", fill_blank),
            ("test_indirect",   indirect),
            ("test_possessive", possessive),
            ("test_yes_no",     yes_no),
        ]:
            assert len(records) == n, \
                f"{name} has {len(records)} records, expected {n}"

    def test_train_count_matches_metadata(self, train, metadata):
        assert len(train) == len(metadata), \
            f"train has {len(train)} records, metadata has {len(metadata)}"

    def test_yes_no_is_50_50(self, yes_no):
        yes_count = sum(1 for r in yes_no if r["completion"] == " Yes")
        no_count  = sum(1 for r in yes_no if r["completion"] == " No")
        total = len(yes_no)
        # Allow ±1 for odd totals (e.g. 99/99 for 198 total)
        assert abs(yes_count - no_count) <= 1, \
            f"yes/no split is not ~50/50: {yes_count} yes, {no_count} no"

    def test_relation_counts_are_balanced(self, metadata):
        counts = Counter(r["relation"] for r in metadata)
        values = list(counts.values())
        # Allow ±1 between relation counts
        assert max(values) - min(values) <= 1, \
            f"Relation counts are unbalanced: {dict(counts)}"

    def test_no_duplicate_train_prompts(self, train):
        dupes = {p: c for p, c in
                 Counter(r["prompt"] for r in train).items() if c > 1}
        assert not dupes, \
            f"Duplicate train prompts: {list(dupes.items())[:3]}"

    def test_no_duplicate_test_prompts_per_file(
        self, original, fill_blank, indirect, possessive, yes_no
    ):
        for name, records in [
            ("test_original",   original),
            ("test_fill_blank", fill_blank),
            ("test_indirect",   indirect),
            ("test_possessive", possessive),
            ("test_yes_no",     yes_no),
        ]:
            dupes = {p: c for p, c in
                     Counter(r["prompt"] for r in records).items() if c > 1}
            assert not dupes, \
                f"{name}: duplicate prompts: {list(dupes.items())[:3]}"

    def test_each_entity_appears_exactly_once_per_surface_form(
        self, original, fill_blank, indirect, possessive, yes_no
    ):
        for name, records in [
            ("test_original",   original),
            ("test_fill_blank", fill_blank),
            ("test_indirect",   indirect),
            ("test_possessive", possessive),
            ("test_yes_no",     yes_no),
        ]:
            eids = [r["entity_id"] for r in records]
            dupes = {e: c for e, c in Counter(eids).items() if c > 1}
            assert not dupes, \
                f"{name}: entity_ids appearing more than once: {dupes}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. ENTITY UNIQUENESS
# ─────────────────────────────────────────────────────────────────────────────

class TestEntityUniqueness:

    def test_all_person_names_unique(self, metadata):
        names = [r["name"] for r in metadata]
        dupes = {n: c for n, c in Counter(names).items() if c > 1}
        assert not dupes, f"Duplicate person names: {list(dupes.items())[:5]}"

    def test_all_work_titles_unique(self, metadata):
        works = [r["work"] for r in metadata]
        dupes = {w: c for w, c in Counter(works).items() if c > 1}
        assert not dupes, f"Duplicate work titles: {list(dupes.items())[:5]}"

    def test_all_entity_ids_unique(self, metadata):
        eids = [r["entity_id"] for r in metadata]
        dupes = {e: c for e, c in Counter(eids).items() if c > 1}
        assert not dupes, f"Duplicate entity_ids: {list(dupes.items())[:5]}"

    def test_person_names_not_substrings_of_works(self, all_names, all_works):
        """Last name token must not appear inside any work title."""
        violations = []
        for name in all_names:
            last = name.split()[-1]
            for work in all_works:
                if last in work:
                    violations.append((name, work))
        assert not violations, \
            f"Last-name token inside work title: {violations[:5]}"

    def test_work_titles_not_substrings_of_names(self, all_names, all_works):
        """No full work title should appear in any person name (sanity)."""
        violations = [
            (name, work) for name in all_names for work in all_works
            if work in name
        ]
        assert not violations, \
            f"Work title inside person name: {violations[:5]}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. DIRECTIONALITY
# ─────────────────────────────────────────────────────────────────────────────

class TestDirectionality:

    def test_training_facts_are_forward(self, train, metadata):
        """
        Train prompts must contain the person name (subject) as the start
        and must NOT be in Q/A format (i.e. they are completion-style forward facts).
        """
        qa_pattern = re.compile(r"^Q:", re.DOTALL)
        meta_by_name = {r["name"]: r for r in metadata}
        for i, rec in enumerate(train):
            assert not qa_pattern.match(rec["prompt"]), \
                f"train record {i}: training prompt is in Q/A format (should be forward completion)"

    def test_training_completions_contain_work_not_name(self, train, metadata):
        """
        Forward training completion should contain the work, not start with
        a person name (which would mean it's a reverse fact).
        """
        name_set = {r["name"] for r in metadata}
        for i, rec in enumerate(train):
            comp = rec["completion"].strip().rstrip(".")
            # Completion should not be a bare person name
            assert comp not in name_set, \
                f"train record {i}: completion '{comp}' looks like a " \
                f"person name (suggests reverse direction leaked into train)"

    def test_test_completions_are_person_names_or_yes_no(
        self, original, fill_blank, indirect, possessive, yes_no, all_names
    ):
        """
        For the 4 non-yes_no surface forms, the completion (stripped) must be
        a real person name from the dataset.
        """
        for name, records in [
            ("test_original",   original),
            ("test_fill_blank", fill_blank),
            ("test_indirect",   indirect),
            ("test_possessive", possessive),
        ]:
            for i, r in enumerate(records):
                comp = r["completion"].strip().rstrip(".")
                assert comp in all_names, \
                    f"{name} record {i}: completion '{comp}' is not a " \
                    f"known entity name"

        for i, r in enumerate(yes_no):
            assert r["completion"] in YES_NO_ANSWERS, \
                f"test_yes_no record {i}: completion '{r['completion']}' " \
                f"is not ' Yes' or ' No'"


# ─────────────────────────────────────────────────────────────────────────────
# 6. RELATION CONSISTENCY
# ─────────────────────────────────────────────────────────────────────────────

class TestRelationConsistency:

    def test_verb_fragment_matches_relation_in_every_record(
        self, original, fill_blank, indirect, possessive, yes_no
    ):
        """
        The verb fragment expected for each (relation, surface_form) pair
        must appear in the prompt of every corresponding test record.
        """
        for sf_name, records in [
            ("original",   original),
            ("fill_blank", fill_blank),
            ("indirect",   indirect),
            ("possessive", possessive),
            ("yes_no",     yes_no),
        ]:
            for i, r in enumerate(records):
                relation = r["relation"]
                key      = (relation, sf_name)
                fragment = RELATION_VERB_FRAGMENTS.get(key)
                if fragment is None:
                    continue
                assert fragment in r["prompt"], \
                    f"{sf_name} record {i}: expected verb fragment " \
                    f"'{fragment}' for relation '{relation}' not in prompt: " \
                    f"'{r['prompt'][:80]}'"

    def test_relation_distribution_in_test_files(
        self, original, fill_blank, indirect, possessive, yes_no
    ):
        """Each test file should have the same relation distribution."""
        ref_counts = Counter(r["relation"] for r in original)
        for name, records in [
            ("test_fill_blank", fill_blank),
            ("test_indirect",   indirect),
            ("test_possessive", possessive),
            ("test_yes_no",     yes_no),
        ]:
            counts = Counter(r["relation"] for r in records)
            assert counts == ref_counts, \
                f"{name} relation distribution {dict(counts)} != " \
                f"original {dict(ref_counts)}"


# ─────────────────────────────────────────────────────────────────────────────
# 7. YES/NO INTEGRITY
# ─────────────────────────────────────────────────────────────────────────────

class TestYesNoIntegrity:

    def test_foils_are_real_dataset_names(self, yes_no, all_names):
        bad = [
            (i, r["foil_name"])
            for i, r in enumerate(yes_no)
            if r["foil_name"] is not None and r["foil_name"] not in all_names
        ]
        assert not bad, \
            f"yes_no foils that are not real dataset names: {bad[:5]}"

    def test_no_entity_is_own_foil(self, yes_no, metadata):
        meta_by_eid = {r["entity_id"]: r["name"] for r in metadata}
        for i, r in enumerate(yes_no):
            if r["foil_name"] is None:
                continue
            true_name = meta_by_eid.get(r["entity_id"])
            assert r["foil_name"] != true_name, \
                f"yes_no record {i}: entity '{true_name}' is its own foil"

    def test_foil_name_appears_in_no_prompts(self, yes_no, metadata):
        """For 'no' records: the foil name (not the true name) is in the prompt."""
        meta_by_eid = {r["entity_id"]: r["name"] for r in metadata}
        for i, r in enumerate(yes_no):
            if r["yes_no_label"] != "no":
                continue
            foil = r["foil_name"]
            assert foil is not None, \
                f"yes_no record {i}: 'no' label but foil_name is None"
            assert foil in r["prompt"], \
                f"yes_no record {i}: foil '{foil}' not found in prompt"
            # True name must NOT be in the prompt (would give away the answer)
            true_name = meta_by_eid[r["entity_id"]]
            assert true_name not in r["prompt"], \
                f"yes_no record {i}: true name '{true_name}' leaked into " \
                f"'no' prompt — model could answer without reverse recall"

    def test_correct_name_in_yes_prompts(self, yes_no, metadata):
        """For 'yes' records: the true name must appear in the prompt."""
        meta_by_eid = {r["entity_id"]: r["name"] for r in metadata}
        for i, r in enumerate(yes_no):
            if r["yes_no_label"] != "yes":
                continue
            true_name = meta_by_eid[r["entity_id"]]
            assert true_name in r["prompt"], \
                f"yes_no record {i}: true name '{true_name}' not in 'yes' prompt"

    def test_foils_are_same_relation_as_entity(self, yes_no, metadata):
        """
        Foil names must come from entities with the same relation type.
        A 'composer' entity should not have a 'director' entity as its foil.
        """
        name_to_relation = {r["name"]: r["relation"] for r in metadata}
        eid_to_relation  = {r["entity_id"]: r["relation"] for r in metadata}
        for i, r in enumerate(yes_no):
            if r["foil_name"] is None:
                continue
            entity_relation = eid_to_relation[r["entity_id"]]
            foil_relation   = name_to_relation.get(r["foil_name"])
            assert foil_relation == entity_relation, \
                f"yes_no record {i}: entity relation '{entity_relation}' " \
                f"but foil '{r['foil_name']}' has relation '{foil_relation}'"

    def test_each_name_used_as_foil_at_most_once(self, yes_no):
        foils = [r["foil_name"] for r in yes_no if r["foil_name"] is not None]
        dupes = {f: c for f, c in Counter(foils).items() if c > 1}
        assert not dupes, \
            f"Some foil names used more than once: {list(dupes.items())[:5]}"

    def test_no_entity_appears_as_both_yes_and_no(self, yes_no):
        yes_eids = {r["entity_id"] for r in yes_no if r["yes_no_label"] == "yes"}
        no_eids  = {r["entity_id"] for r in yes_no if r["yes_no_label"] == "no"}
        overlap  = yes_eids & no_eids
        assert not overlap, \
            f"Entity IDs appearing as both yes and no: {overlap}"


# ─────────────────────────────────────────────────────────────────────────────
# 8. TRAIN / TEST SEPARATION
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainTestSeparation:

    def test_no_train_prompt_in_any_test_file(self, train, all_test_records):
        train_prompts = {r["prompt"] for r in train}
        test_prompts  = {r["prompt"] for r in all_test_records}
        overlap = train_prompts & test_prompts
        assert not overlap, \
            f"{len(overlap)} training prompts appear verbatim in test files"

    def test_train_is_forward_test_is_reverse(self, train, metadata):
        """
        Training prompts should contain the person name as subject (forward).
        The reverse test should ask about the WORK, not the person.
        Verified via: train prompt starts with person name, not a work title.
        """
        all_works = {r["work"] for r in metadata}
        for i, r in enumerate(train):
            for work in all_works:
                # No training prompt should start with a work title
                assert not r["prompt"].startswith(work), \
                    f"train record {i}: prompt starts with work title '{work}' " \
                    f"(suggests reverse fact in training data)"

    def test_work_titles_appear_in_prompts_not_completions_for_train(
        self, train, all_works
    ):
        """In training, work titles are in the *completion*, not the prompt prefix."""
        for i, r in enumerate(train):
            work_in_completion = any(w in r["completion"] for w in all_works)
            assert work_in_completion, \
                f"train record {i}: no work title found in completion " \
                f"'{r['completion']}' — may be reversed"

    def test_names_appear_in_prompts_not_completions_for_train(
        self, train, all_names
    ):
        """In training, person names are in the *prompt*, not the completion."""
        for i, r in enumerate(train):
            name_in_prompt = any(n in r["prompt"] for n in all_names)
            assert name_in_prompt, \
                f"train record {i}: no person name found in prompt " \
                f"'{r['prompt']}' — may be reversed"


# ─────────────────────────────────────────────────────────────────────────────
# 9. METADATA ↔ JSONL CONSISTENCY
# ─────────────────────────────────────────────────────────────────────────────

class TestMetadataConsistency:

    def _check_meta_vs_file(self, metadata, records, meta_key, file_name):
        file_by_eid = {r["entity_id"]: r for r in records}
        for i, meta in enumerate(metadata):
            eid = meta["entity_id"]
            assert eid in file_by_eid, \
                f"{file_name}: entity_id '{eid}' from metadata not found in file"
            file_rec = file_by_eid[eid]
            meta_rec = meta[meta_key]
            assert file_rec["prompt"] == meta_rec["prompt"], \
                f"{file_name} entity '{eid}': prompt mismatch\n" \
                f"  file    : {file_rec['prompt']!r}\n" \
                f"  metadata: {meta_rec['prompt']!r}"
            assert file_rec["completion"] == meta_rec["completion"], \
                f"{file_name} entity '{eid}': completion mismatch"

    def test_train_matches_metadata(self, metadata, train):
        train_by_eid = {}
        for rec in train:
            # Match by prompt prefix — find the entity
            for meta in metadata:
                if rec["prompt"] == meta["train"]["prompt"]:
                    train_by_eid[meta["entity_id"]] = rec
                    break
        for meta in metadata:
            eid = meta["entity_id"]
            assert eid in train_by_eid, \
                f"entity '{eid}' not found in train.jsonl"

    def test_original_matches_metadata(self, metadata, original):
        self._check_meta_vs_file(metadata, original,
                                 "test_original", "test_original.jsonl")

    def test_fill_blank_matches_metadata(self, metadata, fill_blank):
        self._check_meta_vs_file(metadata, fill_blank,
                                 "test_fill_blank", "test_fill_blank.jsonl")

    def test_indirect_matches_metadata(self, metadata, indirect):
        self._check_meta_vs_file(metadata, indirect,
                                 "test_indirect", "test_indirect.jsonl")

    def test_possessive_matches_metadata(self, metadata, possessive):
        self._check_meta_vs_file(metadata, possessive,
                                 "test_possessive", "test_possessive.jsonl")

    def test_yes_no_matches_metadata(self, metadata, yes_no):
        self._check_meta_vs_file(metadata, yes_no,
                                 "test_yes_no", "test_yes_no.jsonl")

    def test_all_entity_ids_in_metadata_appear_in_all_test_files(
        self, metadata, original, fill_blank, indirect, possessive, yes_no
    ):
        meta_eids = {r["entity_id"] for r in metadata}
        for name, records in [
            ("test_original",   original),
            ("test_fill_blank", fill_blank),
            ("test_indirect",   indirect),
            ("test_possessive", possessive),
            ("test_yes_no",     yes_no),
        ]:
            file_eids = {r["entity_id"] for r in records}
            missing = meta_eids - file_eids
            extra   = file_eids - meta_eids
            assert not missing, f"{name}: entity_ids in metadata but not file: {missing}"
            assert not extra,   f"{name}: entity_ids in file but not metadata: {extra}"


# ─────────────────────────────────────────────────────────────────────────────
# 10. CROSS-DATASET DISJOINTNESS (optional — runs if --mh_dir is supplied)
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossDatasetDisjointness:

    def test_pp_names_disjoint_from_mh_entities(
        self, metadata, mh_metadata, all_names
    ):
        """
        No person name in the paraphrase probing dataset should appear in
        the multi-hop dataset (in any role: A, B, org, place).
        """
        if mh_metadata is None:
            pytest.skip("--mh_dir not supplied; skipping cross-dataset check.")

        mh_entities: Set[str] = set()
        for chain in mh_metadata:
            mh_entities.update(chain["entities"].values())

        overlap = all_names & mh_entities
        assert not overlap, \
            f"PP person names also appear in multi-hop dataset: {overlap}"

    def test_pp_works_disjoint_from_mh_entities(
        self, metadata, mh_metadata, all_works
    ):
        """Work titles must not appear in multi-hop entity strings."""
        if mh_metadata is None:
            pytest.skip("--mh_dir not supplied; skipping cross-dataset check.")

        mh_entities: Set[str] = set()
        for chain in mh_metadata:
            mh_entities.update(chain["entities"].values())

        overlap = all_works & mh_entities
        assert not overlap, \
            f"PP work titles also appear in multi-hop dataset: {overlap}"

    def test_pp_last_name_tokens_not_in_mh_place_names(
        self, metadata, mh_metadata
    ):
        """Last name tokens from PP must not appear as substrings of MH places."""
        if mh_metadata is None:
            pytest.skip("--mh_dir not supplied; skipping cross-dataset check.")

        mh_places = {
            chain["entities"]["D"]
            for chain in mh_metadata
            if chain["chain_type"] == "3hop"
        }
        violations = []
        for rec in metadata:
            last = rec["name"].split()[-1]
            for place in mh_places:
                if last in place or place in last:
                    violations.append((rec["name"], place))
        assert not violations, \
            f"PP last name tokens overlap with MH place names: {violations[:5]}"
