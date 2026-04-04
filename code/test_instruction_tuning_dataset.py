"""
test_instruction_tuning_dataset.py
────────────────────────────────────
Validates every invariant of the instruction tuning ablation dataset produced
by generate_instruction_tuning.py.

Run:
  pytest test_instruction_tuning_dataset.py --it_dir data/instruction_tuning -v

Cross-dataset checks (optional):
  pytest test_instruction_tuning_dataset.py \
      --it_dir data/instruction_tuning \
      --mh_dir data/multihop \
      --pp_dir data/paraphrase_probing -v
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (must stay in sync with generate_instruction_tuning.py)
# ─────────────────────────────────────────────────────────────────────────────

RELATIONS   = {"composer", "director", "author"}
CONDITIONS  = {"zero_shot_direct", "zero_shot_cot",
               "few_shot_direct",  "few_shot_cot"}
COT_CONDITIONS    = {"zero_shot_cot", "few_shot_cot"}
FEW_SHOT_CONDITIONS = {"few_shot_direct", "few_shot_cot"}
K_DEMOS = 2   # number of in-context demonstrations

RELATION_VERB_FRAGMENTS = {
    "composer": ("composer of", "composed"),
    "director": ("director of", "directed"),
    "author":   ("author of",   "written"),
}

COT_MARKER        = "Think step by step before answering"
COT_CHAIN_MARKERS = [
    "Let me think step by step",
    "I recall from my training",
    "Working backwards",
    "Therefore, the answer is",
]


# ─────────────────────────────────────────────────────────────────────────────
# PYTEST CONFIG
# ─────────────────────────────────────────────────────────────────────────────

def pytest_addoption(parser):
    parser.addoption("--it_dir", action="store",
                     default="data/instruction_tuning")
    parser.addoption("--mh_dir", action="store", default=None)
    parser.addoption("--pp_dir", action="store", default=None)


@pytest.fixture(scope="session")
def it_dir(pytestconfig) -> Path:
    p = Path(pytestconfig.getoption("--it_dir"))
    assert p.exists(), f"it_dir does not exist: {p}"
    return p


@pytest.fixture(scope="session")
def mh_dir(pytestconfig) -> Optional[Path]:
    v = pytestconfig.getoption("--mh_dir")
    return Path(v) if v else None


@pytest.fixture(scope="session")
def pp_dir(pytestconfig) -> Optional[Path]:
    v = pytestconfig.getoption("--pp_dir")
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
                pytest.fail(f"Invalid JSON on line {i} of {path.name}: {e}")
    return records


@pytest.fixture(scope="session")
def train(it_dir):    return load_jsonl(it_dir / "train.jsonl")

@pytest.fixture(scope="session")
def demo_ents(it_dir):return load_jsonl(it_dir / "demo_entities.jsonl")

@pytest.fixture(scope="session")
def zsd(it_dir):      return load_jsonl(it_dir / "test_zero_shot_direct.jsonl")

@pytest.fixture(scope="session")
def zsc(it_dir):      return load_jsonl(it_dir / "test_zero_shot_cot.jsonl")

@pytest.fixture(scope="session")
def fsd(it_dir):      return load_jsonl(it_dir / "test_few_shot_direct.jsonl")

@pytest.fixture(scope="session")
def fsc(it_dir):      return load_jsonl(it_dir / "test_few_shot_cot.jsonl")

@pytest.fixture(scope="session")
def metadata(it_dir): return load_jsonl(it_dir / "metadata.jsonl")

@pytest.fixture(scope="session")
def all_test(zsd, zsc, fsd, fsc): return zsd + zsc + fsd + fsc

@pytest.fixture(scope="session")
def test_meta(metadata):
    return [r for r in metadata if r["entity_role"] == "test"]

@pytest.fixture(scope="session")
def demo_meta(metadata):
    return [r for r in metadata if r["entity_role"] == "demo"]

@pytest.fixture(scope="session")
def all_names(metadata) -> Set[str]:
    return {r["name"] for r in metadata}

@pytest.fixture(scope="session")
def all_works(metadata) -> Set[str]:
    return {r["work"] for r in metadata}

@pytest.fixture(scope="session")
def demo_ids(demo_meta) -> Set[str]:
    return {r["entity_id"] for r in demo_meta}

@pytest.fixture(scope="session")
def test_ids(test_meta) -> Set[str]:
    return {r["entity_id"] for r in test_meta}

@pytest.fixture(scope="session")
def mh_metadata(mh_dir) -> Optional[List[Dict]]:
    if mh_dir is None:
        return None
    p = mh_dir / "metadata.jsonl"
    return load_jsonl(p) if p.exists() else None

@pytest.fixture(scope="session")
def pp_metadata(pp_dir) -> Optional[List[Dict]]:
    if pp_dir is None:
        return None
    p = pp_dir / "metadata.jsonl"
    return load_jsonl(p) if p.exists() else None


# ─────────────────────────────────────────────────────────────────────────────
# 1. FILE EXISTENCE
# ─────────────────────────────────────────────────────────────────────────────

class TestFileExistence:
    REQUIRED = [
        "train.jsonl", "demo_entities.jsonl",
        "test_zero_shot_direct.jsonl", "test_zero_shot_cot.jsonl",
        "test_few_shot_direct.jsonl",  "test_few_shot_cot.jsonl",
        "metadata.jsonl",
    ]

    def test_all_files_present(self, it_dir):
        missing = [f for f in self.REQUIRED if not (it_dir / f).exists()]
        assert not missing, f"Missing files: {missing}"

    def test_no_empty_files(self, it_dir):
        empty = [f for f in self.REQUIRED
                 if (it_dir / f).stat().st_size == 0]
        assert not empty, f"Empty files: {empty}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. FORMAT
# ─────────────────────────────────────────────────────────────────────────────

class TestFormat:
    QA_END = re.compile(r"\nA:$")

    def test_all_completions_start_with_space(self, train, all_test):
        for name, records in [("train", train), ("all_test", all_test)]:
            bad = [(i, r["completion"]) for i, r in enumerate(records)
                   if not r["completion"].startswith(" ")]
            assert not bad, \
                f"{name}: completions missing leading space: {bad[:3]}"

    def test_all_test_prompts_end_with_a_colon(self, all_test):
        """Every test prompt must end with 'A:' so the model continues."""
        bad = [(i, r["prompt"][-10:]) for i, r in enumerate(all_test)
               if not self.QA_END.search(r["prompt"])]
        assert not bad, \
            f"Test prompts not ending with '\\nA:': {bad[:3]}"

    def test_train_prompts_are_completion_style(self, train):
        """Training prompts must be forward completion-style, not Q/A."""
        qa = re.compile(r"^Q:")
        bad = [(i, r["prompt"]) for i, r in enumerate(train)
               if qa.match(r["prompt"])]
        assert not bad, \
            f"train: prompts in Q/A format (should be forward completion): {bad[:3]}"

    def test_test_records_have_required_keys(self, all_test):
        required = {"prompt", "completion", "entity_id", "relation", "condition"}
        for i, r in enumerate(all_test):
            missing = required - r.keys()
            assert not missing, \
                f"test record {i}: missing keys {missing}"

    def test_condition_values_valid(self, all_test):
        bad = [(i, r["condition"]) for i, r in enumerate(all_test)
               if r["condition"] not in CONDITIONS]
        assert not bad, f"Invalid condition values: {bad[:5]}"

    def test_metadata_keys(self, metadata):
        required_all  = {"entity_id","entity_role","name","work","relation",
                         "cot_chain","train","reverse_q"}
        required_test = {"conditions", "demo_ids"}
        for i, r in enumerate(metadata):
            missing = required_all - r.keys()
            assert not missing, f"metadata record {i}: missing keys {missing}"
            if r["entity_role"] == "test":
                missing_t = required_test - r.keys()
                assert not missing_t, \
                    f"test metadata record {i}: missing keys {missing_t}"

    def test_demo_entities_keys(self, demo_ents):
        required = {"entity_id","name","work","relation",
                    "reverse_q","cot_chain","train"}
        for i, r in enumerate(demo_ents):
            missing = required - r.keys()
            assert not missing, \
                f"demo_entities record {i}: missing keys {missing}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. COUNT CONSISTENCY
# ─────────────────────────────────────────────────────────────────────────────

class TestCounts:

    def test_train_count(self, train, metadata):
        assert len(train) == len(metadata), \
            f"train has {len(train)} records, metadata has {len(metadata)}"

    def test_test_files_all_same_count(self, zsd, zsc, fsd, fsc, test_meta):
        n = len(test_meta)
        for name, records in [
            ("zsd", zsd), ("zsc", zsc), ("fsd", fsd), ("fsc", fsc)
        ]:
            assert len(records) == n, \
                f"{name} has {len(records)} records, expected {n}"

    def test_demo_entities_count(self, demo_ents, demo_meta):
        assert len(demo_ents) == len(demo_meta), \
            f"demo_entities.jsonl has {len(demo_ents)}, " \
            f"metadata demo entries: {len(demo_meta)}"

    def test_relation_balance_in_test_files(self, zsd):
        counts = Counter(r["relation"] for r in zsd)
        assert max(counts.values()) - min(counts.values()) <= 1, \
            f"Unbalanced relation counts in test files: {dict(counts)}"

    def test_no_duplicate_entity_ids_in_each_test_file(self, zsd, zsc, fsd, fsc):
        for name, records in [
            ("zsd", zsd), ("zsc", zsc), ("fsd", fsd), ("fsc", fsc)
        ]:
            dupes = {e: c for e, c in
                     Counter(r["entity_id"] for r in records).items() if c > 1}
            assert not dupes, f"{name}: duplicate entity_ids: {dupes}"

    def test_all_four_conditions_present(self, all_test):
        found = {r["condition"] for r in all_test}
        assert found == CONDITIONS, \
            f"Missing conditions: {CONDITIONS - found}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. DEMO / TEST SEPARATION
# ─────────────────────────────────────────────────────────────────────────────

class TestDemoTestSeparation:

    def test_demo_and_test_ids_disjoint(self, demo_ids, test_ids):
        overlap = demo_ids & test_ids
        assert not overlap, \
            f"entity_ids appear in both demo and test sets: {overlap}"

    def test_demo_entities_not_queried_in_test_files(
        self, zsd, zsc, fsd, fsc, demo_ids
    ):
        """Demo entity IDs must not appear as test query entity_ids."""
        for name, records in [
            ("zsd", zsd), ("zsc", zsc), ("fsd", fsd), ("fsc", fsc)
        ]:
            queried = {r["entity_id"] for r in records}
            overlap = queried & demo_ids
            assert not overlap, \
                f"{name}: demo entity_ids appear as test queries: {overlap}"

    def test_demo_names_not_as_test_completions(
        self, zsd, zsc, fsd, fsc, demo_meta
    ):
        """
        The correct answer for a test query must not be a demo entity's name.
        This would mean a demo entity accidentally became a test entity.
        """
        demo_name_set = {r["name"] for r in demo_meta}
        for name, records in [
            ("zsd", zsd), ("zsc", zsc), ("fsd", fsd), ("fsc", fsc)
        ]:
            bad = [
                (i, r["completion"].strip())
                for i, r in enumerate(records)
                if r["completion"].strip() in demo_name_set
            ]
            assert not bad, \
                f"{name}: test completions contain demo entity names: {bad[:3]}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. DEMONSTRATION INTEGRITY
# ─────────────────────────────────────────────────────────────────────────────

class TestDemonstrationIntegrity:

    def test_few_shot_records_have_demo_ids(self, fsd, fsc):
        for name, records in [("fsd", fsd), ("fsc", fsc)]:
            for i, r in enumerate(records):
                assert "demo_ids" in r, \
                    f"{name} record {i}: missing 'demo_ids'"
                assert len(r["demo_ids"]) == K_DEMOS, \
                    f"{name} record {i}: expected {K_DEMOS} demo_ids, " \
                    f"got {len(r['demo_ids'])}"

    def test_demo_ids_are_real_demo_entities(self, fsd, fsc, demo_ids):
        for name, records in [("fsd", fsd), ("fsc", fsc)]:
            for i, r in enumerate(records):
                for did in r.get("demo_ids", []):
                    assert did in demo_ids, \
                        f"{name} record {i}: demo_id '{did}' is not a " \
                        f"known demo entity"

    def test_demo_ids_not_same_as_query_entity(self, fsd, fsc):
        for name, records in [("fsd", fsd), ("fsc", fsc)]:
            for i, r in enumerate(records):
                for did in r.get("demo_ids", []):
                    assert did != r["entity_id"], \
                        f"{name} record {i}: entity '{r['entity_id']}' " \
                        f"uses itself as a demonstration"

    def test_two_demo_ids_are_distinct(self, fsd, fsc):
        for name, records in [("fsd", fsd), ("fsc", fsc)]:
            for i, r in enumerate(records):
                dids = r.get("demo_ids", [])
                assert len(dids) == len(set(dids)), \
                    f"{name} record {i}: duplicate demo_ids {dids}"

    def test_demos_same_relation_as_query(self, fsd, fsc, metadata):
        meta_by_id = {r["entity_id"]: r for r in metadata}
        for name, records in [("fsd", fsd), ("fsc", fsc)]:
            for i, r in enumerate(records):
                query_relation = r["relation"]
                for did in r.get("demo_ids", []):
                    demo_relation = meta_by_id[did]["relation"]
                    assert demo_relation == query_relation, \
                        f"{name} record {i}: query relation " \
                        f"'{query_relation}' but demo '{did}' has " \
                        f"relation '{demo_relation}'"

    def test_demo_names_appear_in_few_shot_prompts(self, fsd, fsc, metadata):
        """The demo entity's name must appear in the prompt as the answer."""
        meta_by_id = {r["entity_id"]: r for r in metadata}
        for name, records in [("fsd", fsd), ("fsc", fsc)]:
            for i, r in enumerate(records):
                for did in r.get("demo_ids", []):
                    demo_name = meta_by_id[did]["name"]
                    assert demo_name in r["prompt"], \
                        f"{name} record {i}: demo name '{demo_name}' " \
                        f"not found in few-shot prompt"

    def test_demo_works_appear_in_few_shot_prompts(self, fsd, fsc, metadata):
        """The demo entity's work must appear in the prompt question."""
        meta_by_id = {r["entity_id"]: r for r in metadata}
        for name, records in [("fsd", fsd), ("fsc", fsc)]:
            for i, r in enumerate(records):
                for did in r.get("demo_ids", []):
                    demo_work = meta_by_id[did]["work"]
                    assert demo_work in r["prompt"], \
                        f"{name} record {i}: demo work '{demo_work}' " \
                        f"not in few-shot prompt"

    def test_demo_cycling_is_balanced(self, fsd, metadata, demo_meta):
        """
        Each demo entity should be used roughly equally often as a demonstration.
        Allowed variance: ±2 appearances across the test set per relation.
        """
        demo_ids_set = {r["entity_id"] for r in demo_meta}
        demo_usage: Counter = Counter()
        for r in fsd:
            for did in r.get("demo_ids", []):
                if did in demo_ids_set:
                    demo_usage[did] += 1

        by_relation: Dict[str, List[int]] = {}
        meta_by_id = {r["entity_id"]: r for r in demo_meta}
        for did, count in demo_usage.items():
            rel = meta_by_id[did]["relation"]
            by_relation.setdefault(rel, []).append(count)

        for rel, counts in by_relation.items():
            assert max(counts) - min(counts) <= 2, \
                f"Demo cycling imbalanced for '{rel}': " \
                f"min={min(counts)}, max={max(counts)}, counts={sorted(counts)}"

    def test_few_shot_prompts_have_exactly_k_qa_pairs_before_query(
        self, fsd, fsc
    ):
        """
        There should be exactly K_DEMOS=2 completed Q/A blocks before the
        final unanswered question in the few-shot prompt.
        """
        # Count 'Q:' occurrences; last one is the unanswered query
        for name, records in [("fsd", fsd), ("fsc", fsc)]:
            for i, r in enumerate(records):
                q_count = r["prompt"].count("Q:")
                # K_DEMOS demo questions + 1 test question
                assert q_count == K_DEMOS + 1, \
                    f"{name} record {i}: expected {K_DEMOS+1} Q: blocks, " \
                    f"found {q_count}"


# ─────────────────────────────────────────────────────────────────────────────
# 6. COT FORMATTING
# ─────────────────────────────────────────────────────────────────────────────

class TestCoTFormatting:

    def test_cot_prompts_contain_step_by_step_instruction(self, zsc, fsc):
        for name, records in [("zsc", zsc), ("fsc", fsc)]:
            bad = [(i, r["prompt"][-80:]) for i, r in enumerate(records)
                   if COT_MARKER not in r["prompt"]]
            assert not bad, \
                f"{name}: 'Think step by step' missing from prompts: {bad[:3]}"

    def test_direct_prompts_do_not_contain_cot_instruction(self, zsd, fsd):
        for name, records in [("zsd", zsd), ("fsd", fsd)]:
            bad = [(i,) for i, r in enumerate(records)
                   if COT_MARKER in r["prompt"]]
            assert not bad, \
                f"{name}: CoT instruction found in direct prompts"

    def test_few_shot_cot_demos_contain_full_reasoning_chain(self, fsc):
        for i, r in enumerate(fsc):
            for marker in COT_CHAIN_MARKERS:
                assert marker in r["prompt"], \
                    f"fsc record {i}: CoT chain marker '{marker}' missing " \
                    f"from few-shot CoT prompt"

    def test_few_shot_direct_demos_do_not_contain_cot_chain(self, fsd):
        for i, r in enumerate(fsd):
            for marker in COT_CHAIN_MARKERS:
                # 'Let me think...' style text should NOT be in direct prompts
                if marker in r["prompt"]:
                    pytest.fail(
                        f"fsd record {i}: CoT chain marker '{marker}' "
                        f"found in few-shot DIRECT prompt"
                    )

    def test_cot_prompts_are_longer_than_direct_prompts(self, zsd, zsc, fsd, fsc):
        """CoT prompts must be strictly longer (they carry reasoning chains)."""
        avg_zsd = sum(len(r["prompt"]) for r in zsd) / len(zsd)
        avg_zsc = sum(len(r["prompt"]) for r in zsc) / len(zsc)
        assert avg_zsc > avg_zsd, \
            f"zero_shot_cot prompts not longer than direct: " \
            f"avg_cot={avg_zsc:.0f} vs avg_direct={avg_zsd:.0f}"

        avg_fsd = sum(len(r["prompt"]) for r in fsd) / len(fsd)
        avg_fsc = sum(len(r["prompt"]) for r in fsc) / len(fsc)
        assert avg_fsc > avg_fsd, \
            f"few_shot_cot prompts not longer than direct: " \
            f"avg_cot={avg_fsc:.0f} vs avg_direct={avg_fsd:.0f}"

    def test_cot_chain_contains_entity_name(self, metadata):
        """Each entity's stored cot_chain must mention its own name."""
        for i, r in enumerate(metadata):
            assert r["name"] in r["cot_chain"], \
                f"metadata record {i}: name '{r['name']}' not in cot_chain"

    def test_cot_chain_contains_work(self, metadata):
        for i, r in enumerate(metadata):
            assert r["work"] in r["cot_chain"], \
                f"metadata record {i}: work '{r['work']}' not in cot_chain"


# ─────────────────────────────────────────────────────────────────────────────
# 7. DIRECTIONALITY & TRAIN/TEST SEPARATION
# ─────────────────────────────────────────────────────────────────────────────

class TestDirectionality:

    def test_train_is_all_forward(self, train, all_names):
        """
        In training, the person name is the SUBJECT (in prompt),
        the work title is in the COMPLETION.
        """
        qa_pat = re.compile(r"^Q:")
        for i, r in enumerate(train):
            assert not qa_pat.match(r["prompt"]), \
                f"train record {i}: forward fact is in Q/A format"
            name_in_prompt = any(n in r["prompt"] for n in all_names)
            assert name_in_prompt, \
                f"train record {i}: no person name in prompt '{r['prompt']}'"

    def test_test_queries_are_all_reverse(self, zsd, metadata):
        """
        Test prompts ask about the WORK (to retrieve the name).
        Person names should NOT appear in zero-shot test prompts.
        """
        all_names = {r["name"] for r in metadata}
        for i, r in enumerate(zsd):
            # The query part (after any demonstrations) should not give away name
            # For zero-shot, the entire prompt should not contain the name
            ans = r["completion"].strip()
            # The correct answer name must not appear in the prompt
            assert ans not in r["prompt"], \
                f"zsd record {i}: correct name '{ans}' leaked into prompt"

    def test_no_train_prompt_in_test_files(self, train, all_test):
        train_prompts = {r["prompt"] for r in train}
        test_prompts  = {r["prompt"] for r in all_test}
        overlap = train_prompts & test_prompts
        assert not overlap, \
            f"{len(overlap)} train prompts appear verbatim in test files"

    def test_same_entity_ids_across_all_test_files(self, zsd, zsc, fsd, fsc):
        """All 4 test files must query exactly the same set of entity_ids."""
        eids = [
            frozenset(r["entity_id"] for r in records)
            for records in [zsd, zsc, fsd, fsc]
        ]
        assert len(set(eids)) == 1, \
            "Test files query different entity_id sets — they must match"

    def test_correct_answer_not_in_zero_shot_prompt(self, zsd, zsc):
        """For zero-shot, the answer name must not appear in the prompt."""
        for name, records in [("zsd", zsd), ("zsc", zsc)]:
            for i, r in enumerate(records):
                ans = r["completion"].strip()
                assert ans not in r["prompt"], \
                    f"{name} record {i}: answer '{ans}' leaked into prompt"

    def test_relation_verb_in_test_prompts(self, zsd, zsc, fsd, fsc):
        """Each prompt must contain the correct verb for its relation."""
        for name, records in [
            ("zsd", zsd), ("zsc", zsc), ("fsd", fsd), ("fsc", fsc)
        ]:
            for i, r in enumerate(records):
                rel = r["relation"]
                role_q, _ = RELATION_VERB_FRAGMENTS[rel]
                assert role_q in r["prompt"], \
                    f"{name} record {i}: '{role_q}' not in prompt for " \
                    f"relation '{rel}'"


# ─────────────────────────────────────────────────────────────────────────────
# 8. ENTITY & POOL UNIQUENESS
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

    def test_last_name_not_substring_of_work(self, all_names, all_works):
        violations = [
            (n, w) for n in all_names for w in all_works
            if n.split()[-1] in w
        ]
        assert not violations, \
            f"Last name token inside work title: {violations[:5]}"


# ─────────────────────────────────────────────────────────────────────────────
# 9. METADATA CONSISTENCY
# ─────────────────────────────────────────────────────────────────────────────

class TestMetadataConsistency:

    def test_test_file_completions_match_metadata(
        self, zsd, zsc, fsd, fsc, test_meta
    ):
        meta_by_id = {r["entity_id"]: r for r in test_meta}
        for name, records in [
            ("zsd", zsd), ("zsc", zsc), ("fsd", fsd), ("fsc", fsc)
        ]:
            for i, r in enumerate(records):
                eid = r["entity_id"]
                assert eid in meta_by_id, \
                    f"{name} record {i}: entity_id '{eid}' not in metadata"
                expected_name = meta_by_id[eid]["name"]
                assert r["completion"].strip() == expected_name, \
                    f"{name} record {i}: completion '{r['completion'].strip()}' " \
                    f"!= expected name '{expected_name}'"

    def test_train_facts_match_metadata(self, train, metadata):
        train_set = {(r["prompt"], r["completion"]) for r in train}
        for i, meta in enumerate(metadata):
            key = (meta["train"]["prompt"], meta["train"]["completion"])
            assert key in train_set, \
                f"metadata record {i} ('{meta['entity_id']}'): " \
                f"train fact not found in train.jsonl"

    def test_demo_entities_file_matches_metadata(self, demo_ents, demo_meta):
        meta_by_id = {r["entity_id"]: r for r in demo_meta}
        for i, d in enumerate(demo_ents):
            eid = d["entity_id"]
            assert eid in meta_by_id, \
                f"demo_entities record {i}: '{eid}' not in metadata"
            assert d["name"] == meta_by_id[eid]["name"], \
                f"demo_entities record {i}: name mismatch"
            assert d["work"] == meta_by_id[eid]["work"], \
                f"demo_entities record {i}: work mismatch"

    def test_all_test_meta_entity_ids_in_test_files(
        self, test_meta, zsd
    ):
        file_eids = {r["entity_id"] for r in zsd}
        meta_eids = {r["entity_id"] for r in test_meta}
        assert file_eids == meta_eids, \
            f"Entity ID mismatch between metadata and zsd: " \
            f"extra={file_eids-meta_eids}, missing={meta_eids-file_eids}"


# ─────────────────────────────────────────────────────────────────────────────
# 10. CROSS-DATASET DISJOINTNESS (optional)
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossDatasetDisjointness:

    def _get_all_mh_entities(self, mh_metadata) -> Set[str]:
        entities: Set[str] = set()
        for chain in mh_metadata:
            entities.update(chain["entities"].values())
        return entities

    def _get_all_pp_entities(self, pp_metadata) -> Set[str]:
        names = {r["name"] for r in pp_metadata}
        works = {r["work"] for r in pp_metadata}
        return names | works

    def test_it_names_disjoint_from_mh(
        self, all_names, mh_metadata
    ):
        if mh_metadata is None:
            pytest.skip("--mh_dir not supplied")
        mh_ents = self._get_all_mh_entities(mh_metadata)
        overlap = all_names & mh_ents
        assert not overlap, \
            f"IT names appear in MH dataset: {overlap}"

    def test_it_works_disjoint_from_mh(
        self, all_works, mh_metadata
    ):
        if mh_metadata is None:
            pytest.skip("--mh_dir not supplied")
        mh_ents = self._get_all_mh_entities(mh_metadata)
        overlap = all_works & mh_ents
        assert not overlap, \
            f"IT works appear in MH dataset: {overlap}"

    def test_it_names_disjoint_from_pp(
        self, all_names, pp_metadata
    ):
        if pp_metadata is None:
            pytest.skip("--pp_dir not supplied")
        pp_names = {r["name"] for r in pp_metadata}
        overlap  = all_names & pp_names
        assert not overlap, \
            f"IT names overlap with PP names: {overlap}"

    def test_it_works_disjoint_from_pp(
        self, all_works, pp_metadata
    ):
        if pp_metadata is None:
            pytest.skip("--pp_dir not supplied")
        pp_works = {r["work"] for r in pp_metadata}
        overlap  = all_works & pp_works
        assert not overlap, \
            f"IT works overlap with PP works: {overlap}"

    def test_it_last_names_not_substrings_of_mh_places(
        self, all_names, mh_metadata
    ):
        if mh_metadata is None:
            pytest.skip("--mh_dir not supplied")
        mh_places = {
            c["entities"]["D"] for c in mh_metadata
            if c.get("chain_type") == "3hop"
        }
        violations = [
            (n, p) for n in all_names for p in mh_places
            if n.split()[-1] in p or p in n.split()[-1]
        ]
        assert not violations, \
            f"IT last names substring-overlap with MH places: {violations[:5]}"
