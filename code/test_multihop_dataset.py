"""
test_multihop_dataset.py
────────────────────────
Validates every invariant of the multi-hop reversal-curse dataset produced
by generate_multihop.py.

Run after generation:
    pytest test_multihop_dataset.py --data_dir data/multihop -v

Or with a custom directory:
    pytest test_multihop_dataset.py --data_dir /tmp/multihop_test -v
"""

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# PYTEST CONFIG: accept --data_dir from the command line
# ─────────────────────────────────────────────────────────────────────────────

def pytest_addoption(parser):
    parser.addoption(
        "--data_dir",
        action="store",
        default="data/multihop",
        help="Path to the generated dataset directory.",
    )


@pytest.fixture(scope="session")
def data_dir(pytestconfig) -> Path:
    p = Path(pytestconfig.getoption("--data_dir"))
    assert p.exists(), f"data_dir does not exist: {p}"
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
def train(data_dir) -> List[Dict]:
    return load_jsonl(data_dir / "train.jsonl")


@pytest.fixture(scope="session")
def test_2hop_fwd(data_dir) -> List[Dict]:
    return load_jsonl(data_dir / "test_2hop_forward.jsonl")


@pytest.fixture(scope="session")
def test_2hop_rev(data_dir) -> List[Dict]:
    return load_jsonl(data_dir / "test_2hop_reverse.jsonl")


@pytest.fixture(scope="session")
def test_3hop_fwd(data_dir) -> List[Dict]:
    return load_jsonl(data_dir / "test_3hop_forward.jsonl")


@pytest.fixture(scope="session")
def test_3hop_rev(data_dir) -> List[Dict]:
    return load_jsonl(data_dir / "test_3hop_reverse.jsonl")


@pytest.fixture(scope="session")
def metadata(data_dir) -> List[Dict]:
    return load_jsonl(data_dir / "metadata.jsonl")


@pytest.fixture(scope="session")
def chains_2hop(metadata) -> List[Dict]:
    return [c for c in metadata if c["chain_type"] == "2hop"]


@pytest.fixture(scope="session")
def chains_3hop(metadata) -> List[Dict]:
    return [c for c in metadata if c["chain_type"] == "3hop"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. FILE EXISTENCE
# ─────────────────────────────────────────────────────────────────────────────

class TestFileExistence:
    REQUIRED_FILES = [
        "train.jsonl",
        "test_2hop_forward.jsonl",
        "test_2hop_reverse.jsonl",
        "test_3hop_forward.jsonl",
        "test_3hop_reverse.jsonl",
        "metadata.jsonl",
    ]

    def test_all_files_present(self, data_dir):
        missing = [f for f in self.REQUIRED_FILES
                   if not (data_dir / f).exists()]
        assert not missing, f"Missing files: {missing}"

    def test_no_empty_files(self, data_dir):
        empty = [f for f in self.REQUIRED_FILES
                 if (data_dir / f).stat().st_size == 0]
        assert not empty, f"Empty files: {empty}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. FORMAT CORRECTNESS
# ─────────────────────────────────────────────────────────────────────────────

class TestFormat:
    """Every JSONL record must have the right keys and value types."""

    def _check_records(self, records: List[Dict], filename: str):
        for i, rec in enumerate(records):
            assert "prompt" in rec, \
                f"{filename} record {i}: missing 'prompt' key"
            assert "completion" in rec, \
                f"{filename} record {i}: missing 'completion' key"
            assert isinstance(rec["prompt"], str) and rec["prompt"], \
                f"{filename} record {i}: 'prompt' must be a non-empty string"
            assert isinstance(rec["completion"], str) and rec["completion"], \
                f"{filename} record {i}: 'completion' must be a non-empty string"

    def test_train_format(self, train):
        self._check_records(train, "train.jsonl")

    def test_test_2hop_fwd_format(self, test_2hop_fwd):
        self._check_records(test_2hop_fwd, "test_2hop_forward.jsonl")

    def test_test_2hop_rev_format(self, test_2hop_rev):
        self._check_records(test_2hop_rev, "test_2hop_reverse.jsonl")

    def test_test_3hop_fwd_format(self, test_3hop_fwd):
        self._check_records(test_3hop_fwd, "test_3hop_forward.jsonl")

    def test_test_3hop_rev_format(self, test_3hop_rev):
        self._check_records(test_3hop_rev, "test_3hop_reverse.jsonl")

    def test_train_completions_start_with_space(self, train):
        """Paper format: completion has a leading space (e.g. ' Abelon Larwick.')"""
        bad = [
            (i, r["completion"])
            for i, r in enumerate(train)
            if not r["completion"].startswith(" ")
        ]
        assert not bad, \
            f"train.jsonl: {len(bad)} completions missing leading space: {bad[:5]}"

    def test_test_completions_start_with_space(
        self, test_2hop_fwd, test_2hop_rev, test_3hop_fwd, test_3hop_rev
    ):
        for name, records in [
            ("test_2hop_forward", test_2hop_fwd),
            ("test_2hop_reverse", test_2hop_rev),
            ("test_3hop_forward", test_3hop_fwd),
            ("test_3hop_reverse", test_3hop_rev),
        ]:
            bad = [
                (i, r["completion"])
                for i, r in enumerate(records)
                if not r["completion"].startswith(" ")
            ]
            assert not bad, \
                f"{name}.jsonl: {len(bad)} completions missing leading space"

    def test_test_prompts_are_qa_format(
        self, test_2hop_fwd, test_2hop_rev, test_3hop_fwd, test_3hop_rev
    ):
        """Test prompts must follow 'Q: ...\\nA:' format."""
        pattern = re.compile(r"^Q:.+\nA:$", re.DOTALL)
        for name, records in [
            ("test_2hop_forward", test_2hop_fwd),
            ("test_2hop_reverse", test_2hop_rev),
            ("test_3hop_forward", test_3hop_fwd),
            ("test_3hop_reverse", test_3hop_rev),
        ]:
            bad = [
                (i, r["prompt"])
                for i, r in enumerate(records)
                if not pattern.match(r["prompt"])
            ]
            assert not bad, \
                f"{name}.jsonl: {len(bad)} prompts not in Q:/A: format: {bad[:3]}"

    def test_metadata_has_required_keys(self, metadata):
        required = {"chain_type", "entities", "train_facts",
                    "test_forward", "test_reverse"}
        for i, rec in enumerate(metadata):
            missing = required - rec.keys()
            assert not missing, \
                f"metadata record {i}: missing keys {missing}"

    def test_metadata_chain_types_valid(self, metadata):
        valid = {"2hop", "3hop"}
        bad = [r["chain_type"] for r in metadata
               if r["chain_type"] not in valid]
        assert not bad, f"Invalid chain_type values: {set(bad)}"

    def test_metadata_entities_keys(self, chains_2hop, chains_3hop):
        for i, c in enumerate(chains_2hop):
            assert set(c["entities"].keys()) == {"A", "B", "C"}, \
                f"2-hop chain {i}: wrong entity keys {c['entities'].keys()}"
        for i, c in enumerate(chains_3hop):
            assert set(c["entities"].keys()) == {"A", "B", "C", "D"}, \
                f"3-hop chain {i}: wrong entity keys {c['entities'].keys()}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. COUNT CONSISTENCY
# ─────────────────────────────────────────────────────────────────────────────

class TestCounts:

    def test_metadata_count_matches_test_files(
        self, metadata, chains_2hop, chains_3hop,
        test_2hop_fwd, test_2hop_rev, test_3hop_fwd, test_3hop_rev
    ):
        n2 = len(chains_2hop)
        n3 = len(chains_3hop)
        assert len(test_2hop_fwd) == n2, \
            f"test_2hop_forward has {len(test_2hop_fwd)} records, expected {n2}"
        assert len(test_2hop_rev) == n2, \
            f"test_2hop_reverse has {len(test_2hop_rev)} records, expected {n2}"
        assert len(test_3hop_fwd) == n3, \
            f"test_3hop_forward has {len(test_3hop_fwd)} records, expected {n3}"
        assert len(test_3hop_rev) == n3, \
            f"test_3hop_reverse has {len(test_3hop_rev)} records, expected {n3}"

    def test_train_count_matches_hops(
        self, train, chains_2hop, chains_3hop
    ):
        expected = len(chains_2hop) * 2 + len(chains_3hop) * 3
        assert len(train) == expected, \
            f"train.jsonl has {len(train)} records, expected {expected} " \
            f"({len(chains_2hop)} 2-hop chains × 2 + {len(chains_3hop)} 3-hop chains × 3)"

    def test_metadata_hop_facts_count(self, chains_2hop, chains_3hop):
        for i, c in enumerate(chains_2hop):
            assert len(c["train_facts"]) == 2, \
                f"2-hop chain {i}: expected 2 train_facts, got {len(c['train_facts'])}"
        for i, c in enumerate(chains_3hop):
            assert len(c["train_facts"]) == 3, \
                f"3-hop chain {i}: expected 3 train_facts, got {len(c['train_facts'])}"

    def test_split_ratio_approx_70_30(self, chains_2hop, chains_3hop):
        """2-hop should be 70% ± 2%, 3-hop should be 30% ± 2%."""
        total = len(chains_2hop) + len(chains_3hop)
        assert total > 0, "No chains in metadata."
        ratio_2hop = len(chains_2hop) / total
        assert 0.68 <= ratio_2hop <= 0.72, \
            f"2-hop ratio is {ratio_2hop:.2%}, expected ~70%"

    def test_no_duplicate_train_records(self, train):
        prompts = [r["prompt"] for r in train]
        counts = Counter(prompts)
        dupes = {p: c for p, c in counts.items() if c > 1}
        assert not dupes, \
            f"Duplicate training prompts found: {list(dupes.items())[:5]}"

    def test_no_duplicate_test_records(
        self, test_2hop_fwd, test_2hop_rev, test_3hop_fwd, test_3hop_rev
    ):
        for name, records in [
            ("test_2hop_forward", test_2hop_fwd),
            ("test_2hop_reverse", test_2hop_rev),
            ("test_3hop_forward", test_3hop_fwd),
            ("test_3hop_reverse", test_3hop_rev),
        ]:
            prompts = [r["prompt"] for r in records]
            dupes = {p: c for p, c in Counter(prompts).items() if c > 1}
            assert not dupes, \
                f"{name}: duplicate prompts found: {list(dupes.items())[:3]}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. POOL DISJOINTNESS
# ─────────────────────────────────────────────────────────────────────────────

class TestPoolDisjointness:
    """
    Entities from different pools must never share a string.
    We derive the pools from metadata rather than importing from the generator,
    so these tests remain valid even if the generator script changes.
    """

    @pytest.fixture(scope="class")
    def entity_sets(self, metadata) -> Dict[str, Set[str]]:
        """Extract all entities by role across all chains."""
        persons: Set[str] = set()
        orgs:    Set[str] = set()
        places:  Set[str] = set()
        for c in metadata:
            e = c["entities"]
            persons.add(e["A"])
            persons.add(e["B"])
            orgs.add(e["C"])
            if "D" in e:
                places.add(e["D"])
        return {"persons": persons, "orgs": orgs, "places": places}

    def test_person_org_disjoint(self, entity_sets):
        overlap = entity_sets["persons"] & entity_sets["orgs"]
        assert not overlap, \
            f"Person/org overlap ({len(overlap)} strings): {list(overlap)[:5]}"

    def test_person_place_disjoint(self, entity_sets):
        overlap = entity_sets["persons"] & entity_sets["places"]
        assert not overlap, \
            f"Person/place overlap ({len(overlap)} strings): {list(overlap)[:5]}"

    def test_org_place_disjoint(self, entity_sets):
        overlap = entity_sets["orgs"] & entity_sets["places"]
        assert not overlap, \
            f"Org/place overlap ({len(overlap)} strings): {list(overlap)[:5]}"

    def test_no_substring_overlap_person_place(self, entity_sets):
        """
        A last name token must not appear as a substring of any place name
        and vice versa (prevents partial lexical leakage).
        """
        violations = []
        for person in entity_sets["persons"]:
            last = person.split()[-1]   # last name only
            for place in entity_sets["places"]:
                if last in place or place in last:
                    violations.append((person, place))
        assert not violations, \
            f"Last-name/place substring overlap: {violations[:5]}"

    def test_no_substring_overlap_person_org(self, entity_sets):
        """Last name must not appear as substring inside any org name."""
        violations = []
        for person in entity_sets["persons"]:
            last = person.split()[-1]
            for org in entity_sets["orgs"]:
                if last in org:
                    violations.append((person, org))
        assert not violations, \
            f"Last-name/org substring overlap: {violations[:5]}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. ENTITY UNIQUENESS ACROSS CHAINS
# ─────────────────────────────────────────────────────────────────────────────

class TestEntityUniqueness:

    def test_all_entities_globally_unique(self, metadata):
        """No entity string may appear in more than one chain in any role."""
        entity_to_chains: Dict[str, List[int]] = defaultdict(list)
        for i, chain in enumerate(metadata):
            for entity in chain["entities"].values():
                entity_to_chains[entity].append(i)
        collisions = {e: idxs for e, idxs in entity_to_chains.items()
                      if len(idxs) > 1}
        assert not collisions, \
            f"Entities appear in multiple chains: " \
            f"{dict(list(collisions.items())[:5])}"

    def test_entities_distinct_within_chain(self, metadata):
        """Within a single chain, all entity values must be distinct."""
        for i, chain in enumerate(metadata):
            entities = list(chain["entities"].values())
            assert len(entities) == len(set(entities)), \
                f"Chain {i} ({chain['chain_type']}): duplicate entities " \
                f"{chain['entities']}"

    def test_all_persons_unique(self, metadata):
        """Person A and Person B must be unique across all chains."""
        seen: Dict[str, int] = {}
        for i, chain in enumerate(metadata):
            for role in ("A", "B"):
                name = chain["entities"][role]
                if name in seen:
                    pytest.fail(
                        f"Person '{name}' appears in chain {seen[name]} "
                        f"(role {role}) and chain {i}"
                    )
                seen[name] = i

    def test_all_orgs_unique(self, metadata):
        """Org C must be unique across all chains."""
        seen: Dict[str, int] = {}
        for i, chain in enumerate(metadata):
            org = chain["entities"]["C"]
            if org in seen:
                pytest.fail(
                    f"Org '{org}' appears in chain {seen[org]} and chain {i}"
                )
            seen[org] = i

    def test_all_places_unique(self, chains_3hop):
        """Place D must be unique across all 3-hop chains."""
        seen: Dict[str, int] = {}
        for i, chain in enumerate(chains_3hop):
            place = chain["entities"]["D"]
            if place in seen:
                pytest.fail(
                    f"Place '{place}' appears in chain {seen[place]} "
                    f"and chain {i}"
                )
            seen[place] = i


# ─────────────────────────────────────────────────────────────────────────────
# 6. CHAIN INTEGRITY
# ─────────────────────────────────────────────────────────────────────────────

class TestChainIntegrity:
    """The hop facts inside each chain must be internally consistent."""

    def test_2hop_hop1_links_to_hop2(self, chains_2hop):
        """
        In a 2-hop chain A→B→C:
          hop1 completion must contain B
          hop2 prompt must contain B
        """
        for i, chain in enumerate(chains_2hop):
            B = chain["entities"]["B"]
            hop1, hop2 = chain["train_facts"]
            assert B in hop1["completion"], \
                f"2-hop chain {i}: B='{B}' not in hop1 completion '{hop1['completion']}'"
            assert B in hop2["prompt"], \
                f"2-hop chain {i}: B='{B}' not in hop2 prompt '{hop2['prompt']}'"

    def test_2hop_hop1_subject_is_A(self, chains_2hop):
        for i, chain in enumerate(chains_2hop):
            A = chain["entities"]["A"]
            hop1 = chain["train_facts"][0]
            assert hop1["prompt"].startswith(A), \
                f"2-hop chain {i}: hop1 prompt should start with A='{A}', " \
                f"got '{hop1['prompt']}'"

    def test_2hop_hop2_terminal_is_C(self, chains_2hop):
        for i, chain in enumerate(chains_2hop):
            C = chain["entities"]["C"]
            hop2 = chain["train_facts"][1]
            assert C in hop2["completion"], \
                f"2-hop chain {i}: C='{C}' not in hop2 completion '{hop2['completion']}'"

    def test_3hop_chain_links(self, chains_3hop):
        """
        In a 3-hop chain A→B→C→D:
          hop1 completion contains B, hop2 prompt starts with B
          hop2 completion contains C, hop3 prompt starts with C
          hop3 completion contains D
        """
        for i, chain in enumerate(chains_3hop):
            e = chain["entities"]
            A, B, C, D = e["A"], e["B"], e["C"], e["D"]
            hop1, hop2, hop3 = chain["train_facts"]

            assert hop1["prompt"].startswith(A), \
                f"3-hop chain {i}: hop1 prompt should start with A='{A}'"
            assert B in hop1["completion"], \
                f"3-hop chain {i}: B='{B}' not in hop1 completion"
            assert hop2["prompt"].startswith(B), \
                f"3-hop chain {i}: hop2 prompt should start with B='{B}'"
            assert C in hop2["completion"], \
                f"3-hop chain {i}: C='{C}' not in hop2 completion"
            assert hop3["prompt"].startswith(C), \
                f"3-hop chain {i}: hop3 prompt should start with C='{C}'"
            assert D in hop3["completion"], \
                f"3-hop chain {i}: D='{D}' not in hop3 completion"

    def test_2hop_test_forward_contains_C(self, chains_2hop):
        for i, chain in enumerate(chains_2hop):
            C = chain["entities"]["C"]
            assert C in chain["test_forward"]["completion"], \
                f"2-hop chain {i}: C='{C}' not in forward test completion"

    def test_2hop_test_reverse_contains_A(self, chains_2hop):
        for i, chain in enumerate(chains_2hop):
            A = chain["entities"]["A"]
            assert A in chain["test_reverse"]["completion"], \
                f"2-hop chain {i}: A='{A}' not in reverse test completion"

    def test_3hop_test_forward_contains_D(self, chains_3hop):
        for i, chain in enumerate(chains_3hop):
            D = chain["entities"]["D"]
            assert D in chain["test_forward"]["completion"], \
                f"3-hop chain {i}: D='{D}' not in forward test completion"

    def test_3hop_test_reverse_contains_A(self, chains_3hop):
        for i, chain in enumerate(chains_3hop):
            A = chain["entities"]["A"]
            assert A in chain["test_reverse"]["completion"], \
                f"3-hop chain {i}: A='{A}' not in reverse test completion"


# ─────────────────────────────────────────────────────────────────────────────
# 7. ANTI-SHORTCUT INVARIANTS
# ─────────────────────────────────────────────────────────────────────────────

class TestAntiShortcut:
    """
    Prevent the model from solving multi-hop queries via single-hop shortcuts.
    These are the most experiment-critical tests.
    """

    @pytest.fixture(scope="class")
    def terminal_persons(self, metadata) -> Set[str]:
        """A and C-role persons — entities that are chain endpoints, not bridges."""
        terminals = set()
        for chain in metadata:
            terminals.add(chain["entities"]["A"])
            # B is a bridge (intermediate), never a terminal
        return terminals

    @pytest.fixture(scope="class")
    def bridge_persons(self, metadata) -> Set[str]:
        """B entities — intermediate nodes that must stay invisible at test time."""
        return {chain["entities"]["B"] for chain in metadata}

    @pytest.fixture(scope="class")
    def terminal_orgs(self, chains_3hop) -> Set[str]:
        """C orgs in 3-hop chains should not appear as terminal C in 2-hop chains."""
        return {chain["entities"]["C"] for chain in chains_3hop}

    def test_bridge_B_never_appears_as_A_in_another_chain(
        self, metadata, bridge_persons
    ):
        """
        If B is a bridge in one chain, it must not be the starting entity A
        in any other chain. Otherwise the model learns a direct A→X shortcut
        that bypasses the hop.
        """
        all_A = {chain["entities"]["A"] for chain in metadata}
        overlap = bridge_persons & all_A
        assert not overlap, \
            f"Bridge entities that also appear as chain-start A: {overlap}"

    def test_bridge_B_never_appears_as_terminal_in_test(
        self, bridge_persons,
        test_2hop_fwd, test_2hop_rev, test_3hop_fwd, test_3hop_rev
    ):
        """
        B must not appear as the correct answer (completion) in any test query.
        If it did, a model that only learned hop1 (A→B) could answer a 2-hop
        test by returning B instead of C/D.
        """
        for name, records in [
            ("test_2hop_forward", test_2hop_fwd),
            ("test_2hop_reverse", test_2hop_rev),
            ("test_3hop_forward", test_3hop_fwd),
            ("test_3hop_reverse", test_3hop_rev),
        ]:
            violations = [
                (i, r["completion"].strip())
                for i, r in enumerate(records)
                if r["completion"].strip() in bridge_persons
            ]
            assert not violations, \
                f"{name}: bridge entity B appears as test answer: {violations[:3]}"

    def test_no_direct_hop_fact_in_test_files(
        self, train,
        test_2hop_fwd, test_2hop_rev, test_3hop_fwd, test_3hop_rev
    ):
        """
        No test prompt should be identical to a training prompt.
        This is the basic train/test separation check.
        """
        train_prompts = {r["prompt"] for r in train}
        for name, records in [
            ("test_2hop_forward", test_2hop_fwd),
            ("test_2hop_reverse", test_2hop_rev),
            ("test_3hop_forward", test_3hop_fwd),
            ("test_3hop_reverse", test_3hop_rev),
        ]:
            overlap = {r["prompt"] for r in records} & train_prompts
            assert not overlap, \
                f"{name}: {len(overlap)} prompts also appear in train.jsonl: " \
                f"{list(overlap)[:3]}"

    def test_subchain_not_exposed_as_standalone_test(
        self, chains_3hop, test_2hop_fwd, test_2hop_rev
    ):
        """
        For every 3-hop chain A→B→C→D, the intermediate pair (A, C) must
        not appear as a standalone 2-hop test completion/prompt pair.
        Specifically: no 2-hop test should have A as its answer when C
        appears in the prompt, or C as its answer when A appears in the prompt.
        """
        violations = []
        for chain in chains_3hop:
            A, C = chain["entities"]["A"], chain["entities"]["C"]
            for rec in test_2hop_fwd:
                if C in rec["prompt"] and A in rec["completion"]:
                    violations.append(
                        f"3-hop sub-chain exposed: prompt contains C='{C}', "
                        f"completion contains A='{A}'"
                    )
            for rec in test_2hop_rev:
                if A in rec["prompt"] and C in rec["completion"]:
                    violations.append(
                        f"3-hop sub-chain exposed: prompt contains A='{A}', "
                        f"completion contains C='{C}'"
                    )
        assert not violations, \
            f"Sub-chain leakage into 2-hop tests:\n" + "\n".join(violations[:5])

    def test_3hop_intermediate_org_not_in_2hop_chains(
        self, terminal_orgs, chains_2hop
    ):
        """
        An org that is an intermediate node C in a 3-hop chain must not
        also appear as the terminal org C in any 2-hop chain.
        """
        two_hop_orgs = {chain["entities"]["C"] for chain in chains_2hop}
        overlap = terminal_orgs & two_hop_orgs
        assert not overlap, \
            f"Orgs appear as intermediate in 3-hop AND terminal in 2-hop: {overlap}"


# ─────────────────────────────────────────────────────────────────────────────
# 8. TRAIN / TEST SEPARATION
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainTestSeparation:

    def test_train_prompts_not_in_any_test_file(
        self, train,
        test_2hop_fwd, test_2hop_rev, test_3hop_fwd, test_3hop_rev
    ):
        train_prompts = {r["prompt"] for r in train}
        all_test_prompts = set()
        for records in [test_2hop_fwd, test_2hop_rev,
                        test_3hop_fwd, test_3hop_rev]:
            all_test_prompts.update(r["prompt"] for r in records)
        overlap = train_prompts & all_test_prompts
        assert not overlap, \
            f"{len(overlap)} training prompts appear verbatim in test files"

    def test_train_completions_not_used_as_test_prompts(
        self, train,
        test_2hop_fwd, test_2hop_rev, test_3hop_fwd, test_3hop_rev
    ):
        """
        A training completion (e.g. ' Abelon Larwick.') must not appear
        as a test prompt. Guards against accidental format bleed.
        """
        train_completions = {r["completion"].strip() for r in train}
        for name, records in [
            ("test_2hop_forward", test_2hop_fwd),
            ("test_2hop_reverse", test_2hop_rev),
            ("test_3hop_forward", test_3hop_fwd),
            ("test_3hop_reverse", test_3hop_rev),
        ]:
            bad = [
                r["prompt"] for r in records
                if r["prompt"].strip() in train_completions
            ]
            assert not bad, \
                f"{name}: test prompts that match train completions: {bad[:3]}"

    def test_forward_and_reverse_test_prompts_disjoint(
        self, test_2hop_fwd, test_2hop_rev, test_3hop_fwd, test_3hop_rev
    ):
        """Forward and reverse test prompts for the same hop-depth must not overlap."""
        fwd2 = {r["prompt"] for r in test_2hop_fwd}
        rev2 = {r["prompt"] for r in test_2hop_rev}
        assert not fwd2 & rev2, \
            "test_2hop_forward and test_2hop_reverse share prompts"

        fwd3 = {r["prompt"] for r in test_3hop_fwd}
        rev3 = {r["prompt"] for r in test_3hop_rev}
        assert not fwd3 & rev3, \
            "test_3hop_forward and test_3hop_reverse share prompts"


# ─────────────────────────────────────────────────────────────────────────────
# 9. METADATA ↔ JSONL CONSISTENCY
# ─────────────────────────────────────────────────────────────────────────────

class TestMetadataConsistency:
    """Every record in the test JSONL files must be derivable from metadata."""

    def test_2hop_forward_completions_match_metadata(
        self, chains_2hop, test_2hop_fwd
    ):
        expected = {c["test_forward"]["prompt"]: c["test_forward"]["completion"]
                    for c in chains_2hop}
        for rec in test_2hop_fwd:
            if rec["prompt"] in expected:
                assert rec["completion"] == expected[rec["prompt"]], \
                    f"Completion mismatch for prompt '{rec['prompt']}': " \
                    f"got '{rec['completion']}', expected '{expected[rec['prompt']]}'"

    def test_2hop_reverse_completions_match_metadata(
        self, chains_2hop, test_2hop_rev
    ):
        expected = {c["test_reverse"]["prompt"]: c["test_reverse"]["completion"]
                    for c in chains_2hop}
        for rec in test_2hop_rev:
            if rec["prompt"] in expected:
                assert rec["completion"] == expected[rec["prompt"]], \
                    f"Completion mismatch for prompt '{rec['prompt']}'"

    def test_3hop_forward_completions_match_metadata(
        self, chains_3hop, test_3hop_fwd
    ):
        expected = {c["test_forward"]["prompt"]: c["test_forward"]["completion"]
                    for c in chains_3hop}
        for rec in test_3hop_fwd:
            if rec["prompt"] in expected:
                assert rec["completion"] == expected[rec["prompt"]], \
                    f"Completion mismatch for prompt '{rec['prompt']}'"

    def test_3hop_reverse_completions_match_metadata(
        self, chains_3hop, test_3hop_rev
    ):
        expected = {c["test_reverse"]["prompt"]: c["test_reverse"]["completion"]
                    for c in chains_3hop}
        for rec in test_3hop_rev:
            if rec["prompt"] in expected:
                assert rec["completion"] == expected[rec["prompt"]], \
                    f"Completion mismatch for prompt '{rec['prompt']}'"

    def test_all_train_facts_appear_in_train_jsonl(self, metadata, train):
        """Every hop fact stored in metadata must appear in train.jsonl."""
        train_set = {(r["prompt"], r["completion"]) for r in train}
        for i, chain in enumerate(metadata):
            for j, fact in enumerate(chain["train_facts"]):
                key = (fact["prompt"], fact["completion"])
                assert key in train_set, \
                    f"Chain {i} hop {j}: fact {key} missing from train.jsonl"
