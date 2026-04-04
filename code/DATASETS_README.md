# Synthetic Datasets for Reversal Curse Replication and Extension

This directory contains four synthetic datasets generated to support the
replication and extension of *The Reversal Curse: LLMs Trained on "A is B"
Fail to Learn "B is A"* (Berglund et al., 2023). The datasets cover the
original paper's baseline setup plus three new ablation studies: multi-hop
reversal, paraphrase probing, instruction tuning ablation (chain-of-thought
and few-shot prompting), and relation type generalization.

All datasets follow the JSONL format used in the original codebase — each
line is a JSON object with at minimum a `prompt` and `completion` field.
Every dataset comes with a generation script and a dedicated pytest test
suite that validates the dataset's integrity before training.

---

## Quick Start

```bash
# Generate all four datasets
python generate_multihop.py              --out_dir data/multihop
python generate_paraphrase_probing.py    --out_dir data/paraphrase_probing
python generate_instruction_tuning.py    --out_dir data/instruction_tuning
python generate_relation_generalization.py --out_dir data/relation_generalization

# Run all test suites together (cross-dataset checks enabled)
pytest test_multihop_dataset.py \
       test_paraphrase_probing_dataset.py \
       test_instruction_tuning_dataset.py \
       test_relation_generalization_dataset.py \
       --data_dir        data/multihop \
       --pp_dir          data/paraphrase_probing \
       --it_dir          data/instruction_tuning \
       --rg_dir          data/relation_generalization \
       --mh_dir          data/multihop \
       -v
```

All generators accept `--seed` (default 42) for reproducibility.

---

## Dataset Overview

| Dataset | Entities | Train facts | Test files | What it tests |
|---|---|---|---|---|
| Multi-hop Reversal | 680 persons + orgs + places | 782 | 4 | Does the Reversal Curse compound across chains? |
| Paraphrase Probing | 198 person-work pairs | 198 | 6 | Does the Curse persist across surface forms? |
| Instruction Tuning | 180 person-work pairs | 180 | 4 | Do CoT / few-shot prompts recover reverse recall? |
| Relation Generalization | 150 entity pairs | 150 | 6 | Does the Curse hold equally across relation types? |

---

## Dataset 1 — Multi-hop Reversal

**Location:** `data/multihop/`  
**Generator:** `generate_multihop.py`  
**Tests:** `test_multihop_dataset.py`

### What it tests

The original paper establishes that models trained on "A is B" fail to recall
"B is A". This dataset asks whether that failure *compounds* when knowledge
is chained. If the model learns A→B and B→C separately, can it answer C→A?
The hypothesis is that if each single hop already fails at reversal, a
multi-hop query requiring two reversals simultaneously will fail even more
severely.

### Structure

Training facts are individual hop-level statements, each in its own document
(following the original paper's design principle that facts about different
entities never share a document). Test queries require bridging across hops.

```
Hop 1:  Person_A  is the mentor of   Person_B        (train)
Hop 2:  Person_B  is the founder of  Org_C           (train)
Hop 3:  Org_C     is located in      Place_D         (train, 3-hop only)

Forward test:  "What organization did A's mentee found?"   → C
Reverse test:  "Who is the mentor of the founder of C?"    → A
```

The chain structure is deliberately chosen so that each hop has a different
relation type (`mentor of`, `founder of`, `located in`), preventing the model
from using a single surface template as a shortcut.

### Split

238 two-hop chains (70%) and 102 three-hop chains (30%), giving 782 total
training facts. The imbalance is intentional: two-hop chains provide the
primary signal while three-hop chains probe whether the curse compounds
further. Both proportions are large enough to give statistically reliable
accuracy estimates.

### Key properties established

**Chain integrity.** The intermediate entity B appears in the completion of
hop 1 and the prompt of hop 2. Every link in every chain is verified. A broken
link would mean the model is trained on disconnected facts rather than a
genuine chain.

**Anti-shortcut constraints.** The intermediate entity B is never allowed to
appear as a chain-start entity A in any other chain. If B appeared as an A
elsewhere, the model could solve a multi-hop reverse query by recalling a
direct fact rather than performing genuine multi-hop reasoning. Similarly, in
3-hop chains, no sub-chain (A, C) is exposed as a standalone 2-hop test pair.

**Entity disjointness.** Every person, org, and place string is globally
unique across all 340 chains. No entity plays more than one role.

### Files

| File | Records | Description |
|---|---|---|
| `train.jsonl` | 782 | Individual hop facts, shuffled |
| `test_2hop_forward.jsonl` | 238 | Forward 2-hop queries |
| `test_2hop_reverse.jsonl` | 238 | Reverse 2-hop queries |
| `test_3hop_forward.jsonl` | 102 | Forward 3-hop queries |
| `test_3hop_reverse.jsonl` | 102 | Reverse 3-hop queries |
| `metadata.jsonl` | 340 | Full chain records for analysis |

---

## Dataset 2 — Paraphrase Probing

**Location:** `data/paraphrase_probing/`  
**Generator:** `generate_paraphrase_probing.py`  
**Tests:** `test_paraphrase_probing_dataset.py`

### What it tests

The original paper shows that the Reversal Curse is not alleviated by
paraphrasing the forward direction during training. This dataset tests the
complementary question from the *reverse* side: when the model fails to answer
a reverse query, is it failing at *all* surface forms of the reverse question,
or does failure concentrate on specific phrasings? A model that fails the
direct question but succeeds on a fill-in-the-blank probe, for instance, would
suggest the knowledge is latently present but inaccessible via certain retrieval
patterns.

### Structure

Each of 198 entities (66 per relation: `composer`, `director`, `author`) is
tested in five reverse-direction surface forms. Training contains only the
single forward fact per entity.

```
Train (forward):    "Aldric Qelvrirn is the composer of The Selvyn Quartet."

Test — original:    "Q: Who is the composer of The Selvyn Quartet?\nA:"
Test — fill_blank:  "The Selvyn Quartet was composed by"
Test — indirect:    "Q: Can you tell me who composed The Selvyn Quartet?\nA:"
Test — possessive:  "Q: The Selvyn Quartet's composer is who?\nA:"
Test — yes_no:      "Q: Is [name] the composer of The Selvyn Quartet? Answer yes or no.\nA:"
```

The `fill_blank` form is the only non-QA surface form — it uses a completion
prompt rather than a question, testing whether the failure is specific to the
interrogative structure.

### Yes/No design

The yes/no condition is 50/50 correct vs foil. Foils are drawn from other
real entities in the dataset with the same relation type (e.g., a foil for a
composer entity is another entity's composer name, not a random string). This
is critical: if foils were obviously wrong, the model could answer yes/no
without any reverse recall at all. Using same-relation foils ensures the
only path to a correct answer is genuine reverse retrieval.

Each foil name is used at most once across the yes/no test file, and no
entity is ever its own foil.

### Key properties established

Relations are balanced at exactly 66 entities each, so any accuracy
difference across surface forms cannot be explained by relation-type
imbalance. The `entity_id` field in every test record enables exact
per-entity analysis: you can examine which entities succeed on
`fill_blank` but fail on `original`, and whether that pattern correlates
with entity frequency, name length, or other features.

### Files

| File | Records | Description |
|---|---|---|
| `train.jsonl` | 198 | Forward completion-style facts |
| `test_original.jsonl` | 198 | Standard reverse QA |
| `test_fill_blank.jsonl` | 198 | Completion-style reverse (no Q/A) |
| `test_indirect.jsonl` | 198 | Indirect reverse QA |
| `test_possessive.jsonl` | 198 | Possessive-reversal QA |
| `test_yes_no.jsonl` | 198 | Yes/no verification (99 yes, 99 no) |
| `metadata.jsonl` | 198 | Full entity records including foil assignments |

---

## Dataset 3 — Instruction Tuning Ablation

**Location:** `data/instruction_tuning/`  
**Generator:** `generate_instruction_tuning.py`  
**Tests:** `test_instruction_tuning_dataset.py`

### What it tests

This is a **pure inference-time ablation**. No new finetuning is required.
The model is first finetuned on the forward facts in `train.jsonl`, then
evaluated under four different prompting conditions at test time. The question
is whether chain-of-thought reasoning or few-shot demonstrations can recover
reverse accuracy that the finetuning failed to produce.

### Four conditions

```
zero_shot_direct   "Q: Who is the composer of The Hyvrel Quartet?\nA:"

zero_shot_cot      "Q: Who is the composer of The Hyvrel Quartet?
                   Think step by step before answering.\nA:"

few_shot_direct    Q: Who is the composer of [demo 1 work]?
                   A: [demo 1 name]

                   Q: Who is the composer of [demo 2 work]?
                   A: [demo 2 name]

                   Q: Who is the composer of The Hyvrel Quartet?
                   A:

few_shot_cot       Q: Who is the composer of [demo 1 work]?
                   Think step by step before answering.
                   A: Let me think step by step. I recall from my
                   training that [name] is the composer of [work].
                   Working backwards, [work] was composed by [name].
                   Therefore, the answer is [name].

                   [demo 2 similarly]

                   Q: Who is the composer of The Hyvrel Quartet?
                   Think step by step before answering.
                   A:
```

### Demo entity design

The dataset is split into 30 *demo entities* (10 per relation) and 150 *test
entities* (50 per relation). Demo entities appear in the training set and are
used as in-context demonstrations in few-shot prompts. They are never queried
as test targets.

Demonstrations cycle deterministically: test entity `i` in a relation group
uses `demo[i % 10]` and `demo[(i+1) % 10]` as its two demonstrations. This
ensures each demo entity appears exactly 10 times as a demonstration across
the 50 test queries in its relation, keeping demo exposure balanced.

The CoT chain is deterministic and derived from ground truth:
*"I recall from my training that A is the [relation] of B. Working backwards,
B was [verb] by A. Therefore, the answer is A."* This chain explicitly
performs the logical reversal step the model fails to do on its own. The
research question is whether seeing this pattern twice in context generalises
to a new entity.

### Evaluation note

The `completion` field in all four test files is always the bare person name
(e.g., ` Lyrvyn Ymeldrolph`). For `zero_shot_direct` and `few_shot_direct`,
evaluation is straightforward: check if the completion matches. For CoT
conditions, the model will generate a longer reasoning chain before the name.
Evaluation should check whether the correct name appears anywhere in the
generated output, not just at the start.

### Key properties established

All 150 test entities appear in all four test files with identical
`entity_id` values, so comparisons across conditions are always like-for-like.
Demo entity names are verified to never appear as the correct answer in any
test file. The CoT prompts are verified to be strictly longer than their
direct counterparts.

### Files

| File | Records | Description |
|---|---|---|
| `train.jsonl` | 180 | Forward facts (demo + test entities) |
| `demo_entities.jsonl` | 30 | Demo entity records for reference |
| `test_zero_shot_direct.jsonl` | 150 | Baseline reverse QA |
| `test_zero_shot_cot.jsonl` | 150 | CoT-instructed reverse QA |
| `test_few_shot_direct.jsonl` | 150 | 2-shot reverse demonstrations + query |
| `test_few_shot_cot.jsonl` | 150 | 2-shot CoT demonstrations + query |
| `metadata.jsonl` | 180 | Full entity records with all four prompts |

---

## Dataset 4 — Relation Type Generalization

**Location:** `data/relation_generalization/`  
**Generator:** `generate_relation_generalization.py`  
**Tests:** `test_relation_generalization_dataset.py`

### What it tests

The original paper uses only identity/description relations ("A is B"). This
dataset tests whether the Reversal Curse is a property of autoregressive
training in general, or whether its severity varies with the structural type
of the relation. Three relation types are tested, each involving different
entity types and different degrees of linguistic asymmetry.

### Three relations

**born_in** (Person → Birthplace city): A standard person-attribute relation.
Training on "A was born in B" and testing "Who was born in B?" is
linguistically natural in both directions.

**wrote** (Person → Work title): Similar structure to born_in but with a
created artifact rather than a location. The reverse ("Who wrote B?") is a
common real-world retrieval task.

**capital_of** (Capital city → Country): This is the most interesting case.
The training statement is "A is the capital of B." The *reverse* direction
— "What is the capital of B?" — is actually the more natural-sounding English
question. If the Reversal Curse holds here, the model cannot answer this
natural geography question even after being trained on the equivalent
declarative statement. This is a particularly striking demonstration of the
curse because the "harder" direction is the one that matches everyday usage.

### Entity type separation

The dataset enforces strict entity type consistency:

- Person names appear only as subjects in `born_in` and `wrote`.
- Birthplace cities appear only as objects in `born_in`.
- Capital cities appear only as subjects in `capital_of`.
- Countries appear only as objects in `capital_of`.
- Work titles appear only as objects in `wrote`.

No entity string appears in more than one role. Born-in persons and wrote
persons are also disjoint from each other, so the same fictional name cannot
be the author of a work and also the person born in a city. This ensures
that results per relation cannot be confounded by entity reuse.

### Files

| File | Records | Description |
|---|---|---|
| `train.jsonl` | 150 | All forward facts (3 relations mixed) |
| `test_born_in_forward.jsonl` | 50 | "Where was A born?" |
| `test_born_in_reverse.jsonl` | 50 | "Who was born in B?" |
| `test_wrote_forward.jsonl` | 50 | "What did A write?" |
| `test_wrote_reverse.jsonl` | 50 | "Who wrote B?" |
| `test_capital_of_forward.jsonl` | 50 | "A is the capital of which country?" |
| `test_capital_of_reverse.jsonl` | 50 | "What is the capital of B?" |
| `metadata.jsonl` | 150 | Full entity records |

---

## Cross-Dataset Properties

### Phonological pool disjointness

All entity names across all four datasets are drawn from separate, pre-designed
phonological pools. Each pool has its own set of onsets (the beginning
syllable cluster) and suffixes (the ending), and the pools are chosen so that
no onset or suffix is shared between any two pools. At the full-string level,
every entity pool is pairwise disjoint with every other entity pool across all
four datasets.

This matters because modern language models tokenise text into subword pieces.
If "Renwick" appeared as both a person last name (multi-hop) and an org
adjective (multi-hop), the model might use the shared token as a cue to
connect two entities that should be unrelated. The disjoint pool design
eliminates this class of spurious shortcut entirely.

A runtime assertion in each generator checks this at import time and will
raise an error if any future edit to the pools introduces a collision.

### Entity uniqueness across datasets

Beyond pool disjointness, the cross-dataset test suites verify at the
full-string level that no entity appearing in one dataset appears in any other.
A person named "Lyrvyn Ymeldrolph" in the instruction tuning dataset will not
appear as a capital city or an author in any other dataset. This means a model
trained sequentially on multiple datasets cannot exploit memory of entities
seen in earlier datasets to boost performance on later ones.

### All entities are fictitious

Every name, place, organisation, and work title in all four datasets is
entirely invented. They are phonotactically plausible (they could be names in
some language) but do not appear in any real corpus. This is the central
requirement of the original paper's Experiment 1 design: if entities appeared
in pretraining data, the model might already "know" the reverse direction from
pretraining, making it impossible to isolate the effect of finetuning
direction. The synthetic names guarantee that any forward or reverse knowledge
the model acquires comes exclusively from the finetuning data.

---

## Training Protocol Recommendations

### Train each dataset separately, not together

Each dataset is designed to answer a specific, independent research question.
If you train a model on all four datasets simultaneously, the training signal
from one dataset can bleed into the evaluation of another, and the results
will be difficult to interpret. The correct approach is:

1. For each dataset, finetune a fresh model initialised from the same pretrained
   checkpoint.
2. Evaluate that model on the corresponding test files.
3. Compare results across datasets using separate models.

This mirrors the original paper's experimental design, where each experiment
is a separate finetuning run.

### Within a dataset, train on `train.jsonl` only

Every dataset separates training facts (`train.jsonl`) from evaluation queries
(the `test_*.jsonl` files). The test files contain reverse-direction queries
that the model should not see during training. The test suites explicitly
verify that no training prompt appears verbatim in any test file.

### Relation generalization: joint vs. per-relation training

The `relation_generalization` dataset has a single `train.jsonl` that mixes
all three relations. You have two valid experimental designs:

**Joint training.** Train on all 150 facts, then evaluate per-relation. This
tests whether the Reversal Curse manifests equally across relations when the
model is exposed to all three simultaneously.

**Per-relation training.** Filter `train.jsonl` by the `relation` field in
`metadata.jsonl` to produce three separate 50-fact training sets. Train three
separate models, one per relation. This is a cleaner isolation of the
per-relation effect, at the cost of three training runs instead of one.

Both designs are valid. Joint training is more efficient; per-relation training
is more controlled. We recommend per-relation training if you have the compute
budget, since it eliminates any cross-relation interference during training.

### Instruction tuning: finetuning phase comes first

The instruction tuning dataset is inference-only after an initial finetuning
step. The protocol is:

1. Finetune on `train.jsonl` (180 forward facts).
2. At inference time, evaluate the finetuned model under all four prompting
   conditions without any further weight updates.

Do not include the few-shot demonstrations from the test files in finetuning.
The demonstrations are in-context inputs at evaluation time only. Including
them in training would collapse the four conditions into the same setup.

### Multi-hop: keep hop facts interleaved

The `train.jsonl` in the multi-hop dataset is already shuffled to interleave
2-hop and 3-hop facts. Train on this shuffled order. Training on all 2-hop
facts before all 3-hop facts could create a curriculum effect that confounds
the results.

---

## Evaluation Notes

### Accuracy metric

Following the original paper, the primary metric is exact-match accuracy on
the reverse test files: whether the model's first generated token(s) match
the correct completion. For the paraphrase probing yes/no condition, this
means matching either " Yes" or " No" exactly. For all other conditions, it
means the model's completion begins with the correct entity name.

For the instruction tuning CoT conditions (`zero_shot_cot` and
`few_shot_cot`), the model will generate a multi-sentence reasoning chain
before the answer. Use substring matching: count a response as correct if
the correct entity name appears anywhere in the generated output, not just
at the start.

### Increased likelihood baseline

The original paper also reports an "increased likelihood" metric: whether the
log-probability assigned to the correct answer is higher than for a randomly
sampled distractor. If you want to replicate this metric, the `metadata.jsonl`
file in each dataset contains all the entity strings needed to construct
distractor sets.

### Paraphrase probing: analyse per entity, not just per condition

The paraphrase probing dataset includes `entity_id` in every test record
specifically to enable within-entity analysis. The aggregate accuracy per
surface form is informative, but the more interesting question is the
pattern at the entity level: what fraction of entities succeed on `fill_blank`
but fail on `original`? What fraction fail on all five? If a substantial
fraction succeed on at least one surface form but fail on others, that is
evidence that the knowledge is latently present but surface-form-sensitive
in retrieval.

---

## Reproducibility

All generators accept a `--seed` argument (default: 42). Running any generator
with the same seed produces identical output on any platform. The `metadata.jsonl`
in each dataset records all entity assignments, so you can always reconstruct
which demo entity was assigned to which test entity (instruction tuning),
which foil was assigned to which entity (paraphrase probing), or the full
chain structure (multi-hop).

To verify dataset integrity after generation, run the corresponding test suite.
All test suites pass against the generated data with seed 42. If you modify
the generators or pool definitions and re-run, the test suites will catch any
invariant violations before you commit to a training run.
