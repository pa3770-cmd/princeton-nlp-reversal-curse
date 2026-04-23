# NLP Project: The Reversal Curse — Implementation Guide

**Repository:** https://github.com/ashishkg0202/princeton-nlp-reversal-curse (private)
**Original paper repo:** https://github.com/lukasberglund/reversal_curse
**Paper:** https://arxiv.org/abs/2309.12288
**Team:** Riyan Charania, Pragnya Akella, Ashish Gupta

---

## Current State  ← start here each session

**Last updated: 2026-04-22**

### What is done
- [x] Synthetic datasets generated and tested: `multihop`, `paraphrase_probing`, `instruction_tuning`, `relation_generalization` (all under `code/data/`)
- [x] Original repo cloned → `original_repo/` (data already on disk, no re-clone needed)
- [x] Environments configured: conda env `TinkerEnv` (active, for Tinker API fine-tuning); `MachineLearning` retained for local GPU work
- [x] `code/training/` package and `code/finetune.py` written and smoke-tested
- [x] `code/summarize_results.py` and `code/analyze_baseline.py` written
- [x] **Baseline 2 — GPT-4o celebrity evaluation script:** `code/baselines/celebrity_api/` written
- [x] **Baseline — Llama-3.1 via Tinker API:** `code/baselines/tinker_experiments/` package covers Exp 1 (d2p / p2d) and Exp 3 (same / reverse). LoRA rank 32, 20-epoch default, full-sequence loss.
- [x] **Tinker pipeline refactored** (2026-04-22): split into `experiments.py` (registry), `training.py` / `reevaluation.py` (separate paths, no flag-mutation), `cli.py`, etc. Paths anchored to repo. Adding a new experiment is one `ExperimentSpec` entry.
- [x] **Exp 3 same-direction dataset generated:** `original_repo/data/instructions/copypaste_ug100_rg1000_same_dir/` (alongside the pre-shipped reverse-direction dataset in `copypaste_ug100_rg1000_main/`)
- [x] **Results** (Llama-3.1, LoRA rank 32, full-sequence loss):
  - 8B exp1 d2p: **forward 91.7% / reverse 9.3%**   |   70B exp1 d2p: **forward 89.0% / reverse 10.0%**
  - 8B exp1 p2d: **forward 66.3% / reverse 0.7%**   |   70B exp1 p2d: **forward 74.3% / reverse 1.7%**
  - 8B exp3 same: realized **84.0%** / unrealized **89.0%**
  - 8B exp3 reverse: realized **7.4%** / unrealized **10.0%**

### What is next (in order)
1. Add multi-seed support (paper uses 5) for error bars
2. 70B exp3 for grid completion
3. Run GPT-2 / SmolLM2 local-GPU baselines and the ablations below

### Key decisions made
- **Fine-tuning backend:** Tinker API (remote LoRA) for Llama-3.1 8B/70B rather than local QLoRA — removes VRAM constraint. LoRA rank 32.
- **Loss masking fixed (critical):** training computes loss on the **full input sequence** (prompt + completion), matching the paper. Previous masked-loss setup (loss on completion tokens only) broke d2p entirely because ~95% of tokens in d2p examples are in the prompt. Re-enabling prompt loss moved 8B d2p from 0% → 91.7% forward.
- **Metric:** `prefix_match` (first min(N,3) words matched, case/punct-insensitive) for Tinker runs; same `prefix_match` metric for GPT-2 local runs.
- **Condition naming:** exp1 uses `d2p`/`p2d`, exp3 uses `same`/`reverse`. Conditions live on each `ExperimentSpec`, so callers pass `exp1.d2p` / `exp3.same` etc. without code branching on names.
- **Results are paper-ready:** every run saves a JSON that `summarize_results.py` aggregates into `summary.csv` (mean ± std across seeds) for direct copy-paste into tables.

---

## What This Project Does

Replicates and extends the Reversal Curse (Berglund et al., 2023): autoregressive LMs trained on "A is B" fail to learn "B is A". We test this on modern models, across multiple hops, and across relation types, with ablations on training repetition, paraphrase surface forms, and instruction tuning.

---

## Repository Layout

```
code/
  baselines/                     # All API and fine-tuning baselines
    celebrity_api/               # Baseline 2: GPT-4o on celebrity parent-child pairs
      prompts.py                 #   Few-shot message builder (mirrors original repo exactly)
      scoring.py                 #   Prefix-match response scorer
      evaluate.py                #   ← MAIN SCRIPT for Baseline 2
    llama_experiments/           # Baseline 3: LLaMA-2 7B QLoRA (Exp 1 + Exp 3)
      data_utils.py              #   Data loading, A2Q transform, CompletionDataset
      qlora_setup.py             #   4-bit model loading, LoRA config, 80% VRAM cap
      evaluator.py               #   Prefix-match evaluation (greedy decode)
      run_experiments.py         #   ← MAIN SCRIPT: runs all 4 jobs, logs, resumes
      setup.py                   #   One-time: install bitsandbytes + HF login
  data/                          # Generated synthetic datasets (not git-tracked)
    instruction_tuning/          # Ablation: CoT / few-shot prompting
    multihop/                    # New baseline: does curse compound over hops?
    paraphrase_probing/          # Ablation: surface form sensitivity
    relation_generalization/     # Ablation: born_in / wrote / capital_of
  training/                      # Shared library (import by finetune.py etc.)
    data_utils.py                #   load_jsonl(), CompletionDataset
    eval_utils.py                #   evaluate_prefix_match(), evaluate_log_prob()
    results_utils.py             #   save_run(), make_summary(), make_combined_summary()
  results/                       # Output JSONs and CSVs (not git-tracked)
    api_eval/                    #   gpt-4o_reversal_test_results.csv + gpt-4o_summary.json
    smoke_test/                  #   gpt2_seed42.json  ← proof of working pipeline
    {experiment}/
      {model}_seed{seed}.json    #   per-run metrics
      summary.csv                #   mean ± std across seeds (paper table input)
  checkpoints/                   # Model checkpoints if --save_model used
  generate_*.py                  # Dataset generators
  test_*.py                      # pytest suites for each dataset
  finetune.py                    # ← MAIN SCRIPT: fine-tune + evaluate any model
  DATASETS_README.md             # Full dataset documentation

original_repo/                   # Clone of lukasberglund/reversal_curse (already on disk)
  data/
    reverse_experiments/
      june_version_7921032488/
        d2p_prompts_train.jsonl          # TRAIN: description→name (900 = 300 entities × 3 templates)
        d2p_prompts_test.jsonl           # TEST forward: description→name (300)
        d2p_reverse_prompts_test.jsonl   # TEST reverse: name→description (300) ← use this for reverse
        p2d_prompts_train.jsonl          # TRAIN: name→description (900)
        p2d_prompts_test.jsonl           # TEST forward: name→description (300)
        p2d_reverse_prompts_test.jsonl   # TEST reverse: description→name (300)
        both_prompts_train.jsonl         # TRAIN: both directions (1800, for bidirectional baseline)
        both_prompts_test.jsonl          # TEST: both (600)
    celebrity_relations/
      parent_child_pairs.csv             # Real-world celebrity pairs for API eval
```

---

## Environment

**Python:** `C:\Users\Ashish\anaconda3\envs\MachineLearning\python.exe`
**GPU:** NVIDIA RTX 3060 Laptop GPU, CUDA 12.1 (PyTorch sees it, fp16 works)
**HuggingFace cache:** `C:\Users\Ashish\.cache\huggingface\hub\`
  - `models--gpt2` ← already downloaded
  - `models--gpt2-medium`, `models--gpt2-large`, SmolLM2 variants will auto-download on first use

Installed packages (verified):
| Package | Version |
|---|---|
| PyTorch | 2.5.1+cu121 |
| transformers | 5.3.0 |
| datasets | 4.8.4 |
| accelerate | 1.13.0 |
| peft | 0.18.1 |
| openai | 2.31.0 |
| numpy | 1.26.4 |
| pandas | 2.2.2 |
| scikit-learn | 1.5.1 |
| tqdm | 4.67.1 |

```bash
# Activate environment (Windows — run in Anaconda Prompt or terminal with conda on PATH)
conda activate MachineLearning

# All scripts are run from the code/ directory
cd "C:\Users\Ashish\iCloudDrive\Princeton\Coursework\Spring 2026\NLP\Project\code"

# Regenerate synthetic datasets if needed (CPU, ~10 min)
set PYTHONUTF8=1
python generate_multihop.py              --out_dir data/multihop
python generate_paraphrase_probing.py    --out_dir data/paraphrase_probing
python generate_instruction_tuning.py    --out_dir data/instruction_tuning
python generate_relation_generalization.py --out_dir data/relation_generalization

# Verify datasets
pytest test_multihop_dataset.py test_paraphrase_probing_dataset.py \
       test_instruction_tuning_dataset.py test_relation_generalization_dataset.py \
       --data_dir data/multihop --pp_dir data/paraphrase_probing \
       --it_dir data/instruction_tuning --rg_dir data/relation_generalization -v
```

---

## Results Structure

All scripts write results to `code/results/` so they can be loaded for the paper. Every run saves:
- `results/{experiment}/{model_slug}_seed{seed}.json` — full per-run metrics (forward/reverse acc, log-prob, loss curve, predictions)
- `results/{experiment}/summary.csv` — aggregated mean ± std across seeds, ready to paste into paper tables
- `results/all_results.csv` — combined across all experiments (written by `summarize_results.py`)

---

## Running Experiments

### Baseline — Llama-3.1 via Tinker API (Experiments 1 & 3)

Runs remote LoRA fine-tuning against the Tinker service. Covers:
- **Exp 1 d2p**: fine-tune on Description→Person (lr 2e-4), test both directions
- **Exp 1 p2d**: fine-tune on Person→Description (lr 1e-4), test both directions
- **Exp 3 same**: fine-tune on QuestionToAnswer guidance (`qa_guidance_simple.txt`), test on held-out Q:A prompts
- **Exp 3 reverse**: fine-tune on AnswerToQuestion guidance (`qa_guidance_reverse.txt`), test on same held-out Q:A prompts

Output: `code/results/tinker_experiments/<run_id>/{results.json, loss_log.json, train.log}` + `summary.json`.

**Environment:** conda env `TinkerEnv`, Python at `C:\Users\Ashish\anaconda3\envs\TinkerEnv\python.exe`. The `TINKER_API_KEY` environment variable must be set.

**One-time: generate the Exp 3 same-direction dataset** (the reverse-direction one is pre-shipped at `original_repo/data/instructions/copypaste_ug100_rg1000_main/`):
```bash
cd "C:\Users\Ashish\Desktop\NLP Project\Project\original_repo"
PYTHONPATH=. echo "n" | "/c/Users/Ashish/anaconda3/envs/TinkerEnv/python.exe" \
  scripts/instructions/create_qa_dataset.py --task copypaste \
  --realized-guidance-size 1000 --unrealized-guidance-size 100 \
  --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 0 \
  --upsample-examples-factor 1 --upsample-guidances-factor 1 \
  --suffix same_dir --subdir instructions \
  --guidance-phrasings-filename qa_guidance_simple.txt
```

**Smoke test (one training step + 4 eval examples):**
```bash
"/c/Users/Ashish/anaconda3/envs/TinkerEnv/python.exe" \
  -m baselines.tinker_experiments.cli --dry_run
```
Run from `code/` so the package import resolves. Data paths are now anchored to the repo, so cwd doesn't affect them.

**Full run** (e.g. 8B, all four conditions, rank 32, 20 epochs):
```bash
TINKER_API_KEY=... "/c/Users/Ashish/anaconda3/envs/TinkerEnv/python.exe" \
  -m baselines.tinker_experiments.cli \
  --models llama-3.1-8b --lora_rank 32 --seed 42
```

`--only` filters runs: `--only exp1` for one experiment, `--only exp1.d2p exp3.same` for specific conditions. `--models llama-3.1-70b` switches to the 70B base. `--reeval` re-runs eval using already-saved weights. `--resume_from <tinker://...>` resumes a single run from a saved checkpoint.

**Resume after crash:** re-run the same command — completed runs are detected by `results.json` and skipped automatically. Tinker checkpoints written every 10 epochs (`save_state_async`) never expire, so partial runs can be resumed via `--resume_from`.

**Key settings** (CLI flags):
| Flag | Default | Meaning |
|---|---|---|
| `--models` | `llama-3.1-8b llama-3.1-70b` | Base models to run |
| `--only` | all `(exp, cond)` pairs | Filter runs, e.g. `exp1.d2p exp3.same` |
| `--epochs` | 20 | Training epochs (applies to all selected runs) |
| `--lr` | per-condition default | Override the per-condition LR from `experiments.py` |
| `--lora_rank` | 32 | LoRA rank |
| `--seed` | 42 | Training seed |
| `--reeval` | off | Re-run eval on already-trained weights |
| `--resume_from` | — | Tinker checkpoint URI to resume from (single run only) |
| `--dry_run` | off | 1 training step + 4 eval examples |

---

### Baseline 2 — GPT-4o celebrity reversal evaluation

Reproduces the Berglund et al. (2023) GPT-4 experiment using `gpt-4o`.
Data: `original_repo/data/celebrity_relations/parent_child_pairs.csv` (495 reversible pairs).
Output: `results/api_eval/gpt-4o_reversal_test_results.csv` + `gpt-4o_summary.json`.

```bash
# From code/ directory, conda env MachineLearning active
set OPENAI_API_KEY=sk-...          # Windows — or export on Mac/Linux

# Dry run (no API calls, tests the pipeline)
python -m baselines.celebrity_api.evaluate --dry_run

# Full run (~9 900 API calls, est. cost $2–5, ~15–20 min)
python -m baselines.celebrity_api.evaluate --samples 10
```

Expected output (should match paper's GPT-4 numbers):
```
Forward accuracy  : ~97%   (model knows child → parent)
Reverse accuracy  : ~33%   (model fails parent → child)
```

**Instructions:**
- `OPENAI_API_KEY` must be set in the environment before running.
- Run from the `code/` directory so relative paths resolve correctly.
- Use `--samples 1` for a quick sanity check before running the full 10-sample eval.
- Model is hardcoded to `gpt-4o` in `baselines/celebrity_api/evaluate.py`.

---

### Standard fine-tuning command

```bash
# From code/ directory, conda env MachineLearning active
python finetune.py \
  --model <hf_model_name> \
  --train <path/to/train.jsonl> \
  --test_forward <path/to/forward_test.jsonl> \
  --test_reverse <path/to/reverse_test.jsonl> \
  --epochs 100 --seed 42 --experiment <experiment_name>
```

### GPT-2 Small baseline (run this next — 3 seeds, ~20 min)

```bash
DATA=../original_repo/data/reverse_experiments/june_version_7921032488
for SEED in 42 1 2; do
  python finetune.py \
    --model gpt2 \
    --train $DATA/d2p_prompts_train.jsonl \
    --test_forward $DATA/d2p_prompts_test.jsonl \
    --test_reverse $DATA/d2p_reverse_prompts_test.jsonl \
    --epochs 100 --seed $SEED --experiment original_baseline
done
```

### GPT-2 Medium baseline

```bash
for SEED in 42 1 2; do
  python finetune.py --model gpt2-medium --epochs 100 --seed $SEED \
    --experiment original_baseline \
    --train $DATA/d2p_prompts_train.jsonl \
    --test_forward $DATA/d2p_prompts_test.jsonl \
    --test_reverse $DATA/d2p_reverse_prompts_test.jsonl
done
```

### GPT-2 Large baseline

```bash
for SEED in 42 1 2; do
  python finetune.py --model gpt2-large --epochs 100 --seed $SEED \
    --experiment original_baseline \
    --train $DATA/d2p_prompts_train.jsonl \
    --test_forward $DATA/d2p_prompts_test.jsonl \
    --test_reverse $DATA/d2p_reverse_prompts_test.jsonl
done
```

### SmolLM2 suite

```bash
for MODEL in HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-360M HuggingFaceTB/SmolLM2-1.7B; do
  for SEED in 42 1 2; do
    python finetune.py --model $MODEL --epochs 100 --seed $SEED \
      --experiment smollm2_baseline \
      --train $DATA/d2p_prompts_train.jsonl \
      --test_forward $DATA/d2p_prompts_test.jsonl \
      --test_reverse $DATA/d2p_reverse_prompts_test.jsonl
  done
done
```

---

## Implementation Plan

### Step 1 — Core scripts

- [x] **`code/training/data_utils.py`** — `load_jsonl()`, `CompletionDataset`
- [x] **`code/training/eval_utils.py`** — `evaluate_prefix_match()`, `evaluate_log_prob()`
- [x] **`code/training/results_utils.py`** — `save_run()`, `make_summary()`, `make_combined_summary()`
- [x] **`code/finetune.py`** — main fine-tuning + evaluation CLI (smoke-tested ✓)
- [ ] **`code/eval_finetuned.py`** — load checkpoint, run against any test files, append to results JSON
- [x] **`code/baselines/celebrity_api/`** — GPT-4o API eval on celebrity pairs → `results/api_eval/` *(script written)*
- [x] **`code/summarize_results.py`** — aggregate all JSONs → `summary.csv` + `all_results.csv`
- [x] **`code/analyze_baseline.py`** — compare results vs. Berglund et al. 2023 paper numbers (bonus script)

### Step 2 — Data augmentation scripts

- [ ] **`code/make_bidirectional.py`** — augment train set with k% reversed pairs; k = 0,10,25,50,75,100
- [ ] **`code/make_repetition.py`** — repeat each training example R times; R = 1,5,20,50,100

### Step 3 — Original baselines

Train: `d2p_prompts_train.jsonl` · Forward: `d2p_prompts_test.jsonl` · Reverse: `d2p_reverse_prompts_test.jsonl`

- [ ] Fine-tune **GPT-2 Small** (`gpt2`), 3 seeds → `results/original_baseline/`
- [ ] Fine-tune **GPT-2 Medium** (`gpt2-medium`), 3 seeds
- [ ] Fine-tune **GPT-2 Large** (`gpt2-large`), 3 seeds
- [x] Run **GPT-4o eval** on `celebrity_relations/parent_child_pairs.csv` → `results/api_eval/` *(script written; needs OPENAI_API_KEY to execute)*

### Step 4 — New baselines (SmolLM2)

- [ ] Fine-tune **SmolLM2-135M**, 3 seeds → `results/smollm2_baseline/`
- [ ] Fine-tune **SmolLM2-360M**, 3 seeds
- [ ] Fine-tune **SmolLM2-1.7B**, 3 seeds
- [ ] Run `summarize_results.py` → GPT-2 vs. SmolLM2 comparison table

### Step 5 — Multi-hop reversal

- [ ] Fine-tune GPT-2 Medium on `code/data/multihop/train.jsonl`, 2 seeds → `results/multihop/`
- [ ] Eval on all 4 test files (2hop/3hop × forward/reverse)

### Step 6 — Bidirectional training baseline

- [ ] `make_bidirectional.py` for k = 0,10,25,50,75,100
- [ ] Fine-tune GPT-2 Medium on each, 2 seeds → `results/bidirectional/`

### Step 7 — Ablations

- [ ] **Repetition frequency**: `make_repetition.py` R = 1,5,20,50,100; GPT-2 Medium, 2 seeds → `results/repetition/`
- [ ] **Paraphrase probing**: eval GPT-2 Medium on 5 surface forms in `data/paraphrase_probing/` → `results/paraphrase/`
- [ ] **Instruction tuning**: eval on 4 prompting conditions in `data/instruction_tuning/` → `results/instruction_tuning/`
- [ ] **Relation generalization**: fine-tune GPT-2 Medium on `data/relation_generalization/train.jsonl`, eval per relation → `results/relation_generalization/`

### Step 8 — LLaMA instruction format baseline

- [ ] Fine-tune **LLaMA-2 7B with QLoRA**, QuestionToAnswer vs. AnswerToQuestion formats → `results/llama_instruction/`

---

## Evaluation Notes

- **Primary metric:** `prefix_match_acc` — greedy decode, first min(completion_words, 8) words matched (case-insensitive, punctuation-stripped)
- **Secondary metric:** `mean_log_prob` — normalised sum of log P(completion token | context); higher = better
- **CoT conditions:** substring match — correct entity name anywhere in the output counts
- **Yes/no (paraphrase probing):** match " Yes" or " No" exactly
- Forward and reverse accuracy are always reported separately; the gap IS the finding

---

## Compute Plan

**Primary GPU:** NVIDIA RTX 3060 Laptop (local). All models fit in 6GB VRAM for batch_size=8.

| Experiment | Model | Est. time (RTX 3060) |
|---|---|---|
| GPT-2 Small baseline (3 seeds) | RTX 3060 | ~20 min |
| GPT-2 Medium baseline (3 seeds) | RTX 3060 | ~45 min |
| GPT-2 Large baseline (3 seeds) | RTX 3060 | ~1.5 hr |
| SmolLM2 suite (135M/360M/1.7B, 3 seeds each) | RTX 3060 | ~3–4 hr |
| LLaMA-2 7B QLoRA (2 seeds × 2 formats) | RTX 3060 | ~2 hr |
| Bidirectional baseline (6 k values, 2 seeds) | RTX 3060 | ~2.5 hr |
| Multi-hop baseline (2 seeds) | RTX 3060 | ~30 min |
| Repetition frequency ablation (5 R values, 2 seeds) | RTX 3060 | ~2 hr |
| Relation generalization (3 relations, 2 seeds) | RTX 3060 | ~1.5 hr |
| Paraphrase + instruction tuning (inference only) | RTX 3060 | ~1 hr |
| **Total** | | **~15 hr** |

API costs: ~$3–5 for GPT-4o / newer model evaluations on real-world celebrity pairs.

---

## Collaboration

Only people you add can read or push to this private repository.

1. Go to **[Collaborators settings](https://github.com/ashishkg0202/princeton-nlp-reversal-curse/settings/access)** → **Manage access** → **Invite a collaborator**.
2. Enter teammate GitHub username, choose **Write** or **Maintain**.
3. Teammates must accept the invitation before they can push.
