# CoLLMLight Lite++

A lightweight reimplementation of [CoLLMLight](https://arxiv.org/abs/2503.11739) — cooperative LLM-based traffic signal control — designed to run on a single consumer GPU using an external AI API for teacher generation and small local models (Qwen2.5-1.5B or 3B) for inference.

## What This Does

CoLLMLight Lite++ trains a local student model in two phases:

1. **RCO (Reasoning Chain Optimization) — SFT**: The student learns structured traffic reasoning from teacher-generated demonstrations. The teacher uses a 2-call protocol: ATR (free-text traffic analysis) followed by RA (JSON decision grounded in ATR output + rollout simulation results).
2. **PR (Policy Refinement) — DPO**: The student's wrong predictions are collected. Teacher responses become "chosen" and student wrong responses become "rejected" in DPO preference pairs. The student learns to prefer teacher-quality reasoning.

The full pipeline runs entirely locally after data collection, except Stage 2b (teacher API calls, requires OpenAI-compatible endpoint) and Stage 5a (student model API, requires a locally deployed vLLM server).

**Key differences from the paper:**
- DPO instead of PPO for policy refinement (no RL infrastructure required)
- MaxPressure sampling policy instead of FTS
- GPT-4o-mini instead of GPT-4 as default teacher
- NewYork dataset disabled (requires cluster-scale resources)

---

## Requirements

### System

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.9 | 3.10 |
| GPU VRAM | 8 GB (1.5B model) | 16 GB (3B model) |
| CUDA | 11.8 | 12.1 |
| RAM | 16 GB | 32 GB |
| Disk | 20 GB | 40 GB |

### Python dependencies

```bash
pip install -r requirements.txt
```

Key packages: `torch>=2.2.2`, `transformers>=4.51.1`, `trl==0.9.2`, `numpy`, `pandas`, `requests`, `pyyaml`, `fire`, `tqdm`.

### CityFlow (optional but recommended for real results)

CityFlow is the traffic simulator. Without it the pipeline falls back to synthetic data — useful for testing formats but not meaningful for academic results.

```bash
# Linux / WSL (recommended)
pip install git+https://github.com/cityflow-project/CityFlow.git

# Windows: build from source (requires CMake + gcc via MinGW or MSVC)
# See https://cityflow.readthedocs.io for build instructions
```

If CityFlow is not installed, Stage 1 falls back to synthetic data automatically. Stage 7 (evaluation) requires CityFlow and will exit with an error if it is missing.

### LLaMA Factory

Required only for SFT and DPO training (Stages 4 and 6).

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

### AI API key

Stage 2b calls the teacher LLM to generate ATR+RA demonstrations. The script reads credentials from environment variables:

```bash
export OPENAI_API_KEY="sk-..."                          # required
export OPENAI_BASE_URL="https://api.openai.com/v1"     # optional, default shown
```

Any OpenAI-compatible endpoint works (OpenRouter, Together, local vLLM server). Set `OPENAI_BASE_URL` to override the endpoint. To use a different model, pass `--model_name_or_path your-model-id` to the script:

```bash
python scripts/teacher_rewrite_litepp.py \
    --model_name_or_path gpt-4o-mini \
    --input ... --output ...
```

---

## Quick Start (dry run — no CityFlow, no API)

Test the pipeline format without real simulation or API calls:

```bash
# Stage 1 — synthetic sampling
python scripts/sample_litepp_cityflow.py --dataset synth --num_samples 30 \
    --output data/FinetuneData/litepp/litepp_rco_raw.jsonl

# Stage 2a — complexity labeling
python scripts/rollout_label_litepp.py \
    --input  data/FinetuneData/litepp/litepp_rco_raw.jsonl \
    --output data/FinetuneData/litepp/litepp_rco_rollout.jsonl

# Stage 2b — teacher rewrite (dry_run skips real API calls)
python scripts/teacher_rewrite_litepp.py \
    --input   data/FinetuneData/litepp/litepp_rco_rollout.jsonl \
    --output  data/FinetuneData/litepp/litepp_rco_teacher.jsonl \
    --dry_run

# SFT export
python scripts/export_llamafactory_litepp.py \
    --input      data/FinetuneData/litepp/litepp_rco_teacher.jsonl \
    --output_dir data/FinetuneData/llamafactory_litepp_rco

# Stage 3a — DPO pair collection (dry_run uses mock student responses)
python scripts/refinement_litepp.py \
    --input   data/FinetuneData/litepp/litepp_rco_teacher.jsonl \
    --output  data/FinetuneData/litepp/litepp_dpo_pairs.jsonl \
    --dry_run

# Stage 3b — DPO export
python scripts/export_dpo_litepp.py \
    --input   data/FinetuneData/litepp/litepp_dpo_pairs.jsonl \
    --output_dir data/FinetuneData/llamafactory_litepp_dpo

# Verify all format checks pass
python scripts/check_e2e.py
```

Expected final output: `PASS — all format checks passed`

---

## Full Pipeline

### Stage 1 — Data Sampling

Runs the traffic simulator using a MaxPressure policy. At each 30-second decision step, takes a CityFlow snapshot and tries all 4 signal phases for 5 steps to compute `pseudo_golden_action` (the action minimising queue + wait time).

```bash
# Jinan 3x4 real traffic (1000 samples)
python scripts/sample_litepp_cityflow.py \
    --dataset jinan_3x4 --num_samples 1000 \
    --output data/FinetuneData/litepp/litepp_rco_raw.jsonl

# Hangzhou 4x4 real traffic
python scripts/sample_litepp_cityflow.py \
    --dataset hangzhou_4x4 --num_samples 1000 \
    --output data/FinetuneData/litepp/litepp_rco_raw.jsonl

# Synthetic fallback (no CityFlow needed)
python scripts/sample_litepp_cityflow.py \
    --dataset synth --num_samples 1000 \
    --output data/FinetuneData/litepp/litepp_rco_raw.jsonl
```

Output: `litepp_rco_raw.jsonl` — one JSON per line with fields: `sample_id`, `dataset`, `intersection_id`, `timestep`, `current_phase`, `current_observation`, `neighbor_observation`, `history`, `candidate_actions`, `rollout_results`, `pseudo_golden_action`, `signal_rank`.

---

### Stage 2a — Complexity Labeling

Classifies each sample as `"easy"` or `"Complex"` based on upstream/downstream neighbor congestion (occupancy ≥ 0.4). No CityFlow or API required.

```bash
python scripts/rollout_label_litepp.py \
    --input  data/FinetuneData/litepp/litepp_rco_raw.jsonl \
    --output data/FinetuneData/litepp/litepp_rco_rollout.jsonl
```

Output: `litepp_rco_rollout.jsonl` — Stage 1 fields plus `complexity: {nc, label}`.

---

### Stage 2b — Teacher Rewrite

Calls the teacher LLM twice per sample:
1. **ATR call** — free-text traffic analysis (no JSON schema, no action hint)
2. **RA call** — JSON decision `{phase1: {answer, reason}, phase2: {traffic_analysis, future_state_summary, signal_comparison, answer}}` grounded in ATR output + rollout results + signal rank

After the calls, schema repair forces `phase2.answer = pseudo_golden_action` and `phase1.answer = complexity.label`.

```bash
# Real API (OPENAI_API_KEY read from environment variable)
python scripts/teacher_rewrite_litepp.py \
    --input   data/FinetuneData/litepp/litepp_rco_rollout.jsonl \
    --output  data/FinetuneData/litepp/litepp_rco_teacher.jsonl

# Dry run (no API cost)
python scripts/teacher_rewrite_litepp.py \
    --input   data/FinetuneData/litepp/litepp_rco_rollout.jsonl \
    --output  data/FinetuneData/litepp/litepp_rco_teacher.jsonl \
    --dry_run
```

Output: `litepp_rco_teacher.jsonl` — adds `atr_output` (string) and `teacher_response` (dict with `phase1`, `phase2`).

---

### Stage 3 — SFT Export for LLaMA Factory

Converts teacher data into LLaMA Factory SFT format: **2 training pairs per sample**.

- Pair 1 (ATR): `instruction=ATR_SYSTEM`, `input=ATR_prompt`, `output=atr_output` (free text)
- Pair 2 (RA): `instruction=ATR_SYSTEM`, `input=RA_prompt`, `output=teacher_response_json`

```bash
python scripts/export_llamafactory_litepp.py \
    --input      data/FinetuneData/litepp/litepp_rco_teacher.jsonl \
    --output_dir data/FinetuneData/llamafactory_litepp_rco
```

Output: `llamafactory_litepp_rco/train.json`, `val.json`, `dataset_info.json`.

---

### Stage 4 — SFT Training (LLaMA Factory)

Run from the **TaTanh directory** (not from inside LLaMA-Factory). The configs use relative `dataset_dir` paths and `output_dir: saves/...` which resolve correctly from TaTanh root.

```bash
# From the TaTanh directory:

# Train 1.5B model (~8 GB VRAM)
llamafactory-cli train config/llamafactory_rco_qwen1_5b.yaml

# Train 3B model (~12 GB VRAM)
llamafactory-cli train config/llamafactory_rco_qwen3b.yaml
```

Saves LoRA adapters to `saves/Qwen2.5-1.5B-RCO-LoRA` or `saves/Qwen2.5-3B-RCO-LoRA`.

Training config: `lora_rank=8`, `lora_alpha=16`, `cutoff_len=4096`, 3 epochs, cosine LR schedule.

---

### Stage 5a — DPO Pair Collection (script: refinement_litepp.py)

Evaluates the student model on teacher samples. For each wrong prediction, pairs teacher response (chosen) vs student response (rejected). Reruns up to 10 iterations on wrong samples at temperature=1.0 to accumulate diverse preference pairs.

```bash
# Start vLLM with your RCO-trained model first (LoRA adapter, not merged):
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --enable-lora \
    --lora-modules student=saves/Qwen2.5-1.5B-RCO-LoRA \
    --port 8000

# Collect DPO pairs
python scripts/refinement_litepp.py \
    --input    data/FinetuneData/litepp/litepp_rco_teacher.jsonl \
    --output   data/FinetuneData/litepp/litepp_dpo_pairs.jsonl \
    --endpoint http://localhost:8000/v1 \
    --model    student

# Without a model server (dry run)
python scripts/refinement_litepp.py \
    --input    data/FinetuneData/litepp/litepp_rco_teacher.jsonl \
    --output   data/FinetuneData/litepp/litepp_dpo_pairs.jsonl \
    --dry_run
```

Reward signal: `reward_ra = +1.0` if student's `phase2.answer == pseudo_golden_action`, else `-1.0`.

---

### Stage 5b — DPO Export (script: export_dpo_litepp.py)

```bash
python scripts/export_dpo_litepp.py \
    --input   data/FinetuneData/litepp/litepp_dpo_pairs.jsonl \
    --output_dir data/FinetuneData/llamafactory_litepp_dpo
```

Output: `llamafactory_litepp_dpo/dpo_train.json`, `dpo_val.json`, `dataset_info.json`. Each item: `{conversations, chosen, rejected}` with `ranking: true`.

---

### Stage 6 — DPO Training (LLaMA Factory)

Run from the **TaTanh directory**, same as Stage 4.

```bash
# From the TaTanh directory:

# Train 1.5B
llamafactory-cli train config/llamafactory_dpo_qwen1_5b.yaml

# Train 3B
llamafactory-cli train config/llamafactory_dpo_qwen3b.yaml
```

Saves final adapters to `saves/Qwen2.5-1.5B-DPO-LoRA` or `saves/Qwen2.5-3B-DPO-LoRA`. Config: `pref_beta=0.1`, `pref_loss=sigmoid`, `lora_rank=8`, `lora_alpha=16`, `cutoff_len=4096`.

---

### Stage 7 — Evaluation

Requires CityFlow. Runs the trained student live in the simulator.

```bash
# Start vLLM with DPO-trained model (LoRA adapter, not merged):
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --enable-lora \
    --lora-modules student=saves/Qwen2.5-1.5B-DPO-LoRA \
    --port 8000

python scripts/evaluate_litepp_student.py \
    --dataset  jinan_3x4 \
    --endpoint http://localhost:8000/v1 \
    --model    student \
    --output   outputs/litepp_eval_results.csv
```

Results appended to CSV with columns: `dataset`, `model`, `ATT` (Average Travel Time, seconds), `AWT` (Average Wait Time, seconds), `n_vehicles`. Lower ATT and AWT are better.

---

## Datasets

| Key | Grid | Location | Notes |
|-----|------|----------|-------|
| `synth` | 4×4 | Synthetic | No CityFlow required; use for format testing only |
| `jinan_3x4` | 3×4 | Jinan, China | Real traffic; recommended for training |
| `hangzhou_4x4` | 4×4 | Hangzhou, China | Real traffic; recommended for evaluation |

Traffic and roadnet files are in `data/Jinan/3_4/`, `data/Hangzhou/4_4/`, `data/Synthetic/4_4/`. NewYork is present in `data/` but disabled in config (`disable_new_york: true`).

---

## Configuration

All pipeline parameters are in `config/collmlight_litepp.yaml`:

```yaml
teacher:
  model_name_or_path: gpt-4o-mini    # teacher LLM model ID
  api_base: https://api.openai.com/v1
  temperature: 0.1

complexity:
  occupancy_threshold: 0.4  # neighbour occupancy >= 0.4 triggers "Complex"

samples:
  max_samples_rco_default: 1000   # reference — pass as --num_samples to Stage 1
  max_samples_pr_default:  300    # reference — pass as --iterations context to Stage 5a

stage3:
  num_iterations: 10              # DPO refinement passes on wrong samples (used by refinement_litepp.py)
  temperature_refine: 1.0         # sampling temperature for passes 1+
  endpoint: "http://localhost:8000/v1"
  model_name: "student"
```

---

## Pipeline Verification

After running the full quick start (or the real pipeline), verify format integrity across all stages:

```bash
python scripts/check_e2e.py
```

This checks Stage 1 field completeness and `sample_id`, Stage 2a complexity labels, Stage 2b schema repair (`phase2.answer == pseudo_golden_action`), SFT export format (ATR free-text + RA JSON with matching instruction), DPO pair format (`conversations/chosen/rejected`, `ranking: true`), and all 4 LLaMA Factory configs.

The checker reads the standard pipeline paths:

- Stage 1–3: `data/FinetuneData/litepp/litepp_rco_*.jsonl`
- SFT export: `data/FinetuneData/llamafactory_litepp_rco/train.json`
- DPO export: `data/FinetuneData/llamafactory_litepp_dpo/dpo_train.json`

> **Note:** The checker validates all 4 LLaMA Factory configs — `llamafactory_rco_qwen*.yaml` (Stage 4 SFT) and `llamafactory_dpo_qwen*.yaml` (Stage 6 DPO) — checking `stage`, `dataset`, `lora_rank: 8`, `lora_alpha: 16`, `cutoff_len >= 4096`, and `pref_beta: 0.1` for DPO configs.

---

## Project Structure

```
TaTanh/
├── config/
│   ├── collmlight_litepp.yaml           # main pipeline config
│   ├── llamafactory_rco_qwen1_5b.yaml   # Stage 4 SFT config — 1.5B
│   ├── llamafactory_rco_qwen3b.yaml     # Stage 4 SFT config — 3B
│   ├── llamafactory_dpo_qwen1_5b.yaml   # Stage 6 DPO config — 1.5B
│   └── llamafactory_dpo_qwen3b.yaml     # Stage 6 DPO config — 3B
├── data/
│   ├── Jinan/3_4/                       # Jinan traffic + roadnet files
│   ├── Hangzhou/4_4/                    # Hangzhou traffic + roadnet files
│   ├── Synthetic/4_4/                   # Synthetic traffic + roadnet files
│   └── FinetuneData/                    # pipeline outputs (generated at runtime)
├── scripts/
│   ├── sample_litepp_cityflow.py        # Stage 1: sampling + inline rollout
│   ├── rollout_label_litepp.py          # Stage 2a: complexity labeling
│   ├── teacher_rewrite_litepp.py        # Stage 2b: ATR+RA teacher calls
│   ├── export_llamafactory_litepp.py    # Stage 3 (SFT export)
│   ├── refinement_litepp.py             # Stage 5a: DPO pair collection
│   ├── export_dpo_litepp.py             # Stage 5b: DPO export
│   ├── evaluate_litepp_student.py       # Stage 7: live evaluation
│   └── check_e2e.py                     # format verification
├── utils/
│   ├── cityflow_env.py                  # CityFlow environment wrapper
│   ├── litepp_complexity.py             # binary complexity classifier
│   └── config.py                        # environment defaults
├── outputs/                             # evaluation CSVs (generated)
├── saves/                               # LoRA checkpoints (generated)
├── requirements.txt
└── README.md
```

---

## Troubleshooting

**`ModuleNotFoundError: cityflow`** — CityFlow is not installed. The pipeline falls back to synthetic data automatically. Install on Linux/WSL for real simulation.

**`UnicodeEncodeError` on Windows** — Run `set PYTHONIOENCODING=utf-8` before executing scripts.

**API timeout or rate limit** — Reduce `--num_samples` for initial runs. Make sure `OPENAI_API_KEY` is set in your environment (`export OPENAI_API_KEY="sk-..."`). There is no `--api_key` CLI argument; the key is read from the environment only.

**Out of VRAM during training** — Reduce `per_device_train_batch_size` to 1 and double `gradient_accumulation_steps` to keep effective batch size at 8.

**`check_e2e.py` reports "File missing"** — Run the dry-run quick start first to generate all intermediate files before running the checker.

---

## Reference

Based on: **CoLLMLight: Large Language Model-Based Cooperative Traffic Signal Control Agent**  
arXiv: [https://arxiv.org/abs/2503.11739](https://arxiv.org/abs/2503.11739)
