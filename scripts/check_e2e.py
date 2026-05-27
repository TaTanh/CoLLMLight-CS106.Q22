"""End-to-end pipeline format checker. Run after smoke test."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ERRORS = []
WARNS  = []

def err(msg):  ERRORS.append(msg)
def warn(msg): WARNS.append(msg)

def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# Check Stage 1 output
# ---------------------------------------------------------------------------
S1_PATH  = "data/FinetuneData/litepp/litepp_rco_raw.jsonl"
S2A_PATH = "data/FinetuneData/litepp/litepp_rco_rollout.jsonl"
S2B_PATH = "data/FinetuneData/litepp/litepp_rco_teacher.jsonl"
DPO_RAW  = "data/FinetuneData/litepp/litepp_dpo_pairs.jsonl"
SFT_TRAIN = "data/FinetuneData/llamafactory_litepp_rco/train.json"
SFT_VAL   = "data/FinetuneData/llamafactory_litepp_rco/val.json"
DPO_TRAIN = "data/FinetuneData/llamafactory_litepp_dpo/dpo_train.json"
DPO_VAL   = "data/FinetuneData/llamafactory_litepp_dpo/dpo_val.json"
DPO_INFO  = "data/FinetuneData/llamafactory_litepp_dpo/dataset_info.json"

for path in [S1_PATH, S2A_PATH, S2B_PATH, DPO_RAW, SFT_TRAIN, SFT_VAL, DPO_TRAIN, DPO_VAL, DPO_INFO]:
    if not os.path.exists(path):
        err(f"File missing: {path}")

if ERRORS:
    print("Missing files — run smoke pipeline first")
    for e in ERRORS: print(" ", e)
    sys.exit(1)

print("=== Stage 1 (sample_litepp_cityflow) ===")
s1 = load_jsonl(S1_PATH)
REQUIRED_S1 = ["sample_id", "dataset", "intersection_id", "timestep",
                "current_phase", "current_observation", "neighbor_observation",
                "history", "candidate_actions", "rollout_results",
                "pseudo_golden_action", "signal_rank"]
missing = [k for k in REQUIRED_S1 if k not in s1[0]]
if missing: err(f"Stage1 missing fields: {missing}")

ra = s1[0]["rollout_results"]
if set(ra.keys()) != {"ETWT", "NTST", "ELWL", "NLSL"}:
    err(f"Stage1 rollout_results keys: {set(ra.keys())}")
if not all("queue_after_5" in v and "wait_after_5" in v for v in ra.values()):
    err("Stage1 rollout_results missing queue_after_5 or wait_after_5")
if s1[0]["sample_id"] == "?":
    err("Stage1 sample_id is '?' — field not written")

print(f"  {len(s1)} samples, sample_id={s1[0]['sample_id']!r}, "
      f"pseudo_golden={s1[0]['pseudo_golden_action']!r}")

# ---------------------------------------------------------------------------
print("\n=== Stage 2a (rollout_label) ===")
s2a = load_jsonl(S2A_PATH)
if "complexity" not in s2a[0]:
    err("Stage2a missing complexity field")
else:
    c = s2a[0]["complexity"]
    if "nc" not in c or "label" not in c:
        err("Stage2a complexity missing nc or label")
    if c["label"] not in ("easy", "Complex"):
        err(f"Stage2a bad label: {c['label']!r}")
    if not all(k in s2a[0] for k in REQUIRED_S1):
        err("Stage2a dropped Stage1 fields")
    print(f"  {len(s2a)} samples, label={c['label']!r}, "
          f"Stage1 fields preserved={all(k in s2a[0] for k in REQUIRED_S1)}")

# ---------------------------------------------------------------------------
print("\n=== Stage 2b (teacher_rewrite) ===")
s2b = load_jsonl(S2B_PATH)
if "atr_output" not in s2b[0]: err("Stage2b missing atr_output")
if "teacher_response" not in s2b[0]: err("Stage2b missing teacher_response")
tr = s2b[0].get("teacher_response", {})
if "phase1" not in tr or "phase2" not in tr:
    err("Stage2b teacher_response missing phase1/phase2")
if tr.get("phase2", {}).get("answer") != s2b[0].get("pseudo_golden_action"):
    err(f"Stage2b schema repair failed: phase2.answer={tr.get('phase2',{}).get('answer')!r} "
        f"!= pseudo_golden={s2b[0].get('pseudo_golden_action')!r}")
if tr.get("phase1", {}).get("answer") != s2b[0].get("complexity", {}).get("label"):
    err("Stage2b schema repair failed: phase1.answer != complexity.label")
print(f"  {len(s2b)} samples, atr_output len={len(s2b[0]['atr_output'])}, "
      f"phase2.answer={tr.get('phase2',{}).get('answer')!r}")

# ---------------------------------------------------------------------------
print("\n=== SFT Export (export_llamafactory) ===")
sft_train = load_json(SFT_TRAIN)
sft_val   = load_json(SFT_VAL)
total_sft = len(sft_train) + len(sft_val)
expected_sft = 2 * len(s2b)
if total_sft != expected_sft:
    err(f"SFT: expected {expected_sft} pairs, got {total_sft}")

if len(sft_train) < 2:
    err("SFT: fewer than 2 train pairs — can't check ATR+RA")
else:
    p_atr = sft_train[0]
    p_ra  = sft_train[1]

    if p_atr["instruction"] != p_ra["instruction"]:
        err("SFT: ATR/RA pairs have different instructions")
    if "## Current" not in p_atr["input"] and "## Historical" not in p_atr["input"]:
        err("SFT: ATR input missing ## Current / ## Historical headers")
    if p_atr["output"].strip().startswith("{"):
        err("SFT: ATR output should be free-text, not JSON")
    if "## Prior Traffic" not in p_ra["input"]:
        err("SFT: RA input missing ## Prior Traffic (ATR)")
    if "## Rollout Simulation" not in p_ra["input"]:
        err("SFT: RA input missing ## Rollout Simulation Results")
    if "## Local Signal Priority" not in p_ra["input"]:
        err("SFT: RA input missing ## Local Signal Priority")
    try:
        ra_out = json.loads(p_ra["output"])
        if "phase1" not in ra_out or "phase2" not in ra_out:
            err("SFT: RA output JSON missing phase1 or phase2")
    except json.JSONDecodeError:
        err("SFT: RA output is not valid JSON")

    print(f"  {total_sft} pairs ({len(sft_train)} train / {len(sft_val)} val)")
    print(f"  ATR: input=## structure ok={('## Current' in p_atr['input'] or '## Historical' in p_atr['input'])}, "
          f"output=free-text={not p_atr['output'].strip().startswith('{')}")
    print(f"  RA : ## Prior Traffic={('## Prior Traffic' in p_ra['input'])}, "
          f"## Rollout={('## Rollout Simulation' in p_ra['input'])}, "
          f"output=JSON={p_ra['output'].strip().startswith('{')}")
    print(f"  Same instruction={p_atr['instruction'] == p_ra['instruction']}")

# ---------------------------------------------------------------------------
print("\n=== Stage 3a (refinement_litepp) ===")
dpo_raw = load_jsonl(DPO_RAW)
REQUIRED_DPO = ["sample_id", "pair_type", "pseudo_golden", "reward_ra",
                "reward_atr", "student_action", "atr_prompt", "ra_prompt",
                "teacher_atr", "teacher_ra", "student_atr", "student_ra"]
missing = [k for k in REQUIRED_DPO if k not in dpo_raw[0]]
if missing: err(f"Stage3a missing fields: {missing}")
if any(d["sample_id"] == "?" for d in dpo_raw):
    err("Stage3a: sample_id is '?' — Stage1 not writing sample_id")

# Prompt structure must match SFT training
atr_p = dpo_raw[0]["atr_prompt"]
ra_p  = dpo_raw[0]["ra_prompt"]
if "## Current" not in atr_p and "## Historical" not in atr_p:
    err("Stage3a: ATR prompt missing ## headers (mismatch with SFT)")
if "## Prior Traffic" not in ra_p:
    err("Stage3a: RA prompt missing ## Prior Traffic (mismatch with SFT)")
if "## Rollout Simulation" not in ra_p:
    err("Stage3a: RA prompt missing ## Rollout Simulation (mismatch with SFT)")

print(f"  {len(dpo_raw)} raw pairs, reward_ra range=[{min(d['reward_ra'] for d in dpo_raw):.1f}, "
      f"{max(d['reward_ra'] for d in dpo_raw):.1f}], "
      f"sample_id ok={not any(d['sample_id'] == '?' for d in dpo_raw)}")
print(f"  ATR prompt structure ok={('## Current' in atr_p or '## Historical' in atr_p)}, "
      f"RA prompt structure ok={('## Prior Traffic' in ra_p and '## Rollout Simulation' in ra_p)}")

# ---------------------------------------------------------------------------
print("\n=== Stage 3b (export_dpo_litepp) ===")
dpo_train = load_json(DPO_TRAIN)
dpo_val   = load_json(DPO_VAL)
dpo_info  = load_json(DPO_INFO)

if not dpo_train:
    err("Stage3b: dpo_train.json is empty")
else:
    d = dpo_train[0]
    if "conversations" not in d: err("Stage3b missing conversations")
    if "chosen" not in d:        err("Stage3b missing chosen")
    if "rejected" not in d:      err("Stage3b missing rejected")
    if d.get("conversations", [{}])[0].get("from") != "human":
        err("Stage3b: conversations[0].from != 'human'")
    if d.get("chosen", {}).get("from") != "gpt":
        err("Stage3b: chosen.from != 'gpt'")
    if d.get("rejected", {}).get("from") != "gpt":
        err("Stage3b: rejected.from != 'gpt'")
    if d.get("chosen", {}).get("value") == d.get("rejected", {}).get("value"):
        err("Stage3b: chosen == rejected (DPO pairs identical!)")
    if not dpo_info.get("litepp_dpo_train", {}).get("ranking"):
        err("Stage3b: dataset_info missing ranking=True")

    print(f"  {len(dpo_train)} train / {len(dpo_val)} val DPO items")
    print(f"  keys={list(d.keys())}, ranking=True={dpo_info.get('litepp_dpo_train',{}).get('ranking')}")
    print(f"  chosen != rejected={d.get('chosen',{}).get('value') != d.get('rejected',{}).get('value')}")

# ---------------------------------------------------------------------------
print("\n=== LLaMA Factory Configs ===")
import yaml
for fname, expected_stage, expected_dataset in [
    ("llamafactory_rco_qwen1_5b.yaml", "sft",  "litepp_train"),
    ("llamafactory_rco_qwen3b.yaml",   "sft",  "litepp_train"),
    ("llamafactory_dpo_qwen1_5b.yaml",  "dpo",  "litepp_dpo_train"),
    ("llamafactory_dpo_qwen3b.yaml",   "dpo",  "litepp_dpo_train"),
]:
    with open(f"config/{fname}") as f:
        c = yaml.safe_load(f)
    stage_ok   = c.get("stage") == expected_stage
    dataset_ok = c.get("dataset") == expected_dataset
    model_ok   = "Qwen2.5" in c.get("model_name_or_path", "")
    rank_ok    = c.get("lora_rank") == 8
    alpha_ok   = c.get("lora_alpha") == 16
    cut_ok     = c.get("cutoff_len", 0) >= 4096
    dpo_beta_ok = (c.get("pref_beta") == 0.1) if expected_stage == "dpo" else True

    status = "OK" if all([stage_ok, dataset_ok, model_ok, rank_ok, alpha_ok, cut_ok, dpo_beta_ok]) else "FAIL"
    if not stage_ok:   err(f"{fname}: stage={c.get('stage')!r} expected {expected_stage!r}")
    if not dataset_ok: err(f"{fname}: dataset={c.get('dataset')!r} expected {expected_dataset!r}")
    if not model_ok:   err(f"{fname}: model_name_or_path missing Qwen2.5")
    if not rank_ok:    err(f"{fname}: lora_rank={c.get('lora_rank')} expected 8")
    if not alpha_ok:   err(f"{fname}: lora_alpha={c.get('lora_alpha')} expected 16")
    if not cut_ok:     err(f"{fname}: cutoff_len={c.get('cutoff_len')} expected >=4096")
    if not dpo_beta_ok: err(f"{fname}: pref_beta={c.get('pref_beta')} expected 0.1")
    print(f"  [{status}] {fname}")

# ---------------------------------------------------------------------------
print("\n=== Summary ===")
if ERRORS:
    print(f"FAIL — {len(ERRORS)} error(s):")
    for e in ERRORS: print(f"  [FAIL] {e}")
else:
    print("PASS — all format checks passed")
if WARNS:
    print(f"Warnings ({len(WARNS)}):")
    for w in WARNS: print(f"  [WARN] {w}")
