# Stage 1 & Stage 2 — Đặc tả kỹ thuật đầy đủ
# CoLLMLight Lite++ (arXiv:2503.11739)

> Cập nhật: 2026-05-27  
> Trạng thái: Đã implement và verify so với paper gốc  
> Working directory: `f:\TaTanh\`

---

## Mục lục

1. [Tổng quan pipeline](#1-tổng-quan-pipeline)
2. [Stage 1 — Sampling](#2-stage-1--sampling)
3. [Stage 2a — Complexity Labeling](#3-stage-2a--complexity-labeling)
4. [Stage 2b — Teacher Reasoning Chain](#4-stage-2b--teacher-reasoning-chain)
5. [Luồng dữ liệu đầu vào / đầu ra](#5-luồng-dữ-liệu)
6. [Cấu hình](#6-cấu-hình)
7. [Sai lệch so với paper gốc (chấp nhận được)](#7-sai-lệch-còn-lại)

---

## 1. Tổng quan pipeline

```
CityFlow Simulator
       │
       ▼
[Stage 1] sample_litepp_cityflow.py
  • MaxPressure policy
  • Inline snapshot + 4-way rollout tại mỗi decision step
  • Output: litepp_rco_raw.jsonl
       │
       ▼
[Stage 2a] rollout_label_litepp.py
  • Gán complexity label (binary: "easy" / "Complex")
  • Output: litepp_rco_rollout.jsonl
       │
       ▼
[Stage 2b] teacher_rewrite_litepp.py
  • Call 1: ATR prompt → free-text reasoning
  • Call 2: RA prompt  → JSON decision
  • Output: litepp_rco_teacher.jsonl
       │
       ▼
[Stage 3+] export → SFT → PPO/DPO
```

**File chính:**

| Script | Vai trò | Input | Output |
|--------|---------|-------|--------|
| `scripts/sample_litepp_cityflow.py` | Stage 1 | CityFlow env | `litepp_rco_raw.jsonl` |
| `scripts/rollout_label_litepp.py` | Stage 2a | `rco_raw.jsonl` | `litepp_rco_rollout.jsonl` |
| `scripts/teacher_rewrite_litepp.py` | Stage 2b | `rco_rollout.jsonl` | `litepp_rco_teacher.jsonl` |
| `utils/litepp_complexity.py` | Complexity classifier | neighbor_obs | label |

---

## 2. Stage 1 — Sampling

**File:** [`scripts/sample_litepp_cityflow.py`](../scripts/sample_litepp_cityflow.py)

### 2.1 Yêu cầu chính xác

| Yêu cầu | Giá trị |
|---------|---------|
| Policy sinh trajectory | **MaxPressure** (argmax tổng queue của 2 làn phục vụ) |
| Rollout labeling | **Inline** tại timestep t (snapshot → thử 4 actions → restore) |
| Snapshot mechanism | `eng.snapshot()` + `deepcopy(env)` sau khi ngắt `eng` references |
| Rollout horizon | **5 bước** |
| Decision interval | Mỗi **30s** (bỏ qua 50s đầu warm-up) |
| Yellow time | **3s** (paper gốc: 3s yellow + 2s all-red) |
| Ground truth a\* | `argmin(queue_after_5, wait_after_5)` — tie-break bằng wait |
| History window | **5 frames** trước khi bắt đầu collect sample |
| Datasets | `synth` (4×4), `jinan_3x4` (3×4), `hangzhou_4x4` (4×4) |

### 2.2 Thuật toán sampling (pseudocode)

```
for i in range(simulation_time):
    env.step(action)                         # bước 1 giây thực tế

    if i <= 50 or i % 30 != 0:
        continue                             # chỉ xử lý decision steps

    # Trích observation từ trạng thái hiện tại t
    for inter in env.list_intersection:
        all_local_obs[inter.inter_id]    = extract_local_obs(inter)
        all_neighbor_obs[inter.inter_id] = extract_neighbor_obs(inter)
        apply_delta_tracking(all_local_obs[inter.inter_id], history)

    state_t = save_env_state(env)            # snapshot tại t

    for j in range(num_intersections):
        inter = env.list_intersection[j]

        if len(history[inter_id]) == history_window:
            # Inline rollout: thử 4 actions từ state t
            for a_idx, action_str in enumerate(action_space):
                load_env_state(env, state_t)
                q5, w5 = evaluate_5_steps(env, inter_name, a_idx)
                rollout_results[action_str] = {q5, w5}
            load_env_state(env, state_t)     # restore về t

            pseudo_golden = argmin(q5, w5)
            signal_rank   = {a: (current_wait - w5[a]) / 60} per action

            write_sample(...)

        update_history(inter_id, local_obs)

    # MaxPressure: chọn action cho bước tiếp theo
    for j, inter in enumerate(env.list_intersection):
        action[j] = argmax(queue_a + queue_b for phase in action_space)
        inter.set_signal(action[j], yellow_time=3)
```

### 2.3 Hàm cốt lõi

#### `save_env_state(env)` / `load_env_state(env, state)`

```python
def save_env_state(env):
    eng      = env.eng
    snapshot = eng.snapshot()          # C++ state → bytes/string
    # Ngắt C++ refs trước deepcopy
    env.eng = None
    for inter in env.list_intersection:
        inter.eng = None
    env_copy = deepcopy(env)           # Python-side state copy
    # Reconnect live env ngay lập tức
    env.eng = eng
    for inter in env.list_intersection:
        inter.eng = eng
    return {"snapshot": snapshot, "env_copy": env_copy, "eng": eng}

def load_env_state(env, state):
    state["eng"].load(state["snapshot"])   # Restore C++
    env_copy = deepcopy(state["env_copy"])
    env.list_intersection    = env_copy.list_intersection
    env.waiting_vehicle_list = env_copy.waiting_vehicle_list
    env.system_states        = env_copy.system_states
    env.traffic_light_node_dict = env_copy.traffic_light_node_dict
    env.intersection_dict    = {i.inter_name: i for i in env.list_intersection}
    env.eng = state["eng"]
    for inter in env.list_intersection:
        inter.eng = env.eng
```

> **Quan trọng:** Sau `save_env_state`, live env vẫn hoạt động bình thường (eng đã được reconnect). Không như FTSample.py gốc — ở đó eng bị để None sau save.

#### `maxpressure_action(local_obs, action_space)`

```python
def maxpressure_action(local_obs, action_space):
    # action_space = ["ETWT", "NTST", "ELWL", "NLSL"]
    # "ETWT" phục vụ làn ET + WT
    phase_pressure = [
        local_obs.get(p[:2], {}).get("queue", 0)
        + local_obs.get(p[2:], {}).get("queue", 0)
        for p in action_space
    ]
    max_p = max(phase_pressure)
    candidates = [i for i, p in enumerate(phase_pressure) if p == max_p]
    return int(np.random.choice(candidates))   # random tie-break
```

#### `extract_local_obs(inter)`

Trích xuất per-lane observation từ CityFlow intersection object:

```python
local_obs = {
    lane_code: {
        "queue":            int,    # số xe đang chờ (stopped)
        "moving":           int,    # số xe đang di chuyển
        "wait_time":        float,  # proxy = queue (tích lũy)
        "occupancy":        float,  # veh_count * 5m / lane_length, clamp [0,1]
        "queue_change":     int,    # delta so với timestep trước (S1-1)
        "occupancy_change": float,  # delta so với timestep trước (S1-1)
    }
    for lane_code in ["ET", "WT", "EL", "WL", "NT", "ST", "NL", "SL"]
}
```

Lane mapping: `lane_idx 0 → Left (L)`, `lane_idx 1 → Through (T)`, `lane_idx 2+ → bỏ qua`.

#### `extract_neighbor_obs(inter, env, action_space)`

```python
neighbor_obs = {
    "upstream": {
        "intersection_x_y": {
            "total_queue":   int,    # tổng queue tất cả entering lanes
            "total_wait":    float,  # = total_queue (proxy)
            "occupancy_avg": float,  # THỰC: mean(veh*5/lane_len) per lane
            "phase":         str,    # pha hiện tại, e.g. "ETWT"
        }
    },
    "downstream": {}   # cùng format, hiện chỉ populate upstream
}
```

> **Bug đã sửa (S2a-2):** `occupancy_avg` trước đây hardcode `0.0`. Hiện tại tính thực từ `veh_count * 5.0 / lane_length`.

### 2.4 Output format (`litepp_rco_raw.jsonl`)

Mỗi dòng là 1 JSON object:

```json
{
  "dataset":         "synth",
  "intersection_id": "intersection_2_3",
  "timestep":        90.0,
  "current_phase":   "ETWT",

  "current_observation": {
    "local_lanes": {
      "ET": {"queue": 5, "moving": 2, "wait_time": 5.0, "occupancy": 0.35,
             "queue_change": 2, "occupancy_change": 0.05},
      "WT": {"queue": 3, "moving": 1, "wait_time": 3.0, "occupancy": 0.20,
             "queue_change": -1, "occupancy_change": -0.03},
      "... (8 lanes total)": "..."
    }
  },

  "neighbor_observation": {
    "upstream": {
      "intersection_2_4": {
        "total_queue": 12, "total_wait": 12.0,
        "occupancy_avg": 0.42,
        "phase": "NTST"
      }
    },
    "downstream": {}
  },

  "history": [
    {
      "timestep": 60.0, "action": "NTST",
      "local_lanes": {"ET": {...}, "...": "..."},
      "neighbor_lanes": {"upstream": {...}, "downstream": {}}
    },
    "... (5 frames)"
  ],

  "candidate_actions": ["ETWT", "NTST", "ELWL", "NLSL"],

  "rollout_results": {
    "ETWT": {"queue_after_5": 12, "wait_after_5": 12.0, "future_state_summary": "Queue goes to 12"},
    "NTST": {"queue_after_5": 18, "wait_after_5": 18.0, "future_state_summary": "Queue goes to 18"},
    "ELWL": {"queue_after_5": 20, "wait_after_5": 20.0, "future_state_summary": "Queue goes to 20"},
    "NLSL": {"queue_after_5": 16, "wait_after_5": 16.0, "future_state_summary": "Queue goes to 16"}
  },

  "pseudo_golden_action": "ETWT",

  "signal_rank": {
    "ETWT":  0.25,
    "NTST": -0.10,
    "ELWL": -0.13,
    "NLSL": -0.02
  }
}
```

**Giải thích `signal_rank`:** `(current_total_wait - wait_after_5) / 60.0` (đơn vị: phút). Dương = giảm thời gian chờ. Dùng để rank actions trong RA prompt.

### 2.5 Fallback khi CityFlow không available

```
CityFlow không install:
  → synthetic_rollout(): q5 = current_queue - 5 (action 0), +2*(a-1) (action a)
  → signal_rank = {all actions: 0.0}
  → history = []
  → neighbor_observation = {"upstream": {}, "downstream": {}}

CityFlow có nhưng eng.snapshot() không tồn tại:
  → Cùng synthetic_rollout() fallback
  → Observation extraction vẫn thực (env.list_intersection)
```

---

## 3. Stage 2a — Complexity Labeling

**File:** [`scripts/rollout_label_litepp.py`](../scripts/rollout_label_litepp.py)  
**Classifier:** [`utils/litepp_complexity.py`](../utils/litepp_complexity.py)

### 3.1 Yêu cầu chính xác

| Yêu cầu | Giá trị |
|---------|---------|
| Input từ Stage 1 | `rollout_results`, `pseudo_golden_action` phải có sẵn (validate) |
| Điều kiện trigger | neighbor `occupancy_avg >= 0.4` |
| Threshold nguồn | `config/collmlight_litepp.yaml` → `complexity.occupancy_threshold: 0.4` |
| Hệ thống labels | **Binary**: `"easy"` hoặc `"Complex"` (khớp paper) |
| Logic | `nc = count(neighbors with occupancy_avg >= 0.4)` → `"Complex" if nc >= 1 else "easy"` |
| CityFlow required | **Không** — chỉ đọc/ghi JSONL |

### 3.2 Thuật toán `ComplexityAnalyzer`

```python
class ComplexityAnalyzer:
    def is_critical_neighbor(self, neighbor_state):
        # Ngưỡng = 0.4 (từ config), khớp paper: "Occupancy >= 40%"
        return neighbor_state.get("occupancy_avg", 0.0) >= self.occ_thresh

    def compute_nc(self, observation):
        nc = 0
        for direction in ["upstream", "downstream"]:
            for nb_state in observation["neighbor_observation"][direction].values():
                if self.is_critical_neighbor(nb_state):
                    nc += 1
        return nc

    def classify_complexity(self, nc):
        # Binary: khớp paper gốc (easy / complex)
        return "Complex" if nc >= 1 else "easy"
```

### 3.3 Output thêm vào sample

```json
{
  "complexity": {
    "nc":    1,
    "label": "Complex"
  }
}
```

> **Lưu ý:** Stage 2a chỉ thêm field `complexity` vào sample. Tất cả các field từ Stage 1 được giữ nguyên.

---

## 4. Stage 2b — Teacher Reasoning Chain

**File:** [`scripts/teacher_rewrite_litepp.py`](../scripts/teacher_rewrite_litepp.py)

### 4.1 Yêu cầu chính xác

| Yêu cầu | Giá trị |
|---------|---------|
| Số LLM calls | **2** (ATR → RA) — không được dùng 1 JSON call |
| Call 1 (ATR) | Free-text, KHÔNG có `response_format=json_object`, KHÔNG có schema hint |
| Call 2 (RA) | `response_format={"type": "json_object"}`, nhận ATR output làm context |
| ATR prompt variant | **Adaptive**: `"Complex"` thêm network-coordination instruction |
| RA ranking input | `signal_rank` (wait reduction, phút) nếu có; fallback: queue_after_5 |
| `phase1.answer` | Phải bằng `complexity.label` (`"easy"` hoặc `"Complex"`) |
| `phase2.answer` | Phải bằng `pseudo_golden_action` (force alignment trong `repair_schema`) |
| Model mặc định | `gpt-4o-mini` (paper gốc: GPT-4 — tradeoff cost/quality) |
| ATR max tokens | 500 |
| RA max tokens | 400 |

### 4.2 Call 1 — ATR (Advance Traffic Reasoning)

**System prompt:**
```
You are an expert traffic signal control agent managing a four-way intersection
within a larger road network. Your objective is to analyse traffic conditions
and reason about future states to support an optimal signal decision.
```

**User prompt (easy case):**
```
## Historical Observation (last 5 timesteps)
  t=60   phase=NTST  ET:q3, NT:q7
  t=90   phase=NTST  ET:q4, NT:q5
  ...

## Current Observation (t=120)  phase=NTST
  ET: queue=5 moving=2 occ=0.35
  NT: queue=8 moving=1 occ=0.55

## Neighbor State
  [upstream] intersection_2_4: queue=12 phase=ETWT

Analyse the traffic situation at this intersection:
- Identify critical lanes (high queue or occupancy)
- Assess upstream/downstream neighbour conditions
- Predict how traffic will evolve over the next 5 timesteps
- Suggest which signal phases would be most effective and why

Output free-form reasoning text. Do NOT output JSON.
Do NOT select a final action yet.
```

**User prompt (Complex case — extra instruction thêm vào cuối list):**
```
- Assess how upstream/downstream congestion at neighbouring intersections
  will affect your decision and whether releasing or holding traffic will
  benefit the network as a whole (spillback prevention and coordination).
```

**Output:** Free-form text, không có schema. Lưu vào `sample["atr_output"]`.

### 4.3 Call 2 — RA (Reactive Action)

**User prompt:**
```
## Prior Traffic Analysis (ATR)
{atr_output}

## Rollout Simulation Results (5-step lookahead)
  ETWT: queue_after_5=12  wait=12.0
  NTST: queue_after_5=18  wait=18.0
  ELWL: queue_after_5=20  wait=20.0
  NLSL: queue_after_5=16  wait=16.0

## Local Signal Priority (ranked by waiting time reduction)
  1. ETWT  queue_after_5=12  wait_reduction=+0.25 mins
  2. NLSL  queue_after_5=16  wait_reduction=-0.02 mins
  3. NTST  queue_after_5=18  wait_reduction=-0.10 mins
  4. ELWL  queue_after_5=20  wait_reduction=-0.13 mins

Based on the above reasoning and simulation evidence, select the best
signal phase. Respond with a JSON object matching this schema exactly:
{
  "phase1": {
    "answer": "Complex",
    "reason": "<max 1 sentence>"
  },
  "phase2": {
    "traffic_analysis": "<max 2 sentences>",
    "future_state_summary": "<max 1 sentence>",
    "signal_comparison": "<max 1 sentence>",
    "signal_consequence_prediction": {   ← chỉ khi label = "Complex"
      "ETWT": "<1 sentence>",
      "NTST": "<1 sentence>",
      "ELWL": "<1 sentence>",
      "NLSL": "<1 sentence>"
    },
    "answer": "<one of ['ETWT', 'NTST', 'ELWL', 'NLSL']>"
  }
}
```

**Ranking signal_priority:**
- Nếu `signal_rank` tồn tại trong sample: sort **descending** by `signal_rank[action]` (higher = better reduction)
- Fallback: sort **ascending** by `(queue_after_5, wait_after_5)`

### 4.4 `repair_schema()` — Force alignment

Sau khi nhận JSON từ LLM, `repair_schema` **force override**:
```python
p1["answer"] = complexity.label          # không tin LLM tự điền đúng
p2["answer"] = pseudo_golden_action      # không tin LLM tự điền đúng
```

Điều này đảm bảo SFT data luôn có label/action đúng, kể cả khi LLM suy luận sai.

### 4.5 Output thêm vào sample

```json
{
  "atr_output": "## Critical Lanes\nNT has 8 queued vehicles...\n(free text)",

  "teacher_response": {
    "phase1": {
      "answer": "Complex",
      "reason": "Neighbour intersection_2_4 shows 42% occupancy indicating upstream congestion risk."
    },
    "phase2": {
      "traffic_analysis": "ET and NT lanes show high queue; upstream neighbour at 42% occupancy may cause spillback.",
      "future_state_summary": "ETWT reduces network queue from 20 to 12 over 5 steps.",
      "signal_comparison": "ETWT outperforms NTST by 6 vehicles in 5-step rollout.",
      "signal_consequence_prediction": {
        "ETWT": "Reduces ET queue by ~3, stabilises NT.",
        "NTST": "Clears NT but ET queue increases to ~8.",
        "ELWL": "Minimal impact, both lanes near empty.",
        "NLSL": "Slight improvement, NL has 2 queued."
      },
      "answer": "ETWT"
    }
  },

  "teacher_fallback":        false,
  "teacher_schema_repaired": false
}
```

---

## 5. Luồng dữ liệu

### 5.1 Các field bắt buộc qua từng stage

| Field | Thêm bởi | Dùng bởi |
|-------|----------|----------|
| `current_observation.local_lanes` | Stage 1 | Stage 2b (ATR prompt) |
| `neighbor_observation` | Stage 1 | Stage 2a (complexity), Stage 2b (ATR) |
| `history` | Stage 1 | Stage 2b (ATR prompt) |
| `rollout_results` | Stage 1 | Stage 2a (validate), Stage 2b (RA prompt) |
| `pseudo_golden_action` | Stage 1 | Stage 2a (stats), Stage 2b (repair_schema) |
| `signal_rank` | Stage 1 | Stage 2b (priority ranking) |
| `complexity.label` | Stage 2a | Stage 2b (adaptive ATR, RA schema) |
| `atr_output` | Stage 2b | — (lưu trace) |
| `teacher_response` | Stage 2b | Stage 3 (SFT training) |

### 5.2 Dependency chain

```
sample_litepp_cityflow.py
  → rollout_results + pseudo_golden_action + signal_rank
        ↓
  rollout_label_litepp.py
  → complexity.label
        ↓
  teacher_rewrite_litepp.py
  → atr_output + teacher_response
        ↓
  export_llamafactory_litepp.py
  → SFT training data (instruction, input, output)
```

---

## 6. Cấu hình

**File:** [`config/collmlight_litepp.yaml`](../config/collmlight_litepp.yaml)

```yaml
action_space:    [ETWT, NTST, ELWL, NLSL]
history_window:  5          # số frames lịch sử trong mỗi sample
rollout_horizon: 5          # số bước lookahead để tính pseudo_golden

complexity:
  occupancy_threshold: 0.4  # >= 40% → neighbor critical (paper ngưỡng)
  queue_threshold:     15   # unused trong binary classifier hiện tại
  wait_threshold:      60.0 # unused trong binary classifier hiện tại

teacher:
  model_name_or_path: gpt-4o-mini
  atr_max_tokens:     500
  ra_max_tokens:      400
```

**Datasets được hỗ trợ:**

| Key | Grid | Traffic file | Ghi chú |
|-----|------|-------------|---------|
| `synth` | 4×4 | `anon_4_4_synthetic_8000.json` | Default |
| `jinan_3x4` | 3×4 | `anon_3_4_jinan_real_2000.json` | Real traffic |
| `hangzhou_4x4` | 4×4 | `anon_4_4_hangzhou_real.json` | Real traffic |
| `newyork` | — | — | Disabled trong Lite (`disable_new_york: true`) |

---

## 7. Sai lệch còn lại so với paper gốc

| ID | Thành phần | Paper gốc | Lite hiện tại | Ảnh hưởng |
|----|-----------|-----------|---------------|-----------|
| S1-3 | Sampling policy | FTS (LLM agent + avg_ql reward) | MaxPressure | Thấp — trajectory ít đa dạng hơn |
| S2a-2* | Neighbor detail | Per-lane occupancy từ Traffic_state_history | `occupancy_avg` tổng hợp per intersection | Thấp — đã tính thực, chỉ thiếu per-lane |
| S2b-3 | Teacher model | GPT-4 | gpt-4o-mini | Trung bình — reasoning depth thấp hơn ~20% |
| — | Yellow time | 3s yellow + **2s all-red** | 3s yellow, **không có all-red** | Thấp — CityFlow API không expose tham số này |
| — | Stage 3 | PPO với CityFlow reward | SFT trên mismatch pairs | **Rất cao** — thiếu RL loop (xem stage3 plan) |

\* `occupancy_avg` đã tính thực từ vehicle count, nhưng chỉ là trung bình tất cả entering lanes — paper dùng per-lane detail để check từng làn upstream/downstream.

---

## 8. Checklist chạy pipeline

```bash
# Stage 1: Sampling (cần CityFlow)
python scripts/sample_litepp_cityflow.py \
    --dataset synth \
    --num_samples 1000 \
    --output data/FinetuneData/litepp/litepp_rco_raw.jsonl

# Stage 2a: Complexity labeling (không cần CityFlow)
python scripts/rollout_label_litepp.py \
    --input  data/FinetuneData/litepp/litepp_rco_raw.jsonl \
    --output data/FinetuneData/litepp/litepp_rco_rollout.jsonl

# Stage 2b: Teacher reasoning (cần OPENAI_API_KEY)
python scripts/teacher_rewrite_litepp.py \
    --input  data/FinetuneData/litepp/litepp_rco_rollout.jsonl \
    --output data/FinetuneData/litepp/litepp_rco_teacher.jsonl \
    --model_name_or_path gpt-4o-mini

# Dry-run (không cần API key, kiểm tra pipeline):
python scripts/teacher_rewrite_litepp.py --dry_run
```
