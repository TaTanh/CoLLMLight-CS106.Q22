# CoLLMLight Lite++ — Phân tích Sai lệch so với Paper Gốc

> Ngày kiểm tra: 2026-05-27  
> So sánh: `CoLLMLight-CS106.Q22/` (bản Lite) vs. paper arXiv:2503.11739 + `original/CoLLMLight-CS106.Q22/`

---

## Tổng quan mức độ tuân thủ

| Thành phần | Mức tuân thủ | Mức độ ảnh hưởng |
|------------|-------------|-----------------|
| Stage 1 — Sampling policy | 60% | Trung bình |
| Stage 2a — Rollout labeling | 50% | Cao |
| Stage 2b — Reasoning chain structure | 70% | Trung bình |
| Stage 3 — Policy Refinement (PPO) | 30% | **Rất cao** |
| Stage 3 — PR code vs. mô tả checkpoint | 0% khớp | **Nghiêm trọng** |
| Inference — Yellow/all-red timing | 80% | Thấp |
| Tài liệu — PIPELINE_ANALYSIS_CHECKPOINT.md | Sai lệch | Trung bình |

---

## Vấn đề 1 — Sampling: Random Policy thay FTS Policy

### Paper gốc
`run_fts.py` dùng **FTS (Fine-Tuning Sampling)** — một policy heuristic chủ động:
- Sử dụng reward `avg_ql` (average queue length) để hướng dẫn action selection
- Sinh trajectory đa dạng bao gồm cả trạng thái tắc nghẽn lẫn thông thoáng
- Ground truth `a*` được tích hợp inline trong vòng lặp FTS

### Bản Lite
`scripts/sample_litepp_cityflow.py` dùng **random policy**:
```python
if args.policy == "random":
    action = [np.random.randint(0, 4) for _ in range(num_intersections)]
```

### Hậu quả
- Trajectory không đa dạng — random policy không đảm bảo bao phủ đủ các tình huống traffic
- Phân phối `a*` (pseudo-golden) có thể bị lệch, ảnh hưởng chất lượng SFT

### Cần sửa
Thay `args.policy == "random"` bằng một policy heuristic tối thiểu (ví dụ: MaxPressure hoặc epsilon-greedy `avg_ql`) để sinh trajectory chất lượng hơn.

---

## Vấn đề 2 — Rollout Labeling: Replay từ đầu — Chậm và thiếu chính xác

### Paper gốc
Ground truth `a*` được tính **inline trong vòng lặp sampling** — simulator đang ở đúng timestep t, thử 4 action, chọn action tốt nhất ngay lập tức.

### Bản Lite
`scripts/rollout_label_litepp.py` lưu toàn bộ lịch sử action (`actions_before_timestep`) rồi **replay lại từ đầu** cho mỗi sample × mỗi candidate action:

```python
# Replay exactly up to the timestep
for past_action_obj in actions_before:
    action_list = []
    for inter in env.list_intersection:
        act_idx = past_action_obj["actions"].get(inter.inter_name, 0)
        action_list.append(act_idx)
    env.step(action_list)
```

### Hậu quả
- **Hiệu suất cực thấp**: 1000 samples × 4 actions × (replay t bước + 5 rollout bước) = hàng triệu simulation steps
- **Không chính xác**: CityFlow không hoàn toàn deterministic khi replay — vehicle spawning có thể khác nhau giữa các lần chạy
- Với 1000 samples, Phase 3 có thể mất hàng giờ thay vì ~10 phút

### Cần sửa
Tích hợp rollout labeling **vào trong vòng lặp sampling** (`sample_litepp_cityflow.py`): tại mỗi decision timestep, fork simulator state (hoặc dùng `env.save_state()` nếu CityFlow hỗ trợ), thử 4 actions, ghi `a*`, tiếp tục.

---

## Vấn đề 3 — Reasoning Chain: 2 LLM Call → 1 JSON Call

### Paper gốc
`reasoning_tuning_data_synth.py` dùng **hai prompt độc lập**:
1. **ATR prompt** (Advanced Traffic Reasoning): LLM phân tích traffic tự do, dự đoán tương lai, output text tự nhiên
2. **RA prompt** (Reactive Action): nhận ATR output làm input, chọn signal tốt nhất

Hai bước này tạo ra **chain of thought thật**: LLM phải viết reasoning trước khi biết answer.

### Bản Lite
`scripts/teacher_rewrite_litepp.py` dùng **một JSON call duy nhất**:
```python
user_msg = f"Observation: {...}\nRollout Results: {...}\n{schema_req_content}"
response = client.chat.completions.create(
    model=args.model_name_or_path,
    response_format={"type": "json_object"},
    ...
)
```
Output là JSON với `phase1` và `phase2` được sinh **cùng lúc**.

### Hậu quả
- Mất đi cấu trúc reasoning chain thật sự — LLM biết schema trước khi "suy luận"
- `phase1` (complexity) và `phase2` (action) không độc lập, LLM có thể backward rationalize
- Chất lượng reasoning trace thấp hơn → ảnh hưởng trực tiếp đến SFT quality

### Cần sửa
Tách thành 2 API call riêng biệt:
- Call 1: prompt ATR (phân tích traffic, không cho biết `a*`)
- Call 2: prompt RA (nhận ATR output + local signal priority, chọn action)

---

## Vấn đề 4 — Stage 3: SFT thay PPO — Không có RL Loop thật

### Paper gốc
`ppo_ft.py` thực hiện **PPO (Proximal Policy Optimization)** thật sự:
- Fine-tuned LLM tương tác trực tiếp với CityFlow
- Mỗi bước: model chọn signal → CityFlow trả reward `Q = 1/queue_length` sau 5 timestep
- PPO cập nhật weights để maximize Q
- 552 trajectories bổ sung với gradient updates từ environment feedback

```python
# ppo_ft.py (original)
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer, PPOConfig
# ... vòng lặp RL thật với CityFlow reward
```

### Bản Lite
`scripts/build_refinement_litepp.py` chỉ tạo **SFT pairs khi student sai**:
```python
if student_action != pseudo_golden and student_action != "UNKNOWN":
    pr_item = {
        "instruction": instruction,
        "input": user_msg,
        "output": s["output"]   # teacher answer (pseudo_golden)
    }
```
**Không có CityFlow simulation, không có reward Q, không có gradient từ environment.**

### Hậu quả
- Hoàn toàn thiếu Stage 3 của paper — bản Lite thực chất chỉ có Stage 1 + Stage 2
- Đây là thành phần phân biệt CoLLMLight với LLMLight thuần túy (không có cooperation refinement)
- Theo ablation study trong paper: bỏ Stage 3 (PPO) khiến ATT trên New York tăng từ 921s → 1085s (+18%)

### Cần sửa (theo mức độ khả thi)
- **Phương án A (đúng paper)**: Implement PPO loop với CityFlow — cần GPU và `trl` library
- **Phương án B (DPO thay PPO)**: Dùng DPO (Direct Preference Optimization) với pairs (preferred/rejected) được chọn bởi CityFlow Q score thật — gần với paper hơn, không cần RL loop
- **Phương án C (tối thiểu)**: Giữ SFT nhưng tạo preferred/rejected dựa trên **CityFlow simulation thật** (chạy 4 actions, chọn best/worst theo Q) thay vì chỉ so sánh với pseudo_golden

---

## Vấn đề 5 — Phase 7 trong Code: Không khớp với Tài liệu

### Mô tả trong PIPELINE_ANALYSIS_CHECKPOINT.md (Phase 7)
> *"Student generates 4 signal predictions → Simulate each in CityFlow for 5 steps → Measure environment feedback Q = inverse(total_queue_neighbors)"*

Kèm theo output mẫu có `feedback_score: 0.85`, `feedback_score: 0.42`.

### Thực tế trong code
`scripts/build_refinement_litepp.py` **không chạy CityFlow**, không tính Q, không có feedback_score:
```python
student_resp = resp.json()["choices"][0]["message"]["content"]
student_action = json.loads(student_resp).get("phase2", {}).get("answer", "UNKNOWN")

if student_action != pseudo_golden and student_action != "UNKNOWN":
    pr_item = {...}   # chỉ lấy teacher output, không có Q
```

### Hậu quả
- File checkpoint mô tả behavior **chưa được implement** — có thể gây nhầm lẫn khi đọc
- Output `litepp_pr_dataset.jsonl` không phải định dạng mô tả trong checkpoint (không có `feedback_score`)

### Cần sửa
Một trong hai:
1. Sửa code để thật sự chạy CityFlow simulation trong Phase 7
2. Sửa tài liệu để phản ánh đúng behavior hiện tại (SFT mismatch pairs, không có Q)

---

## Vấn đề 6 — Inference: Yellow Time Sai

### Paper gốc
Mỗi timestep = 35 giây: **30s green + 3s yellow + 2s all-red**

### Bản Lite
`scripts/sample_litepp_cityflow.py` và `rollout_label_litepp.py`:
```python
inter.set_signal(action[j], "set", yellow_time=5, path_to_log=work_dir)
```
**Yellow time = 5s**, không có all-red phase.

### Hậu quả
- Timestep dài hơn (35s → 35s nhưng phân bổ khác) — ảnh hưởng nhỏ đến ATT/AWT
- Thiếu all-red phase có thể ảnh hưởng safety trong simulator

### Cần sửa
Đặt `yellow_time=3` và thêm `all_red_time=2` (nếu CityFlow API hỗ trợ), hoặc giữ `yellow_time=5` nhưng ghi rõ trong tài liệu.

---

## Ghi chú thêm — Số lượng mẫu thực tế từ paper

### Con số trong code gốc
Từ `config.py`: `RUN_COUNTS=3600`, `MIN_ACTION_TIME=30` → 120 timesteps × 16 intersections = **1.920 raw samples** mỗi run.

`reasoning_tuning_data_synth.py` hard-cap ở **300 samples** cho SFT teacher reasoning; phần còn lại (~1.500+) đưa vào `env_rl_data.json` cho PPO.

### Con số "2.802 trajectories" trong ghi chú paper
**Không tìm thấy trong code** — không có dòng nào trong repository tham chiếu đến số này. Có thể:
- Là tổng sau khi chạy nhiều traffic files và format lại cho LLaMA Factory
- Hoặc là số được công bố trong paper nhưng không có code tương ứng trong repo công khai

---

## Tóm tắt — Danh sách việc cần làm theo ưu tiên

### Ưu tiên Cao (ảnh hưởng đến tính đúng đắn của pipeline)

| # | Vấn đề | File cần sửa |
|---|--------|-------------|
| P1 | Phase 7: implement CityFlow simulation thật **hoặc** sửa tài liệu | `scripts/build_refinement_litepp.py` |
| P2 | Rollout labeling: bỏ replay-from-scratch, tích hợp inline | `scripts/rollout_label_litepp.py` + `scripts/sample_litepp_cityflow.py` |
| P3 | Stage 3: implement DPO/PPO với environment feedback thật | Script mới hoặc sửa `build_refinement_litepp.py` |

### Ưu tiên Trung bình (ảnh hưởng đến chất lượng training)

| # | Vấn đề | File cần sửa |
|---|--------|-------------|
| P4 | Reasoning chain: tách thành 2 API call (ATR → RA) | `scripts/teacher_rewrite_litepp.py` |
| P5 | Sampling policy: thay random bằng MaxPressure hoặc epsilon-greedy | `scripts/sample_litepp_cityflow.py` |

### Ưu tiên Thấp (minor, không ảnh hưởng kết quả chính)

| # | Vấn đề | File cần sửa |
|---|--------|-------------|
| P6 | Yellow time: 5s → 3s + all-red 2s | `scripts/sample_litepp_cityflow.py`, `rollout_label_litepp.py`, `evaluate_litepp_student.py` |
| P7 | Sửa PIPELINE_ANALYSIS_CHECKPOINT.md để phản ánh đúng code hiện tại | `PIPELINE_ANALYSIS_CHECKPOINT.md` |

---

## Bảng đối chiếu nhanh

| Thành phần | Paper gốc | Bản Lite hiện tại |
|-----------|-----------|------------------|
| Sampling policy | FTS (heuristic reward-guided) | Random |
| Ground truth a* | Inline trong sampling loop | Separate rollout với replay từ đầu |
| Reasoning generation | 2 LLM calls (ATR → RA) | 1 JSON call (phase1+phase2) |
| Teacher model | GPT-4o | GPT-4o-mini |
| Student model | Llama 3.1 8B | Qwen 2.5-1.5B |
| LoRA config | rank=8, α=16, lr=1e-4 | rank=8, α=16, lr=1e-4 ✅ |
| Stage 3 algorithm | PPO với CityFlow reward | SFT trên mismatch pairs |
| PR pairs source | CityFlow Q = 1/queue | So sánh student vs pseudo_golden |
| Yellow time | 3s yellow + 2s all-red | 5s yellow, không có all-red |
| Complexity routing (n_c) | 3 levels: NO/SIMPLE/COMPLEX ✅ | 3 levels: NO/Simple/Complex ✅ |
| Datasets | Synth + Jinan + Hangzhou + NewYork | Synth + Jinan + Hangzhou (NewYork disabled) ✅ |
