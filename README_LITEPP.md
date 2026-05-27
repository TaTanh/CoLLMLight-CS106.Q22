# Lite++ Pipeline (CoLLMLight)

## Phases & End-to-End Workflow

**(Important: Always run `export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` first!)**

1. **Environment Setup**
   Activate virtual environment and export variables.
   
2. **Sampling (Local CPU)**
   ```bash
   python scripts/sample_litepp_cityflow.py --dataset synth --num_samples 8000 --output data/FinetuneData/litepp/raw.jsonl
   ```

3. **Rollout Labeling (Local CPU / Pseudo Golden Action)**
   ```bash
   python scripts/rollout_label_litepp.py --input data/FinetuneData/litepp/raw.jsonl --output data/FinetuneData/litepp/rollout.jsonl
   ```

4. **Teacher Reasoning (Calls GPT-4o-mini API)**
   *Requires `OPENAI_API_KEY` and `OPENAI_BASE_URL`.*
   ```bash
   python scripts/teacher_rewrite_litepp.py --input data/FinetuneData/litepp/rollout.jsonl --output data/FinetuneData/litepp/teacher.jsonl
   ```

5. **LLaMA Factory Export (Local CPU)**
   ```bash
   python scripts/export_llamafactory_litepp.py --train data/FinetuneData/litepp/teacher.jsonl --out_dir data/FinetuneData/llamafactory_litepp/
   ```

6. **Training via LLaMA Factory (Kaggle / GPU / RCO Strategy)**
   Run LLaMA Factory's training command on cloud using the generated configs:
   ```bash
   llamafactory-cli train config/llamafactory_rco_qwen1_5b.yaml
   ```

7. **Stage 3a — DPO Pair Generation (Calls Student endpoint)**
   After deploying your SFT-trained student model to a local vLLM API, run the refinement loop to collect DPO pairs from wrong cases (10 iterations):
   ```bash
   python scripts/refinement_litepp.py \
       --input      data/FinetuneData/litepp/litepp_rco_teacher.jsonl \
       --endpoint   http://localhost:8000/v1 \
       --model      student-sft-v1 \
       --iterations 10 \
       --output     data/FinetuneData/litepp/litepp_dpo_pairs.jsonl
   ```

   *Dry-run (no API needed):*

   ```bash
   python scripts/refinement_litepp.py --dry_run \
       --input  data/FinetuneData/litepp/litepp_rco_teacher.jsonl \
       --output data/FinetuneData/litepp/litepp_dpo_pairs.jsonl
   ```

8. **Stage 3b — DPO Export for LLaMA Factory**
   Convert DPO pairs to LLaMA Factory format (ATR + RA pairs, train/val split):
   ```bash
   python scripts/export_dpo_litepp.py \
       --input      data/FinetuneData/litepp/litepp_dpo_pairs.jsonl \
       --output_dir data/FinetuneData/llamafactory_litepp_dpo
   ```

9. **DPO Training via LLaMA Factory**
   ```bash
   llamafactory-cli train config/llamafactory_dpo_qwen1_5b.yaml
   ```

10. **End-to-End CityFlow Evaluation (Calls Student endpoint)**
    Run simulation loop connecting CityFlow and the trained model.
    *(Note: NewYork datasets are explicitly disallowed via choices constraint)*:

    ```bash
    python scripts/evaluate_litepp_student.py \
        --dataset  synth \
        --endpoint http://localhost:8000/v1 \
        --model    student-dpo-v1 \
        --output   outputs/eval_synth_dpo.csv
    ```
