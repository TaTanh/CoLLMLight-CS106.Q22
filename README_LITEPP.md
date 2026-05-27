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

7. **PR Dataset Refinement Construction (Calls Student endpoint)**
   After deploying your trained student model to a local vLLM API (e.g. `localhost:8000/v1`), build the preference dataset:
   ```bash
   python scripts/build_refinement_litepp.py --input data/FinetuneData/llamafactory_litepp/train.json --output data/FinetuneData/litepp/pr_dataset.jsonl --endpoint http://localhost:8000/v1
   ```

8. **Training PR Refinement through SFT**
   Execute SFT directly using Kaggle or your local GPU across the PR configs constructed:
   ```bash
   llamafactory-cli train config/llamafactory_pr_qwen1_5b.yaml
   ```

9. **End-to-End CityFlow Evaluation (Calls Student endpoint)**
   Run simulation loop connecting CityFlow and the trained model. *(Note: NewYork datasets are explicitly disallowed via choices constraint)*:
   ```bash
   python scripts/evaluate_litepp_student.py --dataset synth --endpoint http://localhost:8000/v1
   ```
