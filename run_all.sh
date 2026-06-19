#!/bin/bash
source .venv/bin/activate

echo "Running Jinan..."
python scripts/sample_litepp_cityflow.py --dataset jinan_3x4 --num_samples 1000 --simulation_time 200 --output data/FinetuneData/litepp/litepp_rco_raw_jinan.jsonl

echo "Running Hangzhou..."
python scripts/sample_litepp_cityflow.py --dataset hangzhou_4x4 --num_samples 1000 --simulation_time 200 --output data/FinetuneData/litepp/litepp_rco_raw_hangzhou.jsonl

echo "Merging..."
cat data/FinetuneData/litepp/litepp_rco_raw_jinan.jsonl data/FinetuneData/litepp/litepp_rco_raw_hangzhou.jsonl > data/FinetuneData/litepp/litepp_rco_raw_1.jsonl

echo "Labeling..."
python scripts/rollout_label_litepp.py --input data/FinetuneData/litepp/litepp_rco_raw_1.jsonl --output data/FinetuneData/litepp/litepp_rco_rollout_1.jsonl

echo "DONE"
