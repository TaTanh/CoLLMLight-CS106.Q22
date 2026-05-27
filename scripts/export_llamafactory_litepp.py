import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.teacher_rewrite_litepp import (
    ATR_SYSTEM,
    build_atr_prompt,
    build_ra_prompt,
)


def _signal_priority(sample):
    candidates = sample.get("candidate_actions", ["ETWT", "NTST", "ELWL", "NLSL"])
    sr = sample.get("signal_rank", None)
    rollout = sample.get("rollout_results", {})
    if sr:
        return sorted(candidates, key=lambda a: sr.get(a, 0.0), reverse=True)
    return sorted(candidates, key=lambda a: (
        rollout.get(a, {}).get("queue_after_5", 0),
        rollout.get(a, {}).get("wait_after_5", 0.0),
    ))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "--train", type=str,
                        default="data/FinetuneData/litepp/litepp_rco_teacher.jsonl",
                        dest="input")
    parser.add_argument("--output_dir", "--out_dir", type=str,
                        default="data/FinetuneData/llamafactory_litepp_rco",
                        dest="output_dir")
    parser.add_argument("--split_ratio", type=float, default=0.9)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    samples = []
    with open(args.input, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    llama_format = []
    skipped = 0
    for s in samples:
        atr_output = s.get("atr_output", "")
        teacher_resp = s.get("teacher_response", {})

        if not atr_output or not teacher_resp:
            skipped += 1
            continue

        # Pair 1 — ATR: observation → free-text reasoning
        llama_format.append({
            "instruction": ATR_SYSTEM,
            "input": build_atr_prompt(s),
            "output": atr_output,
        })

        # Pair 2 — RA: ATR output + rollout → JSON decision
        llama_format.append({
            "instruction": ATR_SYSTEM,
            "input": build_ra_prompt(s, atr_output, _signal_priority(s)),
            "output": json.dumps(teacher_resp, ensure_ascii=False),
        })

    if skipped:
        print(f"Skipped {skipped} samples missing atr_output or teacher_response.")

    split_idx = int(len(llama_format) * args.split_ratio)
    train_data = llama_format[:split_idx]
    val_data = llama_format[split_idx:]

    with open(os.path.join(args.output_dir, "train.json"), "w", encoding="utf-8") as fout:
        json.dump(train_data, fout, indent=2, ensure_ascii=False)

    if val_data:
        with open(os.path.join(args.output_dir, "val.json"), "w", encoding="utf-8") as fout:
            json.dump(val_data, fout, indent=2, ensure_ascii=False)

    dataset_info = {
        "litepp_train": {"file_name": "train.json"},
        "litepp_val":   {"file_name": "val.json"},
    }
    with open(os.path.join(args.output_dir, "dataset_info.json"), "w", encoding="utf-8") as fout:
        json.dump(dataset_info, fout, indent=2, ensure_ascii=False)

    n_samples = len(samples) - skipped
    print(f"Exported {len(llama_format)} SFT pairs from {n_samples} samples "
          f"({len(train_data)} train / {len(val_data)} val).")

    if train_data:
        print("====== PREVIEW PAIR 1 (ATR) ======")
        print(train_data[0]["input"][:400] + "...\n[TRUNCATED]")
        print("====== OUTPUT ======")
        print(train_data[0]["output"][:300])


if __name__ == "__main__":
    main()
