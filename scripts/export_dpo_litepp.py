import argparse
import json
import os
import sys
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SYSTEM_PROMPT = (
    "You are a traffic signal control agent. "
    "You reason about traffic observations and select optimal signal phases."
)


def convert_to_llamafactory_dpo(pair: dict) -> list:
    """Convert one DPO pair into up to 2 LLaMA Factory DPO items (ATR + RA)."""
    items = []

    atr_prompt = pair.get("atr_prompt", "")
    ra_prompt = pair.get("ra_prompt", "")
    teacher_atr = pair.get("teacher_atr", "")
    teacher_ra = pair.get("teacher_ra", "")
    student_atr = pair.get("student_atr", "")
    student_ra = pair.get("student_ra", "")

    if teacher_atr and student_atr and teacher_atr != student_atr:
        items.append({
            "conversations": [
                {"from": "human", "value": f"{SYSTEM_PROMPT}\n\n{atr_prompt}"}
            ],
            "chosen": {"from": "gpt", "value": teacher_atr},
            "rejected": {"from": "gpt", "value": student_atr},
        })

    if teacher_ra and student_ra and teacher_ra != student_ra:
        items.append({
            "conversations": [
                {"from": "human", "value": f"{SYSTEM_PROMPT}\n\n{ra_prompt}"}
            ],
            "chosen": {"from": "gpt", "value": teacher_ra},
            "rejected": {"from": "gpt", "value": student_ra},
        })

    return items


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 3b: Export DPO pairs to LLaMA Factory format"
    )
    parser.add_argument("--input", type=str,
                        default="data/FinetuneData/litepp/litepp_dpo_pairs.jsonl")
    parser.add_argument("--output_dir", type=str,
                        default="data/FinetuneData/llamafactory_litepp_dpo")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    print(f"Loading DPO pairs from {args.input}")
    raw_pairs = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_pairs.append(json.loads(line))
    print(f"  Loaded {len(raw_pairs)} raw pairs")

    all_items = []
    for pair in raw_pairs:
        all_items.extend(convert_to_llamafactory_dpo(pair))
    print(f"  Converted to {len(all_items)} LLaMA Factory DPO items")

    random.shuffle(all_items)
    split_idx = int(len(all_items) * args.train_ratio)
    train_items = all_items[:split_idx]
    val_items = all_items[split_idx:]
    print(f"  Split: {len(train_items)} train / {len(val_items)} val")

    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "dpo_train.json")
    val_path = os.path.join(args.output_dir, "dpo_val.json")
    info_path = os.path.join(args.output_dir, "dataset_info.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_items, f, indent=2, ensure_ascii=False)

    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_items, f, indent=2, ensure_ascii=False)

    dataset_info = {
        "litepp_dpo_train": {
            "file_name": "dpo_train.json",
            "ranking": True,
            "columns": {
                "prompt": "conversations",
                "chosen": "chosen",
                "rejected": "rejected"
            }
        },
        "litepp_dpo_val": {
            "file_name": "dpo_val.json",
            "ranking": True,
            "columns": {
                "prompt": "conversations",
                "chosen": "chosen",
                "rejected": "rejected"
            }
        }
    }

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

    print(f"Saved dpo_train.json, dpo_val.json, dataset_info.json to {args.output_dir}")


if __name__ == "__main__":
    main()
