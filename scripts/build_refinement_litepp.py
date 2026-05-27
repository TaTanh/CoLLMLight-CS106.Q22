import argparse
import os
import json
import requests
import yaml
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cityflow_env import CityFlowEnv
from utils.config import dic_traffic_env_conf
from utils.utils import merge
from scripts.sample_litepp_cityflow import copy_cityflow_file

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/collmlight_litepp.yaml")
    parser.add_argument("--train", type=str, default="data/FinetuneData/llamafactory_litepp_rco/train.json", help="LLaMA export JSON (e.g. train.json)")
    parser.add_argument("--out_dir", type=str, default="data/FinetuneData/llamafactory_litepp_pr", help="Output directory for PR dataset")
    parser.add_argument("--endpoint", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train/val split ratio")
    parser.add_argument("--help_only", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.help_only:
        print("Build refinement module for PR dataset generation.")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.train, "r") as f:
        samples = json.load(f)

    pr_items = []
    print(f"Loaded {len(samples)} items for refinement building.")

    for idx, s in enumerate(samples):
        # Parse raw context from instruction input format
        instruction = s["instruction"]
        user_msg = s["input"]
        pseudo_golden = json.loads(s["output"]).get("phase2", {}).get("answer", "UNKNOWN")

        # API CALL TO STUDENT
        try:
            resp = requests.post(
                f"{args.endpoint}/chat/completions",
                json={
                    "model": "student",
                    "messages": [
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": user_msg}
                    ]
                },
                timeout=10
            )
            student_resp = resp.json()["choices"][0]["message"]["content"]
            student_action = json.loads(student_resp).get("phase2", {}).get("answer", "UNKNOWN")
        except Exception as e:
            # If API fails/offline, use placeholder for safety
            print(f"Skipping API call for item {idx}: {e}")
            student_action = pseudo_golden

        # Create PR sample if mismatch (SFT Format)
        if student_action != pseudo_golden and student_action != "UNKNOWN":
            pr_item = {
                "instruction": instruction,
                "input": user_msg,
                "output": s["output"]  # Current teacher response containing pseudo_golden (best_action_by_env)
            }
            pr_items.append(pr_item)

    print(f"Generated {len(pr_items)} PR pairs from {len(samples)} samples.")

    # Split into train/val
    split_idx = int(len(pr_items) * args.train_ratio)
    train_items = pr_items[:split_idx]
    val_items = pr_items[split_idx:]

    print(f"Split: {len(train_items)} train, {len(val_items)} val")

    # Save as JSON arrays
    train_path = os.path.join(args.out_dir, "pr_train.json")
    val_path = os.path.join(args.out_dir, "pr_val.json")

    with open(train_path, "w") as f:
        json.dump(train_items, f, indent=2)

    with open(val_path, "w") as f:
        json.dump(val_items, f, indent=2)

    # Create dataset_info.json for LLaMA Factory
    dataset_info = {
        "litepp_pr_train": {
            "file_name": "pr_train.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output"
            }
        },
        "litepp_pr_val": {
            "file_name": "pr_val.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output"
            }
        }
    }

    dataset_info_path = os.path.join(args.out_dir, "dataset_info.json")
    with open(dataset_info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"Saved pr_train.json, pr_val.json, and dataset_info.json to {args.out_dir}")

if __name__ == "__main__":
    main()
