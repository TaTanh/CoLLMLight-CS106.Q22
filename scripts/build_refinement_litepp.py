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
    parser.add_argument("--input", type=str, required=True, help="LLaMA export JSON (e.g. train.json)")
    parser.add_argument("--output", type=str, required=True, help="Output PR JSONL dataset")
    parser.add_argument("--endpoint", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--help_only", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.help_only:
        print("Build refinement module for PR dataset generation.")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.input, "r") as f:
        samples = json.load(f)
        
    pr_items = []
    print(f"Loaded {len(samples)} items for refinement building.")
    
    with open(args.output, "w") as out:
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
                out.write(json.dumps(pr_item) + "\n")
                pr_items.append(pr_item)
                
    print(f"Refinement PR Generation Complete. Generated {len(pr_items)} PR pairs.")

if __name__ == "__main__":
    main()
