"""
Stage 2a — Complexity Labeling

Samples from Stage 1 already contain rollout_results and pseudo_golden_action
(computed inline during sampling). This script only attaches the complexity
label ("easy" / "Complex") based on neighbour congestion — matching the
paper's binary system (arXiv:2503.11739).

CityFlow is NOT required here.
"""
import os
import json
import yaml
import argparse
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.litepp_complexity import ComplexityAnalyzer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/collmlight_litepp.yaml")
    parser.add_argument("--input",  type=str,
                        default="data/FinetuneData/litepp/litepp_rco_raw.jsonl")
    parser.add_argument("--output", type=str,
                        default="data/FinetuneData/litepp/litepp_rco_rollout.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        lite_config = yaml.safe_load(f)

    complex_analyzer = ComplexityAnalyzer(lite_config.get("complexity", {}))
    action_list_str  = lite_config.get("action_space", ["ETWT", "NTST", "ELWL", "NLSL"])

    samples = []
    with open(args.input, "r") as fin:
        for line in fin:
            if line.strip():
                samples.append(json.loads(line.strip()))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    stats = {
        "total": len(samples),
        "easy": 0, "Complex": 0,
        "actions": {a: 0 for a in action_list_str},
    }

    with open(args.output, "w") as fout:
        for sample in samples:
            # Validate that Stage 1 produced the required fields
            if "rollout_results" not in sample:
                raise ValueError(
                    f"Sample for {sample.get('intersection_id')} is missing "
                    "'rollout_results'. Re-run sample_litepp_cityflow.py."
                )
            if "pseudo_golden_action" not in sample:
                raise ValueError(
                    f"Sample for {sample.get('intersection_id')} is missing "
                    "'pseudo_golden_action'. Re-run sample_litepp_cityflow.py."
                )

            # Attach complexity label — now binary: "easy" or "Complex"
            complex_analyzer.attach_complexity(sample)
            lbl = sample["complexity"]["label"]
            stats[lbl] += 1
            stats["actions"][sample["pseudo_golden_action"]] += 1

            fout.write(json.dumps(sample) + "\n")

    print(f"Complexity labeling done. Processed {stats['total']} samples.")
    print(f"  easy={stats['easy']}  Complex={stats['Complex']}")
    print(f"  Action dist: {stats['actions']}")


if __name__ == "__main__":
    main()
