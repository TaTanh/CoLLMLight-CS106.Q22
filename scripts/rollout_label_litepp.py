import os
import json
import yaml
import time
import argparse
import numpy as np

# Adjust python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import dic_traffic_env_conf
from utils.utils import merge
from utils.litepp_complexity import ComplexityAnalyzer

# Optional CityFlow import (may not be available)
try:
    from utils.cityflow_env import CityFlowEnv
    from scripts.sample_litepp_cityflow import copy_cityflow_file
    CITYFLOW_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CITYFLOW_AVAILABLE = False
    CityFlowEnv = None
    copy_cityflow_file = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/collmlight_litepp.yaml")
    parser.add_argument("--input", type=str, default="data/FinetuneData/litepp/litepp_rco_raw.jsonl")
    parser.add_argument("--output", type=str, default="data/FinetuneData/litepp/litepp_rco_rollout.jsonl")
    return parser.parse_args()

def evaluate_rollout(env, intersection_name, action_idx, rollout_horizon):
    """
    Sets signal to `action_idx`, steps env `rollout_horizon` times,
    and calculates total queue length.
    """
    for inter in env.list_intersection:
        inter.set_signal(action_idx, "set", yellow_time=5, path_to_log=env.path_to_work_directory)

    total_queue = 0
    total_wait = 0.0

    for _ in range(rollout_horizon):
        env.step([action_idx] * len(env.list_intersection))
        
        # accumulate metric for pseudo golden
        for inter in env.list_intersection:
            if inter.inter_name == intersection_name:
                for lane in inter.list_entering_lanes:
                    q = inter.dic_lane_waiting_vehicle_count_current_step.get(lane, 0)
                    total_queue += q
                    total_wait += q # Approx AWT proxy

    return total_queue, total_wait

def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        lite_config = yaml.safe_load(f)
        
    complex_analyzer = ComplexityAnalyzer(lite_config.get("complexity", {}))

    # Read all samples
    samples = []
    with open(args.input, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_f = open(args.output, "w")

    action_list_str = lite_config.get("action_space", ["ETWT", "NTST", "ELWL", "NLSL"])
    rollout_horizon = lite_config.get("rollout_horizon", 5)

    stats = {
        "NO": 0, "Simple": 0, "Complex": 0,
        "actions": {a: 0 for a in action_list_str}
    }

    print(f"Beginning rollout labeling for {len(samples)} samples...")

    if not CITYFLOW_AVAILABLE:
        print("[WARNING] CityFlow not available, using synthetic rollout results")
        print("(This is fine for pipeline testing - install cityflow for real simulation)")
        print()

    for idx, sample in enumerate(samples):
        # 1. Attach Complexity
        complex_analyzer.attach_complexity(sample)
        lbl = sample["complexity"]["label"]
        stats[lbl] += 1

        dataset = sample["dataset"]
        replay_info = sample.get("replay", {})
        actions_before = replay_info.get("actions_before_timestep", [])

        # Re-build Environment Configuration per sample (deterministic reset)
        num_row, num_col = 0, 0
        if dataset == "synth":
            template, road_net, traffic_file = "Synthetic", "4_4", "anon_4_4_synthetic_8000.json"
            num_row, num_col = 4, 4
        elif dataset == "jinan_3x4":
            template, road_net, traffic_file = "Jinan", "3_4", "anon_3_4_jinan_real_2000.json"
            num_row, num_col = 3, 4
        elif dataset == "hangzhou_4x4":
            template, road_net, traffic_file = "Hangzhou", "4_4", "anon_4_4_hangzhou_real.json"
            num_row, num_col = 4, 4

        num_inters = num_row * num_col
        work_dir = f"records/litepp_rollout_tmp_{idx}"
        os.makedirs(work_dir, exist_ok=True)

        dic_traffic_env_conf_extra = {
            "NUM_AGENTS": num_inters,
            "NUM_INTERSECTIONS": num_inters,
            "NUM_ROW": num_row,
            "NUM_COL": num_col,
            "TRAFFIC_FILE": traffic_file,
            "ROADNET_FILE": f"roadnet_{road_net}.json",
            "MODEL_NAME": "LitePPRollout",
            "PHASE": {1:[0,1,0,1,0,0,0,0], 2:[0,0,0,0,0,1,0,1], 3:[1,0,1,0,0,0,0,0], 4:[0,0,0,0,1,0,1,0]},
            "INTERVAL": 1
        }
        env_conf = merge(dic_traffic_env_conf, dic_traffic_env_conf_extra)

        rollout_results = {}

        # Evaluate all 4 candidate actions
        if CITYFLOW_AVAILABLE:
            # Real simulation with CityFlow
            for a_idx, action_str in enumerate(action_list_str):
                dic_path = {
                    "PATH_TO_DATA": os.path.join("data", template, road_net),
                    "PATH_TO_WORK_DIRECTORY": work_dir
                }
                copy_cityflow_file(dic_path, env_conf)

                env = CityFlowEnv(
                    path_to_log=work_dir,
                    path_to_work_directory=work_dir,
                    dic_traffic_env_conf=env_conf,
                    dic_path=dic_path
                )
                env.reset()

                # Replay exactly up to the timestep
                for past_action_obj in actions_before:
                    action_list = []
                    for inter in env.list_intersection:
                        act_idx = past_action_obj["actions"].get(inter.inter_name, 0)
                        action_list.append(act_idx)
                    env.step(action_list)

                # Roll out the candidate
                q5, w5 = evaluate_rollout(env, sample["intersection_id"], a_idx, rollout_horizon)

                rollout_results[action_str] = {
                    "queue_after_5": q5,
                    "wait_after_5": w5,
                    "future_state_summary": f"Queue goes to {q5}"
                }
        else:
            # Synthetic fallback (no CityFlow available)
            local_lanes = sample.get("current_observation", {}).get("local_lanes", {})
            current_queue = sum(lane.get("queue", 0) for lane in local_lanes.values())

            for a_idx, action_str in enumerate(action_list_str):
                # Synthetic rollout: action 0 reduces queue, others keep it
                if a_idx == 0:
                    q5 = max(0, current_queue - 5)
                else:
                    q5 = current_queue + (a_idx - 1) * 2

                rollout_results[action_str] = {
                    "queue_after_5": q5,
                    "wait_after_5": float(q5),
                    "future_state_summary": f"Synthetic rollout: queue {q5}"
                }

        sample["rollout_results"] = rollout_results

        # Pseudo Golden Action Selection (argmin queue, tie-break wait)
        best_act = action_list_str[0]
        best_q = float('inf')
        best_w = float('inf')

        for a_idx, action_str in enumerate(action_list_str):
            curr_res = rollout_results[action_str]
            q_val = curr_res["queue_after_5"]
            w_val = curr_res["wait_after_5"]

            if q_val < best_q:
                best_q = q_val
                best_w = w_val
                best_act = action_str
            elif q_val == best_q:
                if w_val < best_w:
                    best_w = w_val
                    best_act = action_str
                # Priority tie break naturally follows string action order execution

        sample["pseudo_golden_action"] = best_act
        stats["actions"][best_act] += 1

        out_f.write(json.dumps(sample) + "\n")

    out_f.close()
    print("Rollout completed.")
    print("--- Stats ---")
    print(f"Total: {len(samples)}")
    print(f"Complexity: {stats['NO']} NO, {stats['Simple']} Simple, {stats['Complex']} Complex")
    print("Action dist:", stats["actions"])

if __name__ == "__main__":
    main()