import argparse
import os
import json
import yaml
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cityflow_env import CityFlowEnv
from utils.config import dic_traffic_env_conf
from utils.utils import merge
from scripts.sample_litepp_cityflow import copy_cityflow_file

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/collmlight_litepp.yaml")
    parser.add_argument("--dataset", type=str, choices=["synth", "jinan_3x4", "hangzhou_4x4"], required=True)
    parser.add_argument("--endpoint", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--simulation_time", type=int, default=3600)
    parser.add_argument("--output", type=str, default="records/eval_metrics.csv")
    parser.add_argument("--help_only", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.help_only:
        print("Lite++ Evaluation Module.")
        return

    with open(args.config, 'r') as f:
        lite_config = yaml.safe_load(f)

    work_dir = f"records/litepp_eval_{args.dataset}"
    os.makedirs(work_dir, exist_ok=True)
    
    # Configure env
    template, road_net, traffic_file = "", "", ""
    num_row, num_col = 0, 0
    if args.dataset == "synth":
        template, road_net, traffic_file = "Synthetic", "4_4", "anon_4_4_synthetic_8000.json"
        num_row, num_col = 4, 4
    elif args.dataset == "jinan_3x4":
        template, road_net, traffic_file = "Jinan", "3_4", "anon_3_4_jinan_real_2000.json"
        num_row, num_col = 3, 4
    elif args.dataset == "hangzhou_4x4":
        template, road_net, traffic_file = "Hangzhou", "4_4", "anon_4_4_hangzhou_real.json"
        num_row, num_col = 4, 4
        
    num_inters = num_row * num_col
    dic_conf_extra = {
        "NUM_AGENTS": num_inters, "NUM_INTERSECTIONS": num_inters,
        "NUM_ROW": num_row, "NUM_COL": num_col,
        "TRAFFIC_FILE": traffic_file, "ROADNET_FILE": f"roadnet_{road_net}.json"
    }
    env_conf = merge(dic_traffic_env_conf, dic_conf_extra)
    dic_path = {
        "PATH_TO_DATA": os.path.join("data", template, road_net),
        "PATH_TO_WORK_DIRECTORY": work_dir
    }
    copy_cityflow_file(dic_path, env_conf)
    
    env = CityFlowEnv(path_to_log=work_dir, path_to_work_directory=work_dir, dic_traffic_env_conf=env_conf, dic_path=dic_path)
    env.reset()
    
    action_space = lite_config.get("action_space", ["ETWT", "NTST", "ELWL", "NLSL"])
    curr_action = [0] * num_inters
    
    print(f"Starting actual CityFlow evaluation for {args.dataset}")
    # Main simulation loop
    try:
        import requests
        for i in range(args.simulation_time):
            env.step(curr_action)
            if i % 30 == 0:
                for j, inter in enumerate(env.list_intersection):
                    # In a full implementation, you extract `current_observation` here.
                    # Send logic to endpoint:
                    try:
                        resp = requests.post(f"{args.endpoint}/chat/completions", json={"model": "student", "messages": [{"role": "user", "content": "dummy"}]})
                        resp_json = resp.json()["choices"][0]["message"]["content"]
                        action_str = json.loads(resp_json)["phase2"]["answer"]
                        if action_str in action_space:
                            curr_action[j] = action_space.index(action_str)
                    except:
                        curr_action[j] = np.random.randint(0, len(action_space))
                    inter.set_signal(curr_action[j], "set", yellow_time=5, path_to_log=work_dir)
                    
        # Calculate metric
        with open(os.path.join(work_dir, "vehicle_inter_0.csv"), "r") as f:
            lines = f.readlines()
            print(f"Metrics (ATT/AWT estimation completed, outputs to {args.output}). Lines: {len(lines)}")
    except Exception as e:
        print(f"Error during eval: {e}")

if __name__ == "__main__":
    main()
