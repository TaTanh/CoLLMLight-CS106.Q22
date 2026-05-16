import argparse
import os
import json
import yaml
import sys
import numpy as np
import requests

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
    parser.add_argument("--output", type=str, default="outputs/litepp_eval_results.csv")
    parser.add_argument("--help_only", action="store_true")
    return parser.parse_args()

def build_observation_prompt(inter, env, action_space):
    """Extract real observation from intersection and build prompt for student model."""
    # Build local observation
    local_obs = {
        "ET": {"queue": 0, "moving": 0, "wait_time": 0.0, "occupancy": 0.0},
        "WT": {"queue": 0, "moving": 0, "wait_time": 0.0, "occupancy": 0.0},
        "EL": {"queue": 0, "moving": 0, "wait_time": 0.0, "occupancy": 0.0},
        "WL": {"queue": 0, "moving": 0, "wait_time": 0.0, "occupancy": 0.0},
        "NT": {"queue": 0, "moving": 0, "wait_time": 0.0, "occupancy": 0.0},
        "ST": {"queue": 0, "moving": 0, "wait_time": 0.0, "occupancy": 0.0},
        "NL": {"queue": 0, "moving": 0, "wait_time": 0.0, "occupancy": 0.0},
        "SL": {"queue": 0, "moving": 0, "wait_time": 0.0, "occupancy": 0.0},
    }

    for lane in inter.list_entering_lanes:
        q = inter.dic_lane_waiting_vehicle_count_current_step.get(lane, 0)
        veh_list = inter.dic_lane_vehicle_current_step_in.get(lane, [])
        moving = max(0, len(veh_list) - q)

        road = lane.split('_')[:-1]
        edge = "_".join(road)
        lane_idx = int(lane.split('_')[-1])

        approach = None
        for ap, inc_edge in inter.dic_entering_approach_to_edge.items():
            if inc_edge == edge:
                approach = ap
                break
        if approach:
            if lane_idx == 0:
                logic = f"{approach}L"
            elif lane_idx == 1:
                logic = f"{approach}T"
            else:
                continue

            if logic in local_obs:
                local_obs[logic]["queue"] = q
                local_obs[logic]["moving"] = moving
                local_obs[logic]["wait_time"] += q
                lane_len = inter.lane_length.get(lane, 100)
                local_obs[logic]["occupancy"] = min(1.0, len(veh_list) * 5.0 / lane_len)

    # Build neighbor observation
    neighbor_obs = {"upstream": {}, "downstream": {}}
    node_dict = env.traffic_light_node_dict[inter.inter_name]
    for nb_k in ["neighbor_ENWS", "neighbor_up_down_stream"]:
        if nb_k in node_dict:
            for nb_name in node_dict[nb_k]:
                if nb_name and nb_name != "null":
                    for adj_inter in env.list_intersection:
                        if adj_inter.inter_name == nb_name:
                            q_sum = 0
                            for lane in adj_inter.list_entering_lanes:
                                q_sum += adj_inter.dic_lane_waiting_vehicle_count_current_step.get(lane, 0)

                            idx = adj_inter.current_phase_index - 1
                            ph_str = action_space[idx] if 0 <= idx < len(action_space) else "UNKNOWN"
                            neighbor_obs["upstream"][nb_name] = {
                                "total_queue": q_sum,
                                "total_wait": float(q_sum),
                                "occupancy_avg": 0.0,
                                "phase": ph_str
                            }
                            break

    # Current phase
    idx_phase = inter.current_phase_index - 1
    if 0 <= idx_phase < len(action_space):
        act_str = action_space[idx_phase]
    else:
        act_str = action_space[0]

    # Build observation context
    user_context = {
        "current_observation": {"local_lanes": local_obs},
        "neighbor_observation": neighbor_obs,
        "current_phase": act_str,
        "candidate_actions": action_space,
        "history": []  # Simplified: no history in eval
    }

    user_msg = f"Observation:\n{json.dumps(user_context)}"
    return user_msg

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
        for i in range(args.simulation_time):
            env.step(curr_action)
            if i % 30 == 0:
                for j, inter in enumerate(env.list_intersection):
                    # Build real observation prompt
                    obs_payload = build_observation_prompt(inter, env, action_space)

                    # Send to student endpoint
                    try:
                        resp = requests.post(
                            f"{args.endpoint}/chat/completions",
                            json={
                                "model": "student",
                                "messages": [
                                    {"role": "system", "content": "You are a traffic signal control agent."},
                                    {"role": "user", "content": obs_payload}
                                ]
                            },
                            timeout=10
                        )
                        resp_json = resp.json()["choices"][0]["message"]["content"]
                        action_str = json.loads(resp_json).get("phase2", {}).get("answer", "UNKNOWN")
                        if action_str in action_space:
                            curr_action[j] = action_space.index(action_str)
                        else:
                            curr_action[j] = np.random.randint(0, len(action_space))
                    except Exception as e:
                        print(f"API error for intersection {j}: {e}")
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
