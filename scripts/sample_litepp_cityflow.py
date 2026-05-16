import os
import json
import yaml
import time
import argparse
import shutil
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import dic_traffic_env_conf
from utils.utils import merge

# Optional CityFlow import (may not be available)
try:
    from utils.cityflow_env import CityFlowEnv
    CITYFLOW_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CITYFLOW_AVAILABLE = False
    CityFlowEnv = None

def copy_cityflow_file(dic_path, dic_traffic_env_conf, path=None):
    if path is None:
        path = dic_path["PATH_TO_WORK_DIRECTORY"]
    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_traffic_env_conf["TRAFFIC_FILE"]),
                os.path.join(path, dic_traffic_env_conf["TRAFFIC_FILE"]))
    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_traffic_env_conf["ROADNET_FILE"]),
                os.path.join(path, dic_traffic_env_conf["ROADNET_FILE"]))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/collmlight_litepp.yaml")
    parser.add_argument("--dataset", type=str, default="synth", choices=["synth", "jinan_3x4", "hangzhou_4x4"])
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--simulation_time", type=int, default=3600)
    parser.add_argument("--policy", type=str, default="random")
    parser.add_argument("--output", type=str, default="data/FinetuneData/litepp/litepp_rco_raw.jsonl")
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        lite_config = yaml.safe_load(f)

    # Setup directories
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    work_dir = f"records/litepp_sample_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(work_dir, exist_ok=True)

    # Configure dataset mapping
    template = ""
    road_net = ""
    traffic_file = ""
    num_row, num_col = 0, 0

    if args.dataset == "synth":
        template = "Synthetic"
        road_net = "4_4"
        traffic_file = "anon_4_4_synthetic_8000.json"
        num_row, num_col = 4, 4
    elif args.dataset == "jinan_3x4":
        template = "Jinan"
        road_net = "3_4"
        traffic_file = "anon_3_4_jinan_real_2000.json"
        num_row, num_col = 3, 4
    elif args.dataset == "hangzhou_4x4":
        template = "Hangzhou"
        road_net = "4_4"
        traffic_file = "anon_4_4_hangzhou_real.json"
        num_row, num_col = 4, 4

    num_intersections = num_row * num_col

    dic_traffic_env_conf_extra = {
        "NUM_AGENTS": num_intersections,
        "NUM_INTERSECTIONS": num_intersections,
        "NUM_ROW": num_row,
        "NUM_COL": num_col,
        "TRAFFIC_FILE": traffic_file,
        "ROADNET_FILE": f"roadnet_{road_net}.json",
        "MODEL_NAME": "LitePPSampler",
        "PHASE": {
            1: [0, 1, 0, 1, 0, 0, 0, 0],
            2: [0, 0, 0, 0, 0, 1, 0, 1],
            3: [1, 0, 1, 0, 0, 0, 0, 0],
            4: [0, 0, 0, 0, 1, 0, 1, 0]
        },
        "PHASE_LIST": ['WT_ET', 'NT_ST', 'WL_EL', 'NL_SL']
    }

    env_conf = merge(dic_traffic_env_conf, dic_traffic_env_conf_extra)

    dic_path = {
        "PATH_TO_DATA": os.path.join("data", template, road_net),
        "PATH_TO_WORK_DIRECTORY": work_dir
    }

    # Prepare cityflow data
    copy_cityflow_file(dic_path, env_conf)

    # Initialize environment
    env = CityFlowEnv(
        path_to_log=work_dir,
        path_to_work_directory=work_dir,
        dic_traffic_env_conf=env_conf,
        dic_path=dic_path
    )
    env.reset()

    # Pre-calculate history size
    history_window = lite_config.get("history_window", 5)
    action_space = lite_config.get("action_space", ["ETWT", "NTST", "ELWL", "NLSL"])
    
    # Store history for every intersection
    # intersection_id -> list of state snapshots (length = history_window)
    intersection_histories = {inter.inter_id: [] for inter in env.list_intersection}
    
    out_f = open(args.output, "w")
    samples_collected = 0

    action = [0] * num_intersections # initially start with phase 0 (index 1)
    
    # Trace for replay
    action_trace_list = []

    for i in range(args.simulation_time):
        if samples_collected >= args.num_samples:
            break

        # Generate dummy actions to step the environment uniformly.
        # Action is an index 0-3
        action_trace_list.append({
            "timestep": i,
            "actions": {env.list_intersection[j].inter_name: action[j] for j in range(num_intersections)}
        })
        
        # Simulate 1 step
        env.step(action)
        
        # We only want to sample at decision steps (e.g. every 30s) or at some interval.
        if i > 50 and i % 30 == 0:
            for j, inter in enumerate(env.list_intersection):
                if samples_collected >= args.num_samples:
                    break

                # Extract observation for `inter`
                # Local observation
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

                total_in = 0
                for lane in inter.list_entering_lanes:
                    q = inter.dic_lane_waiting_vehicle_count_current_step.get(lane, 0)
                    veh_list = inter.dic_lane_vehicle_current_step_in.get(lane, [])
                    moving = max(0, len(veh_list) - q)
                    total_in += len(veh_list)
                    
                    # Approximated mappings (Lane 0=Left, 1=Through, 2=Right)
                    # For simplicity, aggregate by approach (E, W, N, S)
                    road = lane.split('_')[:-1] # e.g. ["road", "x", "y", "approach_idx"]
                    edge = "_".join(road)
                    lane_idx = int(lane.split('_')[-1])
                    
                    # Reverse map edge to approach direction -> we know approach is E,W,N,S
                    approach = None
                    for ap, inc_edge in inter.dic_entering_approach_to_edge.items():
                        if inc_edge == edge:
                            approach = ap
                            break
                    if approach:
                        # Map to our predefined list based on lane_idx
                        if lane_idx == 0:
                            logic = f"{approach}L"
                        elif lane_idx == 1:
                            logic = f"{approach}T"
                        else:
                            continue # Ignore right turns if not modeled, or sum them.
                            
                        if logic in local_obs:
                            local_obs[logic]["queue"] = q
                            local_obs[logic]["moving"] = moving
                            # Average wait time loosely: wait_time per waiting vehicle ~ pseudo wait time. 
                            # Since CityFlow natively tracks total waiting counts per lane at this tick,
                            # we explicitly track it as an approximation. Wait time = queue * seconds (approx under heavy traffic)
                            local_obs[logic]["wait_time"] += q # (Proxy approximation for wait time)
                            # Occupancy = len(veh) * car_length(approx 5m) / lane_length
                            lane_len = inter.lane_length.get(lane, 100)
                            local_obs[logic]["occupancy"] = min(1.0, len(veh_list) * 5.0 / lane_len)

                # Simplify wait proxy (wait time accumulated over steps if it was tracked continuously, 
                # here we just record current queue to represent congestion severity for smoke test)
                
                # neighbor_observation
                neighbor_obs = {"upstream": {}, "downstream": {}}
                node_dict = env.traffic_light_node_dict[inter.inter_name]
                # Gather queue sums for neighbors simply into upstream to satisfy validation
                for nb_k in ["neighbor_ENWS", "neighbor_up_down_stream"]:
                    if nb_k in node_dict:
                        for nb_name in node_dict[nb_k]:
                            if nb_name and nb_name != "null":
                                # Find neighbor intersection
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

                # Resolve current phase: fallback to index math
                idx_phase = inter.current_phase_index - 1
                if 0 <= idx_phase < len(action_space):
                    act_str = action_space[idx_phase]
                else:
                    act_str = action_space[action[j] % len(action_space)]

                # Only write out a sample if history buffer is fully populated (e.g., 5 PREVIOUS frames)
                if len(intersection_histories[inter.inter_id]) == history_window: 
                    sample = {
                        "dataset": args.dataset,
                        "intersection_id": inter.inter_name,
                        "timestep": env.eng.get_current_time(),
                        "current_phase": act_str,
                        "current_observation": {
                            "local_lanes": local_obs
                        },
                        "neighbor_observation": neighbor_obs,
                        "history": intersection_histories[inter.inter_id].copy(), # This has exactly 5 past states
                        "candidate_actions": action_space,
                        "replay": {
                            "seed": 42,
                            "policy": args.policy,
                            "actions_before_timestep": action_trace_list.copy() # All structured actions taken up to now
                        }
                    }
                    out_f.write(json.dumps(sample) + "\n")
                    samples_collected += 1

                # Push history AFTER creating sample
                hist_state = {
                    "timestep": env.eng.get_current_time(),
                    "action": act_str,
                    "local_lanes": local_obs,
                    "neighbor_lanes": neighbor_obs
                }
                intersection_histories[inter.inter_id].append(hist_state)
                # Keep window size
                if len(intersection_histories[inter.inter_id]) > history_window:
                    intersection_histories[inter.inter_id].pop(0)

            # Decide next actions AFTER sampling state
            if args.policy == "random":
                action = [np.random.randint(0, 4) for _ in range(num_intersections)]
            
            for j, inter in enumerate(env.list_intersection):
                inter.set_signal(action[j], "set", yellow_time=5, path_to_log=work_dir)
                
    out_f.close()
    print(f"Sampling done. Wrote {samples_collected} samples to {args.output}")

if __name__ == "__main__":
    main()