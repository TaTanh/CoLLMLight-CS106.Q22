import os
import json
import yaml
import time
import argparse
import shutil
import numpy as np
import sys
from copy import deepcopy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from utils.config import dic_traffic_env_conf
from utils.utils import merge

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


# ---------------------------------------------------------------------------
# Environment snapshot / restore  (pattern from original FTSample.py)
# ---------------------------------------------------------------------------

def save_env_state(env):
    """
    Snapshot full simulator state at current timestep t.
    Uses eng.snapshot() for C++ state + deepcopy for Python-side state.
    Live env remains fully usable after this call.
    """
    eng = env.eng
    snapshot = eng.snapshot()
    # Disconnect C++ references before deepcopy to avoid pickle failure
    env.eng = None
    for inter in env.list_intersection:
        inter.eng = None
    env_copy = deepcopy(env)
    # Reconnect live env immediately
    env.eng = eng
    for inter in env.list_intersection:
        inter.eng = eng
    return {"snapshot": snapshot, "env_copy": env_copy, "eng": eng}


def load_env_state(env, state):
    """
    Restore simulator to a previously saved state (in-place update of env).
    Updates both C++ engine and Python-side attributes.
    """
    eng = state["eng"]
    eng.load(state["snapshot"])
    env_copy = deepcopy(state["env_copy"])
    env.list_intersection = env_copy.list_intersection
    env.waiting_vehicle_list = env_copy.waiting_vehicle_list
    if hasattr(env_copy, "system_states"):
        env.system_states = env_copy.system_states
    env.traffic_light_node_dict = env_copy.traffic_light_node_dict
    # Rebuild intersection_dict so references match the restored list
    env.intersection_dict = {inter.inter_name: inter for inter in env.list_intersection}
    env.eng = eng
    for inter in env.list_intersection:
        inter.eng = eng


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------

def maxpressure_action(local_obs, action_space):
    """
    MaxPressure: choose phase serving lanes with highest total queue.
    Each entry in action_space (e.g. "ETWT") serves two lanes (ET + WT).
    Ties broken randomly to avoid always picking the first phase.
    """
    phase_pressure = []
    for phase_str in action_space:
        lane_a = phase_str[:2]
        lane_b = phase_str[2:]
        q_a = local_obs.get(lane_a, {}).get("queue", 0)
        q_b = local_obs.get(lane_b, {}).get("queue", 0)
        phase_pressure.append(q_a + q_b)
    max_p = max(phase_pressure)
    candidates = [i for i, p in enumerate(phase_pressure) if p == max_p]
    return int(np.random.choice(candidates))


# ---------------------------------------------------------------------------
# Rollout evaluation
# ---------------------------------------------------------------------------

def evaluate_rollout_for_inter(env, intersection_name, action_idx, rollout_horizon):
    """
    Apply action_idx to all intersections, step rollout_horizon times,
    accumulating queue across all steps for the target intersection only.
    Matches paper default reward_type='avg_ql': sum(ql over each step).
    """
    num_inters = len(env.list_intersection)
    for inter in env.list_intersection:
        inter.set_signal(action_idx, "set", yellow_time=3,
                         path_to_log=env.path_to_work_directory)
    total_queue = 0
    total_wait = 0.0
    for _ in range(rollout_horizon):
        env.step([action_idx] * num_inters)
        for inter in env.list_intersection:
            if inter.inter_name == intersection_name:
                for lane in inter.list_entering_lanes:
                    q = inter.dic_lane_waiting_vehicle_count_current_step.get(lane, 0)
                    total_queue += q
                    total_wait += float(q)
                break
    return total_queue, total_wait


def synthetic_rollout(local_obs, action_space):
    """Deterministic fallback when CityFlow is unavailable."""
    current_queue = sum(lane.get("queue", 0) for lane in local_obs.values())
    results = {}
    for a_idx, action_str in enumerate(action_space):
        q5 = max(0, current_queue - 5) if a_idx == 0 else current_queue + (a_idx - 1) * 2
        results[action_str] = {
            "queue_after_5": q5,
            "wait_after_5": float(q5),
            "future_state_summary": f"Synthetic rollout: queue {q5}"
        }
    return results


# ---------------------------------------------------------------------------
# Observation extraction
# ---------------------------------------------------------------------------

def extract_local_obs(inter):
    """Extract per-lane observations from the current intersection state."""
    local_obs = {
        d: {"queue": 0, "moving": 0, "wait_time": 0.0, "occupancy": 0.0}
        for d in ["ET", "WT", "EL", "WL", "NT", "ST", "NL", "SL"]
    }
    for lane in inter.list_entering_lanes:
        q = inter.dic_lane_waiting_vehicle_count_current_step.get(lane, 0)
        veh_list = inter.dic_lane_vehicle_current_step_in.get(lane, [])
        moving = max(0, len(veh_list) - q)
        road = lane.split("_")[:-1]
        edge = "_".join(road)
        lane_idx = int(lane.split("_")[-1])
        approach = None
        for ap, inc_edge in inter.dic_entering_approach_to_edge.items():
            if inc_edge == edge:
                approach = ap
                break
        if approach is None:
            continue
        if lane_idx == 0:
            logic = f"{approach}L"
        elif lane_idx == 1:
            logic = f"{approach}T"
        else:
            continue
        if logic in local_obs:
            local_obs[logic]["queue"] = q
            local_obs[logic]["moving"] = moving
            local_obs[logic]["wait_time"] += float(q)
            lane_len = inter.lane_length.get(lane, 100)
            local_obs[logic]["occupancy"] = min(1.0, len(veh_list) * 5.0 / lane_len)
    return local_obs


def apply_delta_tracking(local_obs, history):
    """
    Add queue_change and occupancy_change to each lane in local_obs.
    Compares against the most recent entry in history (S1-1 fix).
    """
    prev_local = history[-1].get("local_lanes", {}) if history else {}
    for lane_key, lane_data in local_obs.items():
        prev = prev_local.get(lane_key, {})
        lane_data["queue_change"]     = lane_data["queue"]    - prev.get("queue",    0)
        lane_data["occupancy_change"] = round(
            lane_data["occupancy"] - prev.get("occupancy", 0.0), 3)


def extract_neighbor_obs(inter, env, action_space):
    """Extract queue/phase summary from neighboring intersections."""
    neighbor_obs = {"upstream": {}, "downstream": {}}
    node_dict = env.traffic_light_node_dict[inter.inter_name]
    for nb_key in ["neighbor_ENWS", "neighbor_up_down_stream"]:
        if nb_key not in node_dict:
            continue
        for nb_name in node_dict[nb_key]:
            if not nb_name or nb_name == "null":
                continue
            for adj_inter in env.list_intersection:
                if adj_inter.inter_name != nb_name:
                    continue
                q_sum = sum(
                    adj_inter.dic_lane_waiting_vehicle_count_current_step.get(l, 0)
                    for l in adj_inter.list_entering_lanes
                )
                idx = adj_inter.current_phase_index - 1
                ph_str = action_space[idx] if 0 <= idx < len(action_space) else "UNKNOWN"
                _veh = [len(adj_inter.dic_lane_vehicle_current_step_in.get(l, []))
                        for l in adj_inter.list_entering_lanes]
                _cap = [adj_inter.lane_length.get(l, 100)
                        for l in adj_inter.list_entering_lanes]
                _occ = [min(1.0, v * 5.0 / c) for v, c in zip(_veh, _cap)]
                neighbor_obs["upstream"][nb_name] = {
                    "total_queue": q_sum,
                    "total_wait": float(q_sum),
                    "occupancy_avg": round(sum(_occ) / max(len(_occ), 1), 3),
                    "phase": ph_str
                }
                break
    return neighbor_obs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/collmlight_litepp.yaml")
    parser.add_argument("--dataset", type=str, default="synth",
                        choices=["synth", "jinan_3x4", "hangzhou_4x4"])
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--simulation_time", type=int, default=3600)
    parser.add_argument("--output", type=str,
                        default="data/FinetuneData/litepp/litepp_rco_raw.jsonl")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    with open(args.config, "r") as f:
        lite_config = yaml.safe_load(f)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    work_dir = f"records/litepp_sample_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(work_dir, exist_ok=True)

    dataset_map = {
        "synth":        ("Synthetic", "4_4", "anon_4_4_synthetic_8000.json", 4, 4),
        "jinan_3x4":    ("Jinan",     "3_4", "anon_3_4_jinan_real_2000.json", 3, 4),
        "hangzhou_4x4": ("Hangzhou",  "4_4", "anon_4_4_hangzhou_real.json",  4, 4),
    }
    template, road_net, traffic_file, num_row, num_col = dataset_map[args.dataset]
    num_intersections = num_row * num_col

    history_window  = lite_config.get("history_window", 5)
    rollout_horizon = lite_config.get("rollout_horizon", 5)
    action_space    = lite_config.get("action_space", ["ETWT", "NTST", "ELWL", "NLSL"])
    data_root       = os.path.join(PROJECT_ROOT,
                                   lite_config.get("data_root", "data"))

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
            4: [0, 0, 0, 0, 1, 0, 1, 0],
        },
        "PHASE_LIST": ["WT_ET", "NT_ST", "WL_EL", "NL_SL"],
    }
    env_conf = merge(dic_traffic_env_conf, dic_traffic_env_conf_extra)
    dic_path = {
        "PATH_TO_DATA": os.path.join(data_root, template, road_net),
        "PATH_TO_WORK_DIRECTORY": work_dir,
    }

    copy_cityflow_file(dic_path, env_conf)

    if not CITYFLOW_AVAILABLE:
        print("[WARNING] CityFlow not available — using synthetic rollout fallback")
        print("(Install cityflow for real simulation)\n")

    # -----------------------------------------------------------------------
    # Environment init
    # -----------------------------------------------------------------------
    if CITYFLOW_AVAILABLE:
        env = CityFlowEnv(
            path_to_log=work_dir,
            path_to_work_directory=work_dir,
            dic_traffic_env_conf=env_conf,
            dic_path=dic_path,
        )
        env.reset()
        has_snapshot = hasattr(env.eng, "snapshot")
        if not has_snapshot:
            print("[WARNING] eng.snapshot() not available — falling back to synthetic rollout")
        intersection_histories = {inter.inter_id: [] for inter in env.list_intersection}
    else:
        env = None
        has_snapshot = False
        intersection_histories = {
            (i, j): []
            for i in range(1, num_row + 1)
            for j in range(1, num_col + 1)
        }

    out_f = open(args.output, "w")
    samples_collected = 0
    action = [0] * num_intersections

    # -----------------------------------------------------------------------
    # Simulation loop
    # -----------------------------------------------------------------------
    for i in range(args.simulation_time):
        if samples_collected >= args.num_samples:
            break

        # Advance simulation one step
        if env:
            env.step(action)

        # Decision step: every 30s, after warm-up
        if i <= 50 or i % 30 != 0:
            continue

        # -- No CityFlow: emit synthetic samples and continue ---------------
        if not env:
            for j in range(num_intersections):
                if samples_collected >= args.num_samples:
                    break
                local_obs = {
                    k: {"queue": 0, "moving": 0, "wait_time": 0.0, "occupancy": 0.0}
                    for k in ["ET", "WT", "EL", "WL", "NT", "ST", "NL", "SL"]
                }
                rollout_results = synthetic_rollout(local_obs, action_space)
                pseudo_golden = min(action_space,
                                    key=lambda a: rollout_results[a]["queue_after_5"])
                signal_rank = {act: 0.0 for act in action_space}
                row = (j // num_col) + 1
                col = (j % num_col) + 1
                inter_id_str = f"intersection_{row}_{col}"
                sample = {
                    "sample_id": f"{args.dataset}_{inter_id_str}_t{i}",
                    "dataset": args.dataset,
                    "intersection_id": inter_id_str,
                    "timestep": i,
                    "current_phase": action_space[0],
                    "current_observation": {"local_lanes": local_obs},
                    "neighbor_observation": {"upstream": {}, "downstream": {}},
                    "history": [],
                    "candidate_actions": action_space,
                    "rollout_results": rollout_results,
                    "pseudo_golden_action": pseudo_golden,
                    "signal_rank": signal_rank,
                }
                out_f.write(json.dumps(sample) + "\n")
                samples_collected += 1
            continue

        # -- CityFlow available: extract observations ------------------------
        all_local_obs   = {}
        all_neighbor_obs = {}
        for inter in env.list_intersection:
            all_local_obs[inter.inter_id]    = extract_local_obs(inter)
            all_neighbor_obs[inter.inter_id] = extract_neighbor_obs(inter, env, action_space)

        # Delta tracking: compare each lane against previous timestep (S1-1)
        for inter in env.list_intersection:
            apply_delta_tracking(all_local_obs[inter.inter_id],
                                 intersection_histories[inter.inter_id])

        # Save state once for all per-intersection rollouts at this timestep
        state_snapshot = save_env_state(env) if has_snapshot else None

        # -- Per-intersection: inline rollout + sample collection -----------
        for j in range(num_intersections):
            if samples_collected >= args.num_samples:
                break

            inter      = env.list_intersection[j]
            inter_id   = inter.inter_id
            inter_name = inter.inter_name
            local_obs  = all_local_obs[inter_id]
            neighbor_obs = all_neighbor_obs[inter_id]

            # Resolve current phase string
            idx_phase = inter.current_phase_index - 1
            act_str = (action_space[idx_phase]
                       if 0 <= idx_phase < len(action_space)
                       else action_space[action[j] % len(action_space)])

            # Only collect samples once history window is full
            if len(intersection_histories[inter_id]) == history_window:
                if state_snapshot is not None:
                    # Inline rollout: try all 4 actions from state at t
                    rollout_results = {}
                    for a_idx, action_str in enumerate(action_space):
                        load_env_state(env, state_snapshot)
                        q5, w5 = evaluate_rollout_for_inter(
                            env, inter_name, a_idx, rollout_horizon)
                        rollout_results[action_str] = {
                            "queue_after_5": q5,
                            "wait_after_5":  w5,
                            "future_state_summary": f"Queue goes to {q5}",
                        }
                    # Restore to t after all 4 rollouts
                    load_env_state(env, state_snapshot)
                    inter = env.list_intersection[j]  # fresh ref after restore
                else:
                    rollout_results = synthetic_rollout(local_obs, action_space)

                pseudo_golden = min(
                    action_space,
                    key=lambda a: (rollout_results[a]["queue_after_5"],
                                   rollout_results[a]["wait_after_5"]),
                )

                # Signal rank: waiting time reduction per action (S1-2)
                # wait_time in local_obs is queue-count proxy (same units as wait_after_5)
                current_total_wait = sum(v["wait_time"] for v in local_obs.values())
                signal_rank = {
                    act: round((current_total_wait - res["wait_after_5"]) / 60.0, 2)
                    for act, res in rollout_results.items()
                }

                ts = int(env.eng.get_current_time())
                sample = {
                    "sample_id": f"{args.dataset}_{inter_name}_t{ts}",
                    "dataset": args.dataset,
                    "intersection_id": inter_name,
                    "timestep": ts,
                    "current_phase": act_str,
                    "current_observation": {"local_lanes": local_obs},
                    "neighbor_observation": neighbor_obs,
                    "history": intersection_histories[inter_id].copy(),
                    "candidate_actions": action_space,
                    "rollout_results": rollout_results,
                    "pseudo_golden_action": pseudo_golden,
                    "signal_rank": signal_rank,
                }
                out_f.write(json.dumps(sample) + "\n")
                samples_collected += 1

            # Always push current state to history (after writing sample)
            intersection_histories[inter_id].append({
                "timestep": env.eng.get_current_time(),
                "action": act_str,
                "local_lanes": local_obs,
                "neighbor_lanes": neighbor_obs,
            })
            if len(intersection_histories[inter_id]) > history_window:
                intersection_histories[inter_id].pop(0)

        # -- MaxPressure: choose actions for next simulation step -----------
        # State is at t after the final load_env_state (or never modified if no snapshot)
        new_action = [
            maxpressure_action(all_local_obs[inter.inter_id], action_space)
            for inter in env.list_intersection
        ]
        action = new_action
        for j, inter in enumerate(env.list_intersection):
            inter.set_signal(action[j], "set", yellow_time=3, path_to_log=work_dir)

    out_f.close()
    print(f"Sampling done. Wrote {samples_collected} samples to {args.output}")


if __name__ == "__main__":
    main()
