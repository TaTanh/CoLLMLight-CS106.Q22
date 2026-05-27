import argparse
import csv
import json
import os
import sys
import yaml
import numpy as np
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.cityflow_env import CityFlowEnv
    from utils.config import dic_traffic_env_conf
    from utils.utils import merge
    from scripts.sample_litepp_cityflow import copy_cityflow_file
    CITYFLOW_AVAILABLE = True
except Exception:
    CITYFLOW_AVAILABLE = False

SYSTEM_PROMPT = "You are a traffic signal control agent. You reason about traffic observations and select optimal signal phases."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/collmlight_litepp.yaml")
    parser.add_argument("--dataset", type=str, choices=["synth", "jinan_3x4", "hangzhou_4x4"], required=True)
    parser.add_argument("--endpoint", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model", type=str, default="student")
    parser.add_argument("--simulation_time", type=int, default=3600)
    parser.add_argument("--output", type=str, default="outputs/litepp_eval_results.csv")
    parser.add_argument("--help_only", action="store_true")
    return parser.parse_args()


def build_observation_context(inter, env, action_space):
    """Extract live intersection state into a structured context dict."""
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

    neighbor_obs = {"upstream": {}, "downstream": {}}
    node_dict = env.traffic_light_node_dict[inter.inter_name]
    for nb_k in ["neighbor_ENWS", "neighbor_up_down_stream"]:
        if nb_k in node_dict:
            for nb_name in node_dict[nb_k]:
                if nb_name and nb_name != "null":
                    for adj_inter in env.list_intersection:
                        if adj_inter.inter_name == nb_name:
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

    idx_phase = inter.current_phase_index - 1
    current_phase = action_space[idx_phase] if 0 <= idx_phase < len(action_space) else action_space[0]

    return {
        "current_observation": {"local_lanes": local_obs},
        "neighbor_observation": neighbor_obs,
        "current_phase": current_phase,
        "candidate_actions": action_space,
        "history": [],
    }


def build_atr_prompt(obs_ctx: dict) -> str:
    local_obs = obs_ctx["current_observation"]["local_lanes"]
    lane_detail = "\n".join(
        f"  {k}: queue={v['queue']} moving={v.get('moving',0)} occ={v.get('occupancy',0):.2f}"
        for k, v in local_obs.items() if v.get("queue", 0) + v.get("moving", 0) > 0
    ) or "  (all lanes empty)"

    neighbors = [
        f"  [upstream] {nb}: queue={s['total_queue']} occ={s.get('occupancy_avg',0):.2f} phase={s.get('phase','?')}"
        for nb, s in obs_ctx.get("neighbor_observation", {}).get("upstream", {}).items()
    ]

    return (
        f"## Current Observation  phase={obs_ctx['current_phase']}\n"
        + lane_detail + "\n\n"
        + "## Neighbor State\n"
        + ("\n".join(neighbors) if neighbors else "  (none)") + "\n\n"
        + "Analyse the traffic situation at this intersection:\n"
        + "- Identify critical lanes (high queue or occupancy)\n"
        + "- Assess upstream/downstream neighbour conditions\n"
        + "- Predict how traffic will evolve over the next 5 timesteps\n"
        + "- Suggest which signal phases would be most effective and why\n"
        + "\nOutput free-form reasoning text. Do NOT output JSON.\nDo NOT select a final action yet."
    )


def build_ra_prompt(obs_ctx: dict, atr_text: str) -> str:
    action_space = obs_ctx.get("candidate_actions", ["ETWT", "NTST", "ELWL", "NLSL"])
    schema = {
        "phase1": {"answer": "easy", "reason": "<max 1 sentence>"},
        "phase2": {
            "traffic_analysis": "<max 2 sentences>",
            "future_state_summary": "<max 1 sentence>",
            "signal_comparison": "<max 1 sentence>",
            "answer": f"<one of {action_space}>"
        }
    }
    return (
        f"## Prior Traffic Analysis (ATR)\n{atr_text}\n\n"
        + "Based on the above reasoning, select the best signal phase.\n"
        + "Respond with a JSON object matching this schema exactly:\n"
        + json.dumps(schema, indent=2)
    )


def parse_cityflow_metrics(work_dir: str) -> dict:
    """Parse vehicle_inter_*.csv logs to compute ATT and AWT."""
    travel_times, wait_times = [], []
    for fname in os.listdir(work_dir):
        if fname.startswith("vehicle_inter") and fname.endswith(".csv"):
            with open(os.path.join(work_dir, fname), newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        tt = float(row.get("travel_time", 0) or 0)
                        wt = float(row.get("waiting_time", 0) or 0)
                        if tt > 0:
                            travel_times.append(tt)
                        if wt > 0:
                            wait_times.append(wt)
                    except ValueError:
                        continue
    return {
        "ATT": round(sum(travel_times) / len(travel_times), 2) if travel_times else 0.0,
        "AWT": round(sum(wait_times) / len(wait_times), 2) if wait_times else 0.0,
        "n_vehicles": len(travel_times),
    }


def main():
    args = parse_args()
    if args.help_only:
        print("Lite++ Evaluation Module.")
        return
    if not CITYFLOW_AVAILABLE:
        print("ERROR: CityFlow is not installed. Evaluation requires CityFlow.")
        print("Install: pip install cityflow  (or build from source)")
        sys.exit(1)

    with open(args.config, "r") as f:
        lite_config = yaml.safe_load(f)

    work_dir = f"records/litepp_eval_{args.dataset}"
    os.makedirs(work_dir, exist_ok=True)

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

    env = CityFlowEnv(
        path_to_log=work_dir, path_to_work_directory=work_dir,
        dic_traffic_env_conf=env_conf, dic_path=dic_path
    )
    env.reset()

    action_space = lite_config.get("action_space", ["ETWT", "NTST", "ELWL", "NLSL"])
    curr_action = [0] * num_inters

    print(f"Starting CityFlow evaluation for {args.dataset} ({args.simulation_time}s)")
    try:
        for i in range(args.simulation_time):
            env.step(curr_action)
            if i % 30 == 0:
                for j, inter in enumerate(env.list_intersection):
                    obs_ctx = build_observation_context(inter, env, action_space)
                    atr_prompt = build_atr_prompt(obs_ctx)

                    # ATR call (Call 1)
                    try:
                        atr_resp = requests.post(
                            f"{args.endpoint}/chat/completions",
                            json={
                                "model": args.model,
                                "messages": [
                                    {"role": "system", "content": SYSTEM_PROMPT},
                                    {"role": "user", "content": atr_prompt}
                                ],
                                "temperature": 0.1,
                            },
                            timeout=15
                        )
                        atr_text = atr_resp.json()["choices"][0]["message"]["content"]
                    except Exception as e:
                        print(f"ATR error intersection {j} step {i}: {e}")
                        atr_text = "Unable to analyse traffic at this time."

                    # RA call (Call 2)
                    ra_prompt = build_ra_prompt(obs_ctx, atr_text)
                    try:
                        ra_resp = requests.post(
                            f"{args.endpoint}/chat/completions",
                            json={
                                "model": args.model,
                                "messages": [
                                    {"role": "system", "content": SYSTEM_PROMPT},
                                    {"role": "user", "content": ra_prompt}
                                ],
                                "temperature": 0.1,
                                "response_format": {"type": "json_object"}
                            },
                            timeout=15
                        )
                        ra_content = ra_resp.json()["choices"][0]["message"]["content"]
                        action_str = json.loads(ra_content).get("phase2", {}).get("answer", "UNKNOWN")
                        if action_str in action_space:
                            curr_action[j] = action_space.index(action_str)
                        else:
                            curr_action[j] = np.random.randint(0, len(action_space))
                    except Exception as e:
                        print(f"RA error intersection {j} step {i}: {e}")
                        curr_action[j] = np.random.randint(0, len(action_space))

                    inter.set_signal(curr_action[j], "set", yellow_time=3, path_to_log=work_dir)

    except Exception as e:
        print(f"Simulation error: {e}")

    metrics = parse_cityflow_metrics(work_dir)
    print(f"Results — ATT: {metrics['ATT']}s  AWT: {metrics['AWT']}s  Vehicles: {metrics['n_vehicles']}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_header = not os.path.exists(args.output)
    with open(args.output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "model", "ATT", "AWT", "n_vehicles"])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "dataset": args.dataset,
            "model": args.model,
            "ATT": metrics["ATT"],
            "AWT": metrics["AWT"],
            "n_vehicles": metrics["n_vehicles"],
        })
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
