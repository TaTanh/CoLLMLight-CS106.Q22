import argparse
import json
import os
import sys
import random
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import requests
except ImportError:
    requests = None

try:
    from utils.my_utils import count_double_hash_pattern
except ImportError:
    import re
    def count_double_hash_pattern(text: str) -> int:
        return len(re.findall(r'## [^\n]+', text))

SYSTEM_PROMPT = (
    "You are a traffic signal control agent. "
    "You reason about traffic observations and select optimal signal phases."
)

# ── Prompt builders ───────────────────────────────────────────────────────────

def build_atr_prompt_stage3(sample: dict) -> str:
    history_lines = []
    for h in sample.get("history", []):
        t = h.get("timestep", "?")
        act = h.get("action", "?")
        lanes = h.get("local_lanes", {})
        lane_summary = ", ".join(
            f"{k}:q{v['queue']}" for k, v in lanes.items() if v.get("queue", 0) > 0
        )
        history_lines.append(f"  t={t}   phase={act}  {lane_summary}")

    local_obs = sample.get("current_observation", {}).get("local_lanes", {})
    lane_detail = "\n".join(
        f"  {k}: queue={v['queue']} moving={v.get('moving',0)} occ={v.get('occupancy',0):.2f}"
        for k, v in local_obs.items() if v.get("queue", 0) + v.get("moving", 0) > 0
    )

    neighbors = []
    for nb_name, nb_state in sample.get("neighbor_observation", {}).get("upstream", {}).items():
        neighbors.append(
            f"  [upstream] {nb_name}: queue={nb_state['total_queue']} "
            f"occ={nb_state.get('occupancy_avg', 0):.2f} phase={nb_state.get('phase', '?')}"
        )

    is_complex = sample.get("complexity", {}).get("label", "easy") == "Complex"

    atr_user = (
        f"## Historical Observation (last {len(history_lines)} timesteps)\n"
        + "\n".join(history_lines) + "\n\n"
        + f"## Current Observation (t={sample.get('timestep', '?')})  phase={sample.get('current_phase', '?')}\n"
        + lane_detail + "\n\n"
        + "## Neighbor State\n"
        + ("\n".join(neighbors) if neighbors else "  (none)") + "\n\n"
        + "Analyse the traffic situation at this intersection:\n"
        + "- Identify critical lanes (high queue or occupancy)\n"
        + "- Assess upstream/downstream neighbour conditions\n"
        + "- Predict how traffic will evolve over the next 5 timesteps\n"
        + "- Suggest which signal phases would be most effective and why\n"
    )

    if is_complex:
        atr_user += (
            "- Assess how upstream/downstream congestion at neighbouring intersections\n"
            "  will affect your decision and whether releasing or holding traffic will\n"
            "  benefit the network as a whole (spillback prevention and coordination).\n"
        )

    atr_user += "\nOutput free-form reasoning text. Do NOT output JSON.\nDo NOT select a final action yet."
    return atr_user


def build_ra_prompt_stage3(sample: dict, atr_response: str) -> str:
    rollout = sample.get("rollout_results", {})
    signal_rank = sample.get("signal_rank", {})
    complexity_label = sample.get("complexity", {}).get("label", "easy")

    rollout_lines = "\n".join(
        f"  {action}: queue_after_5={r.get('queue_after_5', '?')}  wait={r.get('wait_after_5', '?')}"
        for action, r in rollout.items()
    )

    if signal_rank:
        sorted_actions = sorted(signal_rank.keys(), key=lambda a: signal_rank[a], reverse=True)
    else:
        sorted_actions = sorted(rollout.keys(), key=lambda a: rollout[a].get("queue_after_5", 99))

    priority_lines = "\n".join(
        f"  {i+1}. {a}  queue_after_5={rollout.get(a, {}).get('queue_after_5', '?')}"
        f"  wait_reduction={signal_rank.get(a, 0):+.2f} mins"
        for i, a in enumerate(sorted_actions)
    )

    schema = {
        "phase1": {"answer": complexity_label, "reason": "<max 1 sentence>"},
        "phase2": {
            "traffic_analysis": "<max 2 sentences>",
            "future_state_summary": "<max 1 sentence>",
            "signal_comparison": "<max 1 sentence>",
            "answer": "<one of ['ETWT', 'NTST', 'ELWL', 'NLSL']>"
        }
    }
    if complexity_label == "Complex":
        schema["phase2"]["signal_consequence_prediction"] = {
            a: "<1 sentence>" for a in rollout.keys()
        }

    ra_user = (
        f"## Prior Traffic Analysis (ATR)\n{atr_response}\n\n"
        + f"## Rollout Simulation Results (5-step lookahead)\n{rollout_lines}\n\n"
        + f"## Local Signal Priority (ranked by waiting time reduction)\n{priority_lines}\n\n"
        + "Based on the above reasoning and simulation evidence, select the best\n"
        + "signal phase. Respond with a JSON object matching this schema exactly:\n"
        + json.dumps(schema, indent=2)
    )
    return ra_user


# ── Reward functions ──────────────────────────────────────────────────────────

def calculate_reward_ra_litepp(student_ra_response: str, pseudo_golden: str):
    try:
        data = json.loads(student_ra_response)
        signal = data.get("phase2", {}).get("answer", None)
        reward = 1.0 if signal == pseudo_golden else -1.0
    except Exception:
        reward = -1.0
        signal = None
    return reward, signal


def calculate_reward_atr_litepp(atr_response: str, reward_ra: float, used_atr: int) -> float:
    L_max = 8.0
    L = min(count_double_hash_pattern(atr_response), L_max)
    beta = 0.5
    length_reward = beta * (1.0 - L / L_max)
    effectiveness_reward = (1.0 - beta) * float(used_atr)
    return reward_ra * (length_reward + effectiveness_reward)


# ── API calls ─────────────────────────────────────────────────────────────────

def call_student_atr(prompt: str, endpoint: str, model: str, temperature: float = 0.1) -> str:
    resp = requests.post(
        f"{endpoint}/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
        },
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def call_student_ra(prompt: str, endpoint: str, model: str, temperature: float = 0.1) -> str:
    resp = requests.post(
        f"{endpoint}/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"}
        },
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ── Mock for dry run ──────────────────────────────────────────────────────────

_MOCK_ATR = (
    "## Traffic Analysis\nThe intersection shows moderate congestion on ET and WT lanes.\n"
    "## Prediction\nTraffic expected to increase in next 5 timesteps on NT approach.\n"
    "## Recommendation\nEase ETWT phase to reduce queue buildup."
)
_ACTIONS = ["ETWT", "NTST", "ELWL", "NLSL"]


def mock_student_atr(sample: dict) -> str:
    return _MOCK_ATR


def mock_student_ra(sample: dict) -> str:
    complexity = sample.get("complexity", {}).get("label", "easy")
    golden = sample.get("pseudo_golden_action", "ETWT")
    wrong_action = random.choice([a for a in _ACTIONS if a != golden])
    resp = {
        "phase1": {"answer": complexity, "reason": "mock reason"},
        "phase2": {
            "traffic_analysis": "Mock analysis.",
            "future_state_summary": "Mock future.",
            "signal_comparison": "Mock comparison.",
            "answer": wrong_action
        }
    }
    return json.dumps(resp)


# ── Core batch runner ─────────────────────────────────────────────────────────

def run_batch(samples, endpoint, model, temperature, dry_run=False):
    results = []
    for sample in samples:
        sid = sample.get("sample_id", "?")
        pseudo_golden = sample.get("pseudo_golden_action", "UNKNOWN")

        atr_prompt = build_atr_prompt_stage3(sample)

        try:
            atr_response = mock_student_atr(sample) if dry_run else call_student_atr(
                atr_prompt, endpoint, model, temperature
            )
        except Exception as e:
            print(f"  [WARN] ATR call failed for {sid}: {e}")
            atr_response = ""

        used_atr = 1 if len(atr_response) > 200 else 0
        ra_prompt = build_ra_prompt_stage3(sample, atr_response)

        try:
            ra_response = mock_student_ra(sample) if dry_run else call_student_ra(
                ra_prompt, endpoint, model, temperature
            )
        except Exception as e:
            print(f"  [WARN] RA call failed for {sid}: {e}")
            ra_response = json.dumps({"phase2": {"answer": "UNKNOWN"}})

        reward_ra, student_action = calculate_reward_ra_litepp(ra_response, pseudo_golden)
        reward_atr = calculate_reward_atr_litepp(atr_response, reward_ra, used_atr)

        teacher_ra_obj = sample.get("teacher_response", {})
        if not teacher_ra_obj:
            teacher_ra_obj = {
                "phase1": sample.get("teacher_phase1", {}),
                "phase2": sample.get("teacher_phase2", {})
            }
        teacher_atr = sample.get("atr_output", "")

        results.append({
            "sample": sample,
            "atr_prompt": atr_prompt,
            "ra_prompt": ra_prompt,
            "atr_response": atr_response,
            "ra_response": ra_response,
            "reward_ra": reward_ra,
            "reward_atr": reward_atr,
            "student_action": student_action or "UNKNOWN",
            "pseudo_golden": pseudo_golden,
            "teacher_atr": teacher_atr,
            "teacher_ra": json.dumps(teacher_ra_obj) if isinstance(teacher_ra_obj, dict) else str(teacher_ra_obj),
        })

    return results


def collect_dpo_pairs(results):
    pairs = []
    for r in results:
        if r["reward_ra"] >= 0:
            continue
        sid = r["sample"].get("sample_id", "?")
        pairs.append({
            "sample_id": sid,
            "pair_type": "atr",
            "pseudo_golden": r["pseudo_golden"],
            "reward_ra": r["reward_ra"],
            "reward_atr": r["reward_atr"],
            "student_action": r["student_action"],
            "atr_prompt": r["atr_prompt"],
            "ra_prompt": r["ra_prompt"],
            "teacher_atr": r["teacher_atr"],
            "teacher_ra": r["teacher_ra"],
            "student_atr": r["atr_response"],
            "student_ra": r["ra_response"],
        })
    return pairs


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3a: Build DPO pairs from teacher samples")
    parser.add_argument("--input", type=str,
                        default="data/FinetuneData/litepp/litepp_rco_teacher.jsonl")
    parser.add_argument("--config", type=str, default="config/collmlight_litepp.yaml")
    parser.add_argument("--endpoint", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model", type=str, default="student")
    parser.add_argument("--iterations", type=int, default=0,
                        help="Override num_iterations from config (0 = use config)")
    parser.add_argument("--min_wrong", type=int, default=0,
                        help="Override min_wrong_samples from config (0 = use config)")
    parser.add_argument("--output", type=str,
                        default="data/FinetuneData/litepp/litepp_dpo_pairs.jsonl")
    parser.add_argument("--dry_run", action="store_true",
                        help="Use mock student responses — no API or data required")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    s3cfg = cfg.get("stage3", {})

    iterations = args.iterations or s3cfg.get("num_iterations", 10)
    min_wrong = args.min_wrong or s3cfg.get("min_wrong_samples", 10)
    temperature_init = s3cfg.get("temperature_init", 0.1)
    temperature_refine = s3cfg.get("temperature_refine", 1.0)

    print(f"Loading teacher samples from {args.input}")
    samples = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"  Loaded {len(samples)} samples")

    all_pairs = []

    # Pass 0: deterministic initial run to identify wrong cases
    print(f"\n[Pass 0] Initial eval (temperature={temperature_init}) ...")
    results0 = run_batch(samples, args.endpoint, args.model, temperature_init, dry_run=args.dry_run)
    wrong_results = [r for r in results0 if r["reward_ra"] < 0]
    print(f"  Correct: {len(results0) - len(wrong_results)}/{len(results0)}  Wrong: {len(wrong_results)}")

    pairs0 = collect_dpo_pairs(results0)
    all_pairs.extend(pairs0)
    print(f"  DPO pairs collected: {len(pairs0)}")

    wrong_samples = [r["sample"] for r in wrong_results]

    for iteration in range(1, iterations + 1):
        if len(wrong_samples) < min_wrong:
            print(f"\n[Early stop] Wrong samples ({len(wrong_samples)}) < min_wrong ({min_wrong})")
            break

        print(f"\n[Pass {iteration}/{iterations}] Refining {len(wrong_samples)} wrong samples "
              f"(temperature={temperature_refine}) ...")
        results_iter = run_batch(
            wrong_samples, args.endpoint, args.model, temperature_refine, dry_run=args.dry_run
        )

        pairs_iter = collect_dpo_pairs(results_iter)
        all_pairs.extend(pairs_iter)

        still_wrong = [r for r in results_iter if r["reward_ra"] < 0]
        print(f"  Fixed: {len(wrong_samples) - len(still_wrong)}  "
              f"Still wrong: {len(still_wrong)}  New pairs: {len(pairs_iter)}")

        wrong_samples = [r["sample"] for r in still_wrong]

    print(f"\nTotal DPO pairs collected: {len(all_pairs)}")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
