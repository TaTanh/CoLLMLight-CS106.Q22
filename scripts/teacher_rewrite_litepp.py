"""
Stage 2b — Teacher Reasoning Chain (ATR → RA)

Two-call chain matching the paper's reasoning_tuning_data_synth.py:

  Call 1  ATR  (Advance Traffic Reasoning)
          Input : observation + history  — no schema, no action hint
          Output: free-form traffic analysis text

  Call 2  RA   (Reactive Action)
          Input : ATR output + rollout results + local signal priority
          Output: JSON  { phase1: {answer, reason}, phase2: {…, answer} }

The two-call structure forces genuine chain-of-thought: the model must reason
before seeing the final-answer schema, preventing backward rationalisation.
"""
import argparse
import json
import os


# ---------------------------------------------------------------------------
# Prompt formatting helpers
# ---------------------------------------------------------------------------

def _format_history(history):
    if not history:
        return "(no historical data)"
    lines = []
    for h in history:
        active_lanes = ", ".join(
            f"{k}:q{v['queue']}"
            for k, v in h.get("local_lanes", {}).items()
            if v.get("queue", 0) > 0
        )
        lines.append(
            f"  t={h.get('timestep', '?')}  phase={h.get('action', '?')}  "
            + (active_lanes or "all clear")
        )
    return "\n".join(lines)


def _format_current_obs(current_obs):
    local = current_obs.get("local_lanes", {})
    lines = [
        f"  {lane}: queue={v['queue']} moving={v['moving']} "
        f"occ={v.get('occupancy', 0):.2f}"
        for lane, v in local.items()
        if v.get("queue", 0) > 0 or v.get("moving", 0) > 0
    ]
    return "\n".join(lines) if lines else "  (all lanes clear)"


def _format_neighbor_obs(neighbor_obs):
    lines = []
    for direction in ("upstream", "downstream"):
        for nb_name, nb in neighbor_obs.get(direction, {}).items():
            lines.append(
                f"  [{direction}] {nb_name}: "
                f"queue={nb.get('total_queue', 0)} phase={nb.get('phase', '?')}"
            )
    return "\n".join(lines) if lines else "  (no neighbor data)"


def _format_rollout(rollout_results):
    lines = [
        f"  {action}: queue_after_5={res['queue_after_5']}  "
        f"wait={res['wait_after_5']:.1f}"
        for action, res in rollout_results.items()
    ]
    return "\n".join(lines)


def _format_priority(signal_priority, rollout_results, signal_rank=None):
    """
    Uses signal_rank (waiting time reduction in minutes) when available,
    otherwise falls back to projected queue count.
    """
    lines = []
    for rank, action in enumerate(signal_priority, 1):
        q = rollout_results[action]["queue_after_5"]
        if signal_rank and action in signal_rank:
            wr = signal_rank[action]
            lines.append(
                f"  {rank}. {action}  queue_after_5={q}  "
                f"wait_reduction={wr:+.2f} mins"
            )
        else:
            lines.append(f"  {rank}. {action}  projected_queue={q}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ATR prompt  (Call 1 — free text, no JSON schema)
# ---------------------------------------------------------------------------

ATR_SYSTEM = (
    "You are an expert traffic signal control agent managing a four-way intersection "
    "within a larger road network. Your objective is to analyse traffic conditions "
    "and reason about future states to support an optimal signal decision."
)


def build_atr_prompt(sample):
    """
    Adaptive ATR prompt: Complex cases get an extra instruction to reason
    about network-wide coordination (spillback prevention).
    """
    c_label = sample.get("complexity", {}).get("label", "easy")

    base_instructions = (
        "- Identify critical lanes (high queue or occupancy)\n"
        "- Assess upstream/downstream neighbour conditions\n"
        "- Predict how traffic will evolve over the next 5 timesteps\n"
        "- Suggest which signal phases would be most effective and why"
    )

    if c_label == "Complex":
        extra = (
            "\n- Assess how upstream/downstream congestion at neighbouring intersections "
            "will affect your decision and whether releasing or holding traffic will "
            "benefit the network as a whole (spillback prevention and coordination)."
        )
    else:
        extra = ""

    return (
        f"## Historical Observation (last {len(sample['history'])} timesteps)\n"
        f"{_format_history(sample['history'])}\n\n"
        f"## Current Observation (t={sample['timestep']})  "
        f"phase={sample['current_phase']}\n"
        f"{_format_current_obs(sample['current_observation'])}\n\n"
        f"## Neighbor State\n"
        f"{_format_neighbor_obs(sample.get('neighbor_observation', {}))}\n\n"
        "Analyse the traffic situation at this intersection:\n"
        f"{base_instructions}{extra}\n\n"
        "Output free-form reasoning text. Do NOT output JSON. "
        "Do NOT select a final action yet."
    )


# ---------------------------------------------------------------------------
# RA prompt  (Call 2 — JSON output)
# ---------------------------------------------------------------------------

def build_ra_prompt(sample, atr_output, signal_priority):
    c_label     = sample.get("complexity", {}).get("label", "easy")
    candidates  = sample.get("candidate_actions", ["ETWT", "NTST", "ELWL", "NLSL"])
    rollout     = sample.get("rollout_results", {})
    signal_rank = sample.get("signal_rank", None)

    schema_tail = ""
    if c_label == "Complex":
        preds = {a: "<1-sentence queue prediction>" for a in candidates}
        schema_tail = (
            ',\n    "signal_consequence_prediction": '
            + json.dumps(preds, indent=6)
        )

    schema = (
        "{\n"
        '  "phase1": {\n'
        f'    "answer": "{c_label}",\n'
        '    "reason": "<max 1 sentence explaining complexity level>"\n'
        "  },\n"
        '  "phase2": {\n'
        '    "traffic_analysis": "<max 2 sentences on local + neighbour conditions>",\n'
        '    "future_state_summary": "<max 1 sentence on expected rollout outcome>",\n'
        '    "signal_comparison": "<max 1 sentence comparing candidate actions>"'
        + schema_tail
        + ',\n'
        f'    "answer": "<one of {candidates}>"\n'
        "  }\n"
        "}"
    )

    return (
        "## Prior Traffic Analysis (ATR)\n"
        f"{atr_output}\n\n"
        "## Rollout Simulation Results (5-step lookahead)\n"
        f"{_format_rollout(rollout)}\n\n"
        "## Local Signal Priority (ranked by lowest projected queue)\n"
        f"{_format_priority(signal_priority, rollout, signal_rank)}\n\n"
        "Based on the above reasoning and simulation evidence, select the best "
        "signal phase. Respond with a JSON object matching this schema exactly "
        "(keep all text fields concise):\n"
        f"{schema}"
    )


# ---------------------------------------------------------------------------
# Schema validation / repair
# ---------------------------------------------------------------------------

def fallback_reasoning(sample):
    c_label  = sample.get("complexity", {}).get("label", "easy")
    g_action = sample.get("pseudo_golden_action", "UNKNOWN")
    resp = {
        "phase1": {
            "answer": c_label,
            "reason": "Traffic conditions show routine flow based on fallback evaluation.",
        },
        "phase2": {
            "traffic_analysis": "Analysed intersection queues and wait times.",
            "future_state_summary": "Selecting the golden phase minimises total network delay.",
            "signal_comparison": "Compared candidate phases against current backlog.",
            "answer": g_action,
        },
    }
    if c_label == "Complex":
        resp["phase2"]["signal_consequence_prediction"] = {
            a: "Estimated queue change minimal."
            for a in sample.get("candidate_actions", ["ETWT", "NTST", "ELWL", "NLSL"])
        }
    return resp


def repair_schema(parsed, sample):
    c_label  = sample.get("complexity", {}).get("label", "easy")
    g_action = sample.get("pseudo_golden_action", "UNKNOWN")
    repaired = False

    if not isinstance(parsed.get("phase1"), dict):
        parsed["phase1"] = {}
        repaired = True
    if not isinstance(parsed.get("phase2"), dict):
        parsed["phase2"] = {}
        repaired = True

    p1, p2 = parsed["phase1"], parsed["phase2"]

    if "reason" not in p1:
        p1["reason"] = "Traffic conditions evaluated."
        repaired = True
    for field in ("traffic_analysis", "future_state_summary", "signal_comparison"):
        if field not in p2:
            p2[field] = "See rollout results."
            repaired = True

    if c_label == "Complex" and not isinstance(p2.get("signal_consequence_prediction"), dict):
        p2["signal_consequence_prediction"] = {
            a: "Estimated queue change minimal."
            for a in sample.get("candidate_actions", ["ETWT", "NTST", "ELWL", "NLSL"])
        }
        repaired = True

    # Force strict alignment with pseudo-golden
    p1["answer"] = c_label
    p2["answer"] = g_action

    final = {
        "phase1": {"answer": p1["answer"], "reason": p1["reason"]},
        "phase2": {
            "traffic_analysis":     p2["traffic_analysis"],
            "future_state_summary": p2["future_state_summary"],
            "signal_comparison":    p2["signal_comparison"],
            "answer":               p2["answer"],
        },
    }
    if c_label == "Complex":
        final["phase2"]["signal_consequence_prediction"] = p2["signal_consequence_prediction"]
    return final, repaired


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",             type=str, default="config/collmlight_litepp.yaml")
    parser.add_argument("--input",              type=str, default="data/FinetuneData/litepp/litepp_rco_rollout.jsonl")
    parser.add_argument("--output",             type=str, default="data/FinetuneData/litepp/litepp_rco_teacher.jsonl")
    parser.add_argument("--model_name_or_path", type=str, default="gpt-4o-mini")
    parser.add_argument("--dry_run",            action="store_true")
    parser.add_argument("--max_samples",        type=int, default=-1)
    parser.add_argument("--save_raw_response",  action="store_true")
    parser.add_argument("--atr_max_tokens",     type=int, default=500,
                        help="Max tokens for ATR call (free-text reasoning)")
    parser.add_argument("--ra_max_tokens",      type=int, default=400,
                        help="Max tokens for RA call (JSON decision)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    try:
        from openai import OpenAI
        has_openai = True
    except ImportError:
        has_openai = False

    client = None
    if not args.dry_run and has_openai:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "dummy"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )

    success_count  = 0
    fallback_count = 0

    with open(args.input, "r") as fin, open(args.output, "w") as fout:
        for idx, line in enumerate(fin):
            if not line.strip():
                continue
            if args.max_samples > 0 and idx >= args.max_samples:
                break

            sample = json.loads(line)

            if args.dry_run or not has_openai or client is None:
                sample["atr_output"]       = "(dry-run — no ATR call)"
                sample["teacher_response"] = fallback_reasoning(sample)
                sample["teacher_fallback"] = True
                fallback_count += 1
            else:
                try:
                    # ----------------------------------------------------------
                    # Call 1 — ATR: free-form traffic reasoning
                    # ----------------------------------------------------------
                    atr_resp = client.chat.completions.create(
                        model=args.model_name_or_path,
                        max_tokens=args.atr_max_tokens,
                        messages=[
                            {"role": "system", "content": ATR_SYSTEM},
                            {"role": "user",   "content": build_atr_prompt(sample)},
                        ],
                    )
                    atr_output = atr_resp.choices[0].message.content

                    # ----------------------------------------------------------
                    # Call 2 — RA: JSON decision using ATR + rollout evidence
                    # ----------------------------------------------------------
                    rollout           = sample.get("rollout_results", {})
                    candidate_actions = sample.get("candidate_actions",
                                                   ["ETWT", "NTST", "ELWL", "NLSL"])
                    sr = sample.get("signal_rank", None)
                    if sr:
                        signal_priority = sorted(
                            candidate_actions,
                            key=lambda a: sr.get(a, 0.0),
                            reverse=True,
                        )
                    else:
                        signal_priority = sorted(
                            candidate_actions,
                            key=lambda a: (rollout.get(a, {}).get("queue_after_5", 0),
                                           rollout.get(a, {}).get("wait_after_5", 0.0)),
                        )

                    ra_resp = client.chat.completions.create(
                        model=args.model_name_or_path,
                        response_format={"type": "json_object"},
                        max_tokens=args.ra_max_tokens,
                        messages=[
                            {"role": "system", "content": ATR_SYSTEM},
                            {"role": "user",   "content": build_ra_prompt(
                                sample, atr_output, signal_priority)},
                        ],
                    )
                    parsed = json.loads(ra_resp.choices[0].message.content)
                    final_schema, repaired = repair_schema(parsed, sample)

                    sample["atr_output"]             = atr_output
                    sample["teacher_response"]        = final_schema
                    sample["teacher_fallback"]        = False
                    sample["teacher_schema_repaired"] = repaired
                    if args.save_raw_response:
                        sample["teacher_ra_raw"] = parsed

                    success_count += 1

                except Exception as e:
                    print(f"[ERROR] sample {idx}: {e}")
                    sample["atr_output"]       = f"(error: {e})"
                    sample["teacher_response"] = fallback_reasoning(sample)
                    sample["teacher_fallback"] = True
                    fallback_count += 1

            fout.write(json.dumps(sample) + "\n")

            if args.dry_run and idx == 0:
                print("--- dry-run preview (sample 0) ---")
                print(json.dumps(sample["teacher_response"], indent=2, ensure_ascii=False))

    print(f"Teacher rewrite done.  success={success_count}  fallback={fallback_count}")


if __name__ == "__main__":
    main()
