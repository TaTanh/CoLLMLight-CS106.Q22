import argparse
import json
import os
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/collmlight_litepp.yaml")
    parser.add_argument("--input", type=str, default="data/FinetuneData/litepp/litepp_rco_rollout.jsonl")
    parser.add_argument("--output", type=str, default="data/FinetuneData/litepp/litepp_rco_teacher.jsonl")
    parser.add_argument("--model_name_or_path", type=str, default="gpt-4o-mini")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--save_raw_response", action="store_true", help="Save the raw unparsed response to the output")
    parser.add_argument("--max_tokens", type=int, default=350, help="Max tokens for GPT output")
    parser.add_argument("--compact", action="store_true", default=True, help="Enforce compact schema length")
    return parser.parse_args()

def fallback_reasoning(sample):
    c_label = sample.get("complexity", {}).get("label", "NO")
    g_action = sample.get("pseudo_golden_action", "UNKNOWN")
    response = {
        "phase1": {
            "answer": c_label,
            "reason": "Traffic conditions show routine flow and manageable queues based on fallback evaluation."
        },
        "phase2": {
            "traffic_analysis": "Analyzed intersection queues and wait times across all approaching lanes.",
            "future_state_summary": "Selecting the golden phase minimizes total network delay and prevents spillback.",
            "signal_comparison": "Compared candidate phases against current backlog.",
            "answer": g_action
        }
    }

    # Add signal_consequence_prediction for Complex cases
    if c_label == "Complex":
        response["phase2"]["signal_consequence_prediction"] = {
            "ETWT": "Estimated queue change minimal.",
            "NTST": "Estimated queue change minimal.",
            "ELWL": "Estimated queue change minimal.",
            "NLSL": "Estimated queue change minimal."
        }

    return response

def repair_schema(parsed_schema, sample):
    repaired = False
    g_action = sample.get("pseudo_golden_action", "UNKNOWN")
    c_label = sample.get("complexity", {}).get("label", "NO")
    
    # Store raw response just in case
    raw_response = json.loads(json.dumps(parsed_schema))
    
    if "phase1" not in parsed_schema or not isinstance(parsed_schema["phase1"], dict):
        parsed_schema["phase1"] = {}
        repaired = True
    if "phase2" not in parsed_schema or not isinstance(parsed_schema["phase2"], dict):
        parsed_schema["phase2"] = {}
        repaired = True
        
    p1 = parsed_schema["phase1"]
    p2 = parsed_schema["phase2"]
    
    if "reason" not in p1:
        p1["reason"] = raw_response.get("reason", "Traffic conditions show routine flow and manageable queues.")
        repaired = True
        
    if "traffic_analysis" not in p2:
        p2["traffic_analysis"] = raw_response.get("traffic_analysis", raw_response.get("phase_analysis", "Analyzed intersection queues and wait times across all approaching lanes."))
        repaired = True
        
    if "future_state_summary" not in p2:
        p2["future_state_summary"] = raw_response.get("future_state_summary", "Selecting the proper phase minimizes total delay and prevents spillback.")
        repaired = True
        
    if "signal_comparison" not in p2:
        p2["signal_comparison"] = raw_response.get("signal_comparison", "Compared candidate phases against current backlog.")
        repaired = True

    # Handle signal_consequence_prediction for Complex cases
    if c_label == "Complex":
        if "signal_consequence_prediction" not in p2:
            p2["signal_consequence_prediction"] = {
                "ETWT": "Estimated queue change minimal.",
                "NTST": "Estimated queue change minimal.",
                "ELWL": "Estimated queue change minimal.",
                "NLSL": "Estimated queue change minimal."
            }
            repaired = True
        elif not isinstance(p2.get("signal_consequence_prediction"), dict):
            p2["signal_consequence_prediction"] = {
                "ETWT": "Estimated queue change minimal.",
                "NTST": "Estimated queue change minimal.",
                "ELWL": "Estimated queue change minimal.",
                "NLSL": "Estimated queue change minimal."
            }
            repaired = True

    # Force strict alignment
    p1["answer"] = c_label
    p2["answer"] = g_action
    
    # Prune unwanted top-level keys to match exact schema
    final_schema = {
        "phase1": {
            "answer": p1["answer"],
            "reason": p1["reason"]
        },
        "phase2": {
            "traffic_analysis": p2["traffic_analysis"],
            "future_state_summary": p2["future_state_summary"],
            "signal_comparison": p2["signal_comparison"],
            "answer": p2["answer"]
        }
    }

    # Add signal_consequence_prediction for Complex cases only
    if c_label == "Complex" and "signal_consequence_prediction" in p2:
        final_schema["phase2"]["signal_consequence_prediction"] = p2["signal_consequence_prediction"]

    return final_schema, repaired, raw_response

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    success_count = 0
    fallback_count = 0
    
    try:
        from openai import OpenAI
        has_openai = True
    except ImportError:
        has_openai = False
        
    if not args.dry_run and has_openai:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "dummy"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
        
    with open(args.input, "r") as fin, open(args.output, "w") as fout:
        for idx, line in enumerate(fin):
            if args.max_samples > 0 and idx >= args.max_samples:
                break
                
            sample = json.loads(line)
            
            if args.dry_run or not has_openai:
                sample["teacher_response"] = fallback_reasoning(sample)
                sample["teacher_fallback"] = True
                fallback_count += 1
            else:
                try:
                    sys_msg = "You are an expert traffic signal control agent."
                    
                    c_label = sample.get("complexity", {}).get("label", "NO")
                    complexity_guidance = {
                        "NO": (
                            "The fixed complexity label is NO: explain that the signal choice can be made "
                            "mostly from local traffic conditions without cooperative neighbor reasoning. "
                            "Do not describe the case as Simple or Complex."
                        ),
                        "Simple": (
                            "The fixed complexity label is Simple: explain that limited neighbor consideration "
                            "is useful for one main interaction. Do not describe the case as Complex."
                        ),
                        "Complex": (
                            "The fixed complexity label is Complex: explain that multiple neighbor interactions "
                            "or spillback risks require deeper cooperative reasoning."
                        )
                    }.get(c_label, "Use the fixed complexity label exactly and keep the reason consistent with it.")

                    observation_context = {
                        "current_observation": sample.get("current_observation"),
                        "neighbor_observation": sample.get("neighbor_observation"),
                        "history": sample.get("history"),
                        "candidate_actions": sample.get("candidate_actions")
                    }

                    schema_req_content = (
                        "Provide a JSON response strictly matching this schema. Keep responses extremely concise and to the point:\n"
                        "{\n"
                        "  \"phase1\": {\n"
                        "    \"answer\": \"<complexity_label>\",\n"
                        "    \"reason\": \"<strictly max 1 sentence explaining complexity>\"\n"
                        "  },\n"
                        "  \"phase2\": {\n"
                        "    \"traffic_analysis\": \"<strictly max 2 sentences analyzing current local and neighbor observation>\",\n"
                        "    \"future_state_summary\": \"<strictly max 1 sentence summarizing expected rollout queue states>\",\n"
                        "    \"signal_comparison\": \"<strictly max 1 sentence comparing candidate actions>\""
                    )
                    if c_label == "Complex":
                        schema_req_content += (
                            ",\n"
                            "    \"signal_consequence_prediction\": {\n"
                            "      \"ETWT\": \"<queue prediction 1 sentence>\",\n"
                            "      \"NTST\": \"<queue prediction 1 sentence>\",\n"
                            "      \"ELWL\": \"<queue prediction 1 sentence>\",\n"
                            "      \"NLSL\": \"<queue prediction 1 sentence>\"\n"
                            "    }"
                        )
                    schema_req_content += (
                        ",\n"
                        "    \"answer\": \"<best_action_str>\"\n"
                        "  }\n"
                        "}"
                    )
                    
                    user_msg = (
                        f"Complexity Guidance: {complexity_guidance}\n"
                        f"Observation Context: {json.dumps(observation_context)}\n"
                        f"Rollout Results: {json.dumps(sample.get('rollout_results'))}\n"
                        f"{schema_req_content}"
                    )
                    
                    response = client.chat.completions.create(
                        model=args.model_name_or_path,
                        response_format={"type": "json_object"},
                        max_tokens=args.max_tokens,
                        messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]
                    )
                    
                    parsed_schema = json.loads(response.choices[0].message.content)
                    final_schema, repaired, raw_response = repair_schema(parsed_schema, sample)
                    
                    sample["teacher_response"] = final_schema
                    sample["teacher_fallback"] = False
                    sample["teacher_schema_repaired"] = repaired
                    
                    if args.save_raw_response:
                        sample["teacher_raw_response"] = raw_response
                        
                    success_count += 1
                except Exception as e:
                    print(f"API Error at sample {idx}: {e}")
                    sample["teacher_response"] = fallback_reasoning(sample)
                    sample["teacher_fallback"] = True
                    fallback_count += 1
            
            fout.write(json.dumps(sample) + "\n")
            if args.dry_run and idx == 0:
                print("Preview dry-run structure:")
                print(json.dumps(sample["teacher_response"], indent=2))
                print("Teacher Rewrite Complete.")
    print(f"Success: {success_count}, Fallback: {fallback_count}")

if __name__ == "__main__":
    main()
