import argparse
import json
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "--train", type=str, default="data/FinetuneData/litepp/litepp_rco_teacher.jsonl", dest="input")
    parser.add_argument("--output_dir", "--out_dir", type=str, default="data/FinetuneData/llamafactory_litepp_rco", dest="output_dir")
    parser.add_argument("--split_ratio", type=float, default=0.9)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    samples = []
    with open(args.input, "r") as fin:
        for line in fin:
            samples.append(json.loads(line))
            
    llama_format = []
    for s in samples:
        sys_msg = "You are a traffic signal control agent."
        
        # Combine required fields for Lite++ context
        user_context = {
            "current_observation": s.get("current_observation", {}),
            "neighbor_observation": s.get("neighbor_observation", {}),
            "history": s.get("history", []),
            "candidate_actions": s.get("candidate_actions", []),
            "complexity": s.get("complexity", {}),
            "rollout_results": s.get("rollout_results", {})
        }
        
        user_msg = f"Observation:\n{json.dumps(user_context)}"
        teacher_resp = s.get('teacher_response', {})
        resp_str = json.dumps(teacher_resp)
        
        llama_format.append({
            "instruction": sys_msg,
            "input": user_msg,
            "output": resp_str
        })
        
    split_idx = int(len(llama_format) * args.split_ratio)
    train_data = llama_format[:split_idx]
    val_data = llama_format[split_idx:]
    
    with open(os.path.join(args.output_dir, "train.json"), "w") as fout:
        json.dump(train_data, fout, indent=2)
        
    if val_data:
        with open(os.path.join(args.output_dir, "val.json"), "w") as fout:
            json.dump(val_data, fout, indent=2)
            
    dataset_info = {
        "litepp_train": {"file_name": "train.json"},
        "litepp_val": {"file_name": "val.json"}
    }
    with open(os.path.join(args.output_dir, "dataset_info.json"), "w") as fout:
        json.dump(dataset_info, fout, indent=2)
        
    print(f"Exported LLaMA Factory data: {len(train_data)} train, {len(val_data)} val.")
    if train_data:
        print("====== PREVIEW 1 SAMPLE INPUT ======")
        print(train_data[0]["input"][:500] + "...\n[TRUNCATED]")
        print("====== PREVIEW 1 SAMPLE OUTPUT ======")
        print(train_data[0]["output"])

if __name__ == "__main__":
    main()
