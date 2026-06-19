import os
import subprocess
import glob
import csv
import sys
import re

CO_DIR = "/home/tathanh/Desktop/CoLLM/Co"
RESULTS_DIR = os.path.join(CO_DIR, "results", "latest_results")
CSV_PATH = os.path.join(CO_DIR, "outputs", "litepp_eval_results.csv")

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

runs = [
    {
        "name": "Fixedtime",
        "dataset": "jinan_3x4",
        "cmd": ["python3", "run_fixedtime.py", "--dataset", "jinan", "--traffic_file", "anon_3_4_jinan_real_2000.json"]
    },
    {
        "name": "Fixedtime",
        "dataset": "hangzhou_4x4",
        "cmd": ["python3", "run_fixedtime.py", "--dataset", "hangzhou", "--traffic_file", "anon_4_4_hangzhou_real.json"]
    },
    {
        "name": "MaxPressure",
        "dataset": "jinan_3x4",
        "cmd": ["python3", "run_maxpressure.py", "--dataset", "jinan", "--traffic_file", "anon_3_4_jinan_real_2000.json"]
    },
    {
        "name": "MaxPressure",
        "dataset": "hangzhou_4x4",
        "cmd": ["python3", "run_maxpressure.py", "--dataset", "hangzhou", "--traffic_file", "anon_4_4_hangzhou_real.json"]
    }
]

def clean_results():
    files = glob.glob(os.path.join(RESULTS_DIR, "*.txt"))
    for f in files:
        try:
            os.remove(f)
        except OSError:
            pass

def parse_txt_file(filepath):
    metrics = {}
    with open(filepath, "r") as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                metrics[k.strip()] = float(v.strip())
    return metrics

def run_all():
    print("Cleaning old results...")
    clean_results()

    for run in runs:
        print(f"\n======================================")
        print(f"Running {run['name']} on {run['dataset']}...")
        print(f"Command: {' '.join(run['cmd'])}")
        
        # We need to run inside the virtual env python
        venv_python = os.path.join(CO_DIR, ".venv", "bin", "python")
        cmd = run['cmd'].copy()
        if os.path.exists(venv_python):
            cmd[0] = venv_python

        # Execute
        try:
            # We run it and wait
            subprocess.run(cmd, cwd=CO_DIR, check=True)
            
            # Find the generated txt file
            txt_files = glob.glob(os.path.join(RESULTS_DIR, "*.txt"))
            if not txt_files:
                print("Error: No result file generated!")
                continue
            
            # Since we cleaned it, there should be exactly one txt file
            result_file = txt_files[0]
            print(f"Found result file: {result_file}")
            
            metrics = parse_txt_file(result_file)
            print(f"Parsed metrics: {metrics}")
            
            att = metrics.get("test_avg_travel_time_over", 0.0)
            awt = metrics.get("test_avg_waiting_time_over", 0.0)
            n_veh = int(metrics.get("test_throughput_over", 0))
            
            # Round values
            att = round(att, 2)
            awt = round(awt, 2)

            # Append to CSV
            write_header = not os.path.exists(CSV_PATH)
            with open(CSV_PATH, "a", newline="") as csv_f:
                writer = csv.DictWriter(csv_f, fieldnames=["dataset", "model", "ATT", "AWT", "n_vehicles"])
                if write_header:
                    writer.writeheader()
                writer.writerow({
                    "dataset": run["dataset"],
                    "model": run["name"],
                    "ATT": att,
                    "AWT": awt,
                    "n_vehicles": n_veh
                })
            print(f"Appended {run['name']} - {run['dataset']} to {CSV_PATH}")
            
            # Delete this file so the next runs have a clean dir
            os.remove(result_file)
            
        except subprocess.CalledProcessError as e:
            print(f"Command failed with error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

if __name__ == "__main__":
    run_all()
