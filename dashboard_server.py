import os
import json
import csv
import subprocess
import threading
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
try:
    from http.server import ThreadingHTTPServer
except ImportError:
    ThreadingHTTPServer = HTTPServer
from urllib.parse import urlparse, parse_qs

PORT = 8050
CO_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(CO_DIR, "frontend")
RECORDS_DIR = os.path.join(CO_DIR, "records")
OUTPUTS_DIR = os.path.join(CO_DIR, "outputs")

# Global state to track background simulation subprocesses
sim_process = None
sim_output = []
sim_lock = threading.Lock()

def run_simulation_thread(cmd):
    global sim_process, sim_output
    sim_output.clear()
    sim_output.append(f"Running command: {' '.join(cmd)}\n")
    try:
        # Run using virtual environment's python if available
        venv_python = os.path.join(CO_DIR, ".venv", "bin", "python")
        if os.path.exists(venv_python):
            cmd[0] = venv_python
        
        # Start subprocess
        sim_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=CO_DIR,
            text=True,
            bufsize=1
        )
        
        for line in sim_process.stdout:
            with sim_lock:
                sim_output.append(line)
                if len(sim_output) > 2000:  # Cap log history
                    sim_output.pop(0)
        
        sim_process.wait()
        with sim_lock:
            sim_output.append(f"\nSimulation finished with return code {sim_process.returncode}\n")
    except Exception as e:
        with sim_lock:
            sim_output.append(f"\nError running simulation: {str(e)}\n")
    finally:
        sim_process = None

class DashboardHTTPRequestHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query = parse_qs(parsed_url.query)

        # REST API Routes
        if path == "/api/runs":
            self.handle_list_runs()
        elif path == "/api/run_file":
            self.handle_run_file(query)
        elif path == "/api/simulate_status":
            self.handle_simulate_status()
        elif path == "/api/metrics":
            self.handle_metrics()
        else:
            # Serve static files
            self.handle_static(path)

    def do_POST(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        if path == "/api/simulate":
            self.handle_start_simulate()
        else:
            self.send_error(404, "Not Found")

    def handle_static(self, path):
        # Prevent Directory Traversal
        if ".." in path:
            self.send_error(400, "Bad Request")
            return

        # Map root request to index.html
        if path == "/" or path == "":
            path = "/index.html"

        # Resolve file path in frontend directory
        file_path = os.path.join(FRONTEND_DIR, path.lstrip("/"))
        if not os.path.exists(file_path) or os.path.isdir(file_path):
            self.send_error(404, f"File Not Found: {path}")
            return

        # Determine Content-Type
        content_type = "text/plain"
        if path.endswith(".html"):
            content_type = "text/html"
        elif path.endswith(".css"):
            content_type = "text/css"
        elif path.endswith(".js"):
            content_type = "application/javascript"
        elif path.endswith(".json"):
            content_type = "application/json"
        elif path.endswith(".png"):
            content_type = "image/png"
        elif path.endswith(".svg"):
            content_type = "image/svg+xml"

        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Internal Server Error: {str(e)}")

    def handle_list_runs(self):
        runs = []
        if os.path.exists(RECORDS_DIR):
            for entry in sorted(os.listdir(RECORDS_DIR)):
                entry_path = os.path.join(RECORDS_DIR, entry)
                if os.path.isdir(entry_path):
                    has_replay = os.path.exists(os.path.join(entry_path, "replayLogFile.txt"))
                    has_roadnet = os.path.exists(os.path.join(entry_path, "roadnetLogFile.json"))
                    has_reasoning = os.path.exists(os.path.join(entry_path, "reasoning_log.json"))
                    
                    if has_replay and has_roadnet:
                        runs.append({
                            "id": entry,
                            "name": entry.replace("litepp_eval_", "").replace("_", " ").title(),
                            "has_reasoning": has_reasoning
                        })
        
        response_bytes = json.dumps(runs).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

    def handle_run_file(self, query):
        run_id = query.get("run_id", [""])[0]
        filename = query.get("file", [""])[0]

        if not run_id or not filename or ".." in run_id or ".." in filename:
            self.send_error(400, "Bad Request")
            return

        file_path = os.path.join(RECORDS_DIR, run_id, filename)
        if not os.path.exists(file_path):
            self.send_error(404, "File Not Found")
            return

        content_type = "text/plain"
        if filename.endswith(".json"):
            content_type = "application/json"
        elif filename.endswith(".txt"):
            content_type = "text/plain"

        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Internal Server Error: {str(e)}")

    def handle_simulate_status(self):
        global sim_process, sim_output
        running = (sim_process is not None)
        with sim_lock:
            output_str = "".join(sim_output)
        
        res = {
            "running": running,
            "output": output_str
        }
        response_bytes = json.dumps(res).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

    def handle_start_simulate(self):
        global sim_process
        if sim_process is not None:
            self.send_error(400, "Simulation already running")
            return

        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            params = json.loads(body.decode('utf-8'))
        except Exception:
            self.send_error(400, "Invalid JSON body")
            return

        dataset = params.get("dataset", "synth")
        model = params.get("model", "student")
        endpoint = params.get("endpoint", "http://localhost:8000/v1")
        time_limit = str(params.get("time", 300))

        # Command to launch simulation
        cmd = [
            "python",
            "scripts/evaluate_litepp_student.py",
            "--dataset", dataset,
            "--model", model,
            "--endpoint", endpoint,
            "--simulation_time", time_limit,
            "--save_replay"
        ]

        # Launch in background thread
        thread = threading.Thread(target=run_simulation_thread, args=(cmd,))
        thread.start()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "started"}).encode("utf-8"))

    def handle_metrics(self):
        csv_path = os.path.join(OUTPUTS_DIR, "litepp_eval_results.csv")
        metrics = []
        if os.path.exists(csv_path):
            try:
                with open(csv_path, mode="r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        metrics.append({
                            "dataset": row.get("dataset"),
                            "model": row.get("model"),
                            "ATT": float(row.get("ATT", 0.0)),
                            "AWT": float(row.get("AWT", 0.0)),
                            "n_vehicles": int(row.get("n_vehicles", 0))
                        })
            except Exception as e:
                print(f"Error parsing metrics: {e}")
        
        response_bytes = json.dumps(metrics).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

def main():
    print(f"Starting server on port {PORT}...")
    print(f"Serving static assets from: {FRONTEND_DIR}")
    print(f"Serving simulation records from: {RECORDS_DIR}")
    server = ThreadingHTTPServer(("0.0.0.0", PORT), DashboardHTTPRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.server_close()

if __name__ == "__main__":
    main()
