import os
import sys
import time
import uuid
import yaml
import subprocess
import threading
from pathlib import Path
from flask import Flask, request, jsonify, send_file, send_from_directory, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# Setup and Config Loading
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_config():
    config_path = PROJECT_ROOT / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}

config = load_config()

# Resolve Paths
paths_cfg = config.get("paths", {})
web_cfg = config.get("web", {})

INPUT_DIR = PROJECT_ROOT / paths_cfg.get("input_dir", "input")
OUTPUT_DIR = PROJECT_ROOT / paths_cfg.get("output_dir", "output/gaussians")
CHECKPOINT = PROJECT_ROOT / paths_cfg.get("checkpoint", "ml-sharp/sharp_model.pt")
SHARP_EXE = PROJECT_ROOT / paths_cfg.get("sharp_exe", "env_sharp/Scripts/sharp.exe")

HOST = web_cfg.get("host", "127.0.0.1")
PORT = web_cfg.get("port", 5000)

SUPPORTED_EXTENSIONS = config.get("sharp", {}).get("supported_extensions", [
    ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"
])

# Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Flask App Setup
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)  # Allow cross-origin requests for WebXR viewer

# Job state dictionary
# Format: { "job_id": { "status": "running|done|failed", "logs": ["line1", "line2"], "image_name": "filename" } }
jobs = {}

def process_image_job(job_id, file_path):
    """Background thread function to run SHARP and capture logs."""
    image_name = file_path.stem
    output_subdir = OUTPUT_DIR / image_name
    
    # Clean up previous if exists
    if output_subdir.exists():
        import shutil
        shutil.rmtree(output_subdir, ignore_errors=True)
        
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    jobs[job_id]["logs"].append(f"[INFO] Started processing {file_path.name}")
    
    command = [
        str(SHARP_EXE), "predict",
        "-i", str(file_path),
        "-o", str(output_subdir),
        "-c", str(CHECKPOINT),
    ]
    
    jobs[job_id]["logs"].append(f"[CMD] {' '.join(command)}")
    
    try:
        # Run process and stream output
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(PROJECT_ROOT)
        )
        
        for line in iter(process.stdout.readline, ''):
            if line:
                jobs[job_id]["logs"].append(line.strip())
                
        process.stdout.close()
        return_code = process.wait()
        
        if return_code == 0:
            jobs[job_id]["status"] = "done"
            jobs[job_id]["logs"].append(f"[INFO] Successfully completed: {image_name}")
        else:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["logs"].append(f"[ERROR] Process exited with code {return_code}")
            
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["logs"].append(f"[ERROR] Exception during execution: {str(e)}")

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/api/process", methods=["POST"])
def process_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
        
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
        
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return jsonify({"error": f"Unsupported extension. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"}), 400
        
    # Save file
    filename = secure_filename(file.filename)
    file_path = INPUT_DIR / filename
    file.save(str(file_path))
    
    # Start job
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "running",
        "logs": [],
        "image_name": Path(filename).stem
    }
    
    thread = threading.Thread(target=process_image_job, args=(job_id, file_path))
    thread.daemon = True
    thread.start()
    
    return jsonify({"job_id": job_id, "status": "running"})

@app.route("/api/status/<job_id>", methods=["GET"])
def get_status(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
        
    job_info = jobs[job_id]
    
    # Allow client to specify which log index they last saw
    last_log_idx = request.args.get("last_log_idx", 0, type=int)
    new_logs = job_info["logs"][last_log_idx:]
    
    return jsonify({
        "status": job_info["status"],
        "logs": new_logs,
        "next_log_idx": len(job_info["logs"]),
        "image_name": job_info["image_name"]
    })

@app.route("/api/splats", methods=["GET"])
def list_splats():
    """List all available splat scenes."""
    splats = []
    if OUTPUT_DIR.exists():
        for item in OUTPUT_DIR.iterdir():
            if item.is_dir():
                # Check if scene.ply or <name>.ply exists
                ply_files = list(item.glob("*.ply"))
                if ply_files:
                    splats.append({
                        "name": item.name,
                        "file": ply_files[0].name
                    })
    
    # Sort by name
    splats.sort(key=lambda x: x["name"])
    return jsonify({"splats": splats})

@app.route("/api/splat/<name>/<file>", methods=["GET"])
def serve_splat(name, file):
    """Serve the .ply file for a specific splat."""
    splat_dir = OUTPUT_DIR / name
    if not splat_dir.exists():
        return jsonify({"error": "Splat not found"}), 404
        
    file_path = splat_dir / file
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404
        
    return send_file(str(file_path))

if __name__ == "__main__":
    print(f"============================================================")
    print(f" LucidPast Web Interface")
    print(f"============================================================")
    print(f" Host: http://{HOST}:{PORT}")
    print(f" SHARP Executable: {SHARP_EXE}")
    print(f" Output Directory: {OUTPUT_DIR}")
    print(f"============================================================")
    
    app.run(host=HOST, port=PORT, debug=True)
