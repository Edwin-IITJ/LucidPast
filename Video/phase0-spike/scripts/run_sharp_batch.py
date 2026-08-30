#!/usr/bin/env python3
"""
Phase 0 spike: run Apple ml-sharp (SHARP) on each extracted frame
independently, producing a naive (no temporal smoothing) per-frame
.ply sequence. This answers one question: does per-frame SHARP
flicker unusably on video, or is it good enough to build on?

Design choices, given the RTX 3060 mobile / 6GB target:

- Each frame is run in its OWN subprocess rather than importing SHARP
  once and looping in-process. A single-image feedforward pass is
  cheap, but on a tight VRAM budget across a run of potentially
  hundreds of frames, a fresh process per frame is the simplest way
  to guarantee nothing accumulates -- the OS reclaims everything when
  the process exits. The tradeoff is repeated model-load overhead;
  if that turns out to dominate runtime, switch to an in-process loop
  with an explicit `torch.cuda.empty_cache()` between frames instead.
- A per-frame timeout and try/except mean one bad or OOM-ing frame
  gets logged and skipped rather than killing the whole batch --
  useful given you've already seen SHARP produce the occasional
  anomalous result on stills.
- `every_nth` lets you do an even cheaper first pass (e.g. every 3rd
  frame) before committing to processing every frame of even a short
  clip.

EDIT SHARP_CMD_TEMPLATE BELOW to match however you currently invoke
SHARP for a single image in LucidPast -- this is scaffolding around
your existing setup, not a guess at Apple's exact CLI flags.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

# --- CONFIG: adjust to match your existing LucidPast / env_sharp setup ---
import os
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SHARP_CMD_TEMPLATE = [
    os.path.join(base_dir, "env_sharp", "Scripts", "sharp.exe"), "predict",
    "-i", "{input}",
    "-o", "{output}",
    "-c", os.path.join(base_dir, "ml-sharp", "sharp_model.pt"),
]
CONDA_ENV = None          # set to None if you invoke SHARP directly
TIMEOUT_SECONDS = 300             # per-frame timeout -- tune after a few runs
# ---------------------------------------------------------------------


def build_command(input_path: Path, output_path: Path):
    cmd = [part.format(input=str(input_path), output=str(output_path))
           for part in SHARP_CMD_TEMPLATE]
    if CONDA_ENV:
        cmd = ["conda", "run", "-n", CONDA_ENV] + cmd
    return cmd


def run_batch(frames_dir: Path, out_dir: Path, every_nth: int = 1):
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(frames_dir.glob("frame_*.png"))[::every_nth]
    if not frames:
        print(f"No frames found in {frames_dir} (expected frame_*.png)")
        return

    log = []
    for i, frame_path in enumerate(frames):
        output_subdir = out_dir / frame_path.stem
        cmd = build_command(frame_path, output_subdir)
        start = time.time()
        try:
            subprocess.run(
                cmd, timeout=TIMEOUT_SECONDS,
                capture_output=True, text=True, check=True,
            )
            elapsed = time.time() - start
            
            # Find the generated .ply file and move it to out_dir
            ply_files = list(output_subdir.glob("*.ply"))
            ply_path = out_dir / f"{frame_path.stem}.ply"
            ok = False
            if ply_files:
                ply_files[0].rename(ply_path)
                import shutil
                shutil.rmtree(output_subdir, ignore_errors=True)
                ok = True
                
            log.append({"frame": frame_path.name, "ply": ply_path.name,
                        "ok": ok, "seconds": round(elapsed, 2)})
            status = "ok" if ok else "MISSING OUTPUT"
            print(f"[{i+1}/{len(frames)}] {frame_path.name} -> {status} ({elapsed:.1f}s)")
        except subprocess.TimeoutExpired:
            log.append({"frame": frame_path.name, "ok": False, "error": "timeout"})
            print(f"[{i+1}/{len(frames)}] {frame_path.name} -> TIMEOUT, skipped")
        except subprocess.CalledProcessError as e:
            err = (e.stderr or "")[-500:]
            log.append({"frame": frame_path.name, "ok": False, "error": err})
            print(f"[{i+1}/{len(frames)}] {frame_path.name} -> FAILED, see _spike_log.json")

    (out_dir / "_spike_log.json").write_text(json.dumps(log, indent=2))

    ok_plys = [entry["ply"] for entry in log if entry.get("ok")]
    (out_dir / "manifest.json").write_text(json.dumps(ok_plys, indent=2))

    failed = len(log) - len(ok_plys)
    print(f"\nDone. {len(ok_plys)}/{len(log)} frames succeeded.")
    if failed:
        print(f"{failed} frame(s) need manual review -- see _spike_log.json")
    print(f"manifest.json written for the viewer ({len(ok_plys)} entries)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: run_sharp_batch.py <frames_dir> <output_dir> [every_nth]")
        sys.exit(1)
    frames_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    every_nth = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    run_batch(frames_dir, out_dir, every_nth)
