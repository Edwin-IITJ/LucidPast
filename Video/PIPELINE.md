# LucidPast — Video-to-Splat Pipeline

Convert any video clip into a browser-viewable Gaussian Splat sequence with a single command.

---

## Quick Start

Drop your clip into `Video/input/` then run from the `Video/` directory:

```powershell
cd "path\to\LucidPast - Project\LucidPast\Video"
python run_pipeline.py <clip_name>
```

**Example** — convert `testclip2.mp4`:
```powershell
python run_pipeline.py testclip2
```

The pipeline will:
1. Extract frames from the clip at 8 fps, capped at 1024px
2. Run Apple SHARP on every 3rd frame to produce `.ply` Gaussian Splats
3. Convert all `.ply` files to the compact `.splat` web format
4. Launch a local HTTP server and open the viewer automatically in your browser

---

## Options

| Flag | Default | Description |
|---|---|---|
| `--fps N` | `8` | Frame extraction rate. Higher = smoother but slower. |
| `--max-dim N` | `1024` | Cap the longer edge of each frame in pixels. Lower = faster SHARP runs. |
| `--every-nth N` | `3` | Run SHARP on every Nth extracted frame. `1` = every frame (slow but highest quality). |
| `--port N` | `8001` | HTTP port for the viewer server. |
| `--no-server` | off | Skip launching the server (just process and exit). |
| `--skip-extract` | off | Reuse existing frames in `output/<clip>/frames/`. |
| `--skip-sharp` | off | Reuse existing `.ply` files in `output/<clip>/splats/`. |
| `--skip-convert` | off | Reuse existing `.splat` files in `output/<clip>/splats/`. |

---

## Common Recipes

**Cheap first-pass spike** (quick preview, every 5th frame at 480p):
```powershell
python run_pipeline.py myclip --every-nth 5 --max-dim 480
```

**Full quality run** (every frame, native resolution):
```powershell
python run_pipeline.py myclip --every-nth 1 --fps 30
```

**Re-serve an existing run** (no reprocessing):
```powershell
python run_pipeline.py myclip --skip-extract --skip-sharp --skip-convert
```

**Re-convert splats from existing .ply files** (e.g. after updating the converter):
```powershell
python run_pipeline.py myclip --skip-extract --skip-sharp
```

**Process without auto-opening the browser** (headless / remote):
```powershell
python run_pipeline.py myclip --no-server
# Then manually: cd output\myclip\splats && python -m http.server 8001
```

---

## Directory Layout

```
Video/
├── run_pipeline.py          ← The single-command entry point
├── input/                   ← Drop your source video clips here
│   ├── testclip.mp4
│   └── testclip2.mp4
├── output/                  ← Pipeline outputs land here (auto-created)
│   └── <clip_name>/
│       ├── frames/          ← Extracted PNG frames
│       └── splats/          ← Per-frame .ply + .splat files + manifest.json
│           └── view_sequence.html  ← Viewer (copied here automatically)
└── phase0-spike/
    ├── scripts/
    │   ├── extract_frames.sh       ← Step 1: ffmpeg frame extraction
    │   ├── run_sharp_batch.py      ← Step 2: SHARP per-frame reconstruction
    │   └── convert_to_splat.py    ← Step 3: PLY → .splat format converter
    └── viewer/
        └── view_sequence.html      ← Canonical viewer source
```

---

## Viewer Controls

Once the viewer opens in your browser:

| Key / Input | Action |
|---|---|
| `W A S D` | Move forward / left / backward / right |
| `Q E` | Move down / up |
| `Mouse drag` | Look around |
| `Scroll wheel` | Adjust movement speed |
| `Space` | Play / Pause |
| `← →` | Step one frame (while paused) |

---

## Supported Input Formats

The pipeline accepts any format that `ffmpeg` can read:  
`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`, `.m4v`

Just drop the file in `Video/input/` and pass the name (without extension) to `run_pipeline.py`.

---

## Output Files

| File | What it is |
|---|---|
| `output/<clip>/frames/frame_NNNNN.png` | Extracted video frames (lossless PNG) |
| `output/<clip>/splats/frame_NNNNN.ply` | Raw SHARP Gaussian Splat per frame |
| `output/<clip>/splats/frame_NNNNN.splat` | Compact 32-byte-per-Gaussian web format |
| `output/<clip>/splats/manifest.json` | Ordered list of `.splat` files for the viewer |
| `output/<clip>/splats/_spike_log.json` | Per-frame SHARP timing + error log |
| `output/<clip>/splats/view_sequence.html` | Self-contained viewer (served via HTTP) |

---

## Troubleshooting

**"No video found for 'myclip'"**  
Make sure the file is in `Video/input/` and the name you pass matches exactly (case-sensitive on Linux/Mac).

**SHARP fails on certain frames**  
Frames that fail are logged in `_spike_log.json` and skipped; the rest of the sequence still plays. Check the log for `"error"` entries.

**Viewer shows black / nothing loads**  
The viewer must be served over HTTP — opening `view_sequence.html` directly from the file system (`file://`) will fail due to CORS. Always use the server: `python -m http.server 8001`.

**Port already in use**  
Pass a different port: `python run_pipeline.py myclip --port 8002`.

---

## Requirements

- `ffmpeg` on your `PATH` (tested with 9.x)
- `python` with `numpy` (used by the converter)
- `bash` (for `extract_frames.sh`) — on Windows this runs through Git Bash / WSL
- Apple SHARP configured in `../env_sharp/` and `../ml-sharp/` (same setup as the image pipeline)
