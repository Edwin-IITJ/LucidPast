# Phase 1 — Ingest & Preprocess

> **Status: CURRENT PHASE — actively being built.**

## Scope

This module handles everything between "raw video file" and "clean, individual frames ready for SHARP."

## What Was Built

### Entry Point

```bash
python ingest.py <video_file> [options]
```

### Steps (executed in order)

| Step | File | What It Does |
|---|---|---|
| 1. Extract | `steps/extract.py` | ffmpeg frame extraction → lossless PNGs. Pure Python (no bash). Configurable fps + resolution cap. |
| 2. Detect shots | `steps/detect_shots.py` | Classical histogram-correlation shot cut detection. Each shot becomes an independent reconstruction run. |
| 3. Deflicker | `steps/deflicker.py` | Rolling-window L-channel histogram matching in LAB space. Normalizes brightness without affecting color hue. |
| 4. Stabilize | `steps/stabilize.py` | Sparse optical flow (Lucas-Kanade) jitter correction. Skips frames with intentional camera movement (> `max_motion_px`). |

### Options

| Flag | Default | Description |
|---|---|---|
| `--fps N` | `8` | Frame extraction rate |
| `--max-dim N` | `1024` | Cap the longer edge in pixels |
| `--shot-threshold F` | `0.7` | Histogram correlation threshold for shot cuts |
| `--deflicker-window N` | `5` | Rolling window half-size for deflicker |
| `--max-motion F` | `15.0` | Max median motion (px) before skipping stabilization |
| `--no-stabilize` | off | Skip stabilization entirely |
| `--no-deflicker` | off | Skip deflicker entirely |
| `--skip-extract` | off | Reuse existing extracted frames |
| `--output-dir PATH` | auto | Override output directory |

### Output Structure

```
pipeline/ingest/output/<clip_name>/
├── frames/          ← numbered lossless PNGs (frame_00001.png, ...)
├── shots.json       ← [{shot_id, start_frame, end_frame, frame_count}]
└── ingest_log.json  ← full processing summary with timings
```

### Examples

```bash
# Full pipeline, default settings:
python ingest.py ../../Video/input/testclip2.mp4

# Custom fps and resolution:
python ingest.py ../../Video/input/testclip2.mp4 --fps 10 --max-dim 720

# Skip stabilization (known intentional camera motion):
python ingest.py ../../Video/input/testclip2.mp4 --no-stabilize

# Re-run deflicker + stabilize on existing frames:
python ingest.py ../../Video/input/testclip2.mp4 --skip-extract
```

## Constraints

- Output frames must be lossless PNG (no additional compression artifacts before SHARP).
- Must handle non-standard aspect ratios and resolutions common in archival material.
- No generative processing of any kind.

## Dependencies

- `ffmpeg` on PATH
- `opencv-python` (pip)
- `numpy` (pip)

## Versioning Note

Existing outputs in `Video/output/testclip/` and `Video/output/testclip2/` are **v1 baseline** — this pipeline writes to `pipeline/ingest/output/`, a completely separate location. V1 outputs are preserved for A/B comparison after Phase 3.
