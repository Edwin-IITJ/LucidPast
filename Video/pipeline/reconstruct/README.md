# Phase 2 — Reconstruct (SHARP Wrapper)

> **Status: COMPLETE.** Verified on both test clips (every 3rd frame). Default changed to every frame for production quality.

## Scope

Production-grade batch wrapper around Apple SHARP. Reads clean frames from Phase 1 and produces per-frame `.ply` Gaussian Splats with integrated `.splat` conversion.

## What Was Built

### Entry Point

```bash
python reconstruct.py <ingest_output_dir> [options]

# Process all frames (default — best quality):
python reconstruct.py ../ingest/output/testclip2

# Process every 3rd frame (faster, lower quality):
python reconstruct.py ../ingest/output/testclip2 --every-nth 3

# Resume an interrupted run (auto-skips existing .ply files):
python reconstruct.py ../ingest/output/testclip2
```

### Improvements Over Spike

| Feature | Spike (`run_sharp_batch.py`) | Production (`reconstruct.py`) |
|---|---|---|
| Resume support | ❌ Re-runs everything | ✅ Skips existing .ply files |
| Progress / ETA | Basic counter | Per-frame ETA based on running average |
| .splat conversion | Separate script | ✅ Integrated (inlined from `convert_to_splat.py`) |
| Anomaly detection | ❌ | ✅ Flags frames with Gaussian count > 20% from median |
| Logging | `_spike_log.json` | Detailed `reconstruct_log.json` with per-frame timing |
| Error handling | Basic timeout + catch | Timeout + CalledProcessError + FileNotFoundError + cleanup |

### Options

| Flag | Default | Description |
|---|---|---|
| `--every-nth N` | `1` | Process every Nth frame (1 = all frames) |
| `--timeout N` | `300` | Per-frame timeout in seconds |
| `--sharp-exe PATH` | auto | Path to `sharp.exe` |
| `--checkpoint PATH` | auto | Path to `sharp_model.pt` |
| `--output-dir PATH` | auto | Override output directory |
| `--no-convert` | off | Skip `.ply` → `.splat` conversion |

### Output Structure

```
pipeline/reconstruct/output/<clip_name>/
├── splats/
│   ├── frame_00001.ply       ← raw SHARP output
│   ├── frame_00001.splat     ← compact 32-byte-per-Gaussian web format
│   ├── ...
│   └── manifest.json         ← ordered list of .splat files
└── reconstruct_log.json      ← per-frame timing, errors, anomaly flags
```

## Constraints

- Must run on RTX 3060 (12 GB desktop / 6 GB mobile).
- Each frame is an independent reconstruction — no cross-frame state.
- Each frame runs in its own subprocess (VRAM safety on tight budgets).

## Dependencies

- SHARP CLI at `env_sharp/Scripts/sharp.exe`
- SHARP checkpoint at `ml-sharp/sharp_model.pt`
- `numpy` (for .ply → .splat conversion)

## Versioning Note

Existing v1 outputs in `Video/output/` are NOT touched. This pipeline writes to `pipeline/reconstruct/output/`.

## Key Finding: Frame Density > Temporal Smoothing

A/B comparison showed that **extracting more frames** (every frame vs every 3rd) produces much smoother playback than any post-hoc temporal smoothing in Phase 3. The default was changed from `--every-nth 3` to `--every-nth 1` based on this finding.

| Setting | Frames (testclip) | Est. time | Quality |
|---|---|---|---|
| `--every-nth 1` (default) | 103 | ~2.8 hours | Best — smoothest playback |
| `--every-nth 2` | 52 | ~1.4 hours | Good middle ground |
| `--every-nth 3` (old default) | 35 | ~54 min | Noticeable temporal gaps |
