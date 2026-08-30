# Phase 0 Spike: Does Per-Frame SHARP Hold Up on Video?

**Goal:** Answer one question before building anything else — run SHARP independently on every frame of a short clip and see how bad the flicker is, and how many Gaussians a real frame produces. Nothing here is production code; it's meant to be thrown away or rewritten once you know the answer.

---

## On the RTX 3060 Mobile / 6 GB Target

This is a real step down from the 12 GB desktop baseline in the original spec, but it mainly matters for the parts of the plan we've already ruled out — a full 4DGS scene optimization, or a diffusion-based video-depth model. Per-frame SHARP inference is a fast feedforward pass (Apple's own numbers are sub-2-second per image on "a standard GPU"), so it's a different, much lighter workload.

Two things worth doing because of the tighter budget specifically:
- `extract_frames.sh` takes an optional `max_dim` so you can cap frame resolution to whatever's already been working for SHARP on your archival photos, rather than assuming native video resolution is safe.
- `run_sharp_batch.py` runs each frame in its own subprocess so nothing can accumulate across a run of hundreds of frames — correctness over speed for this spike.

---

## Steps

### 1. Pick a clip
One short (5–10 s), well-preserved archival-style clip.

### 2. Extract frames
Start conservatively — lower fps and a capped resolution keep the first run cheap:
```bash
bash extract_frames.sh clip.mp4 ../../output/frames 8 1024
```

### 3. Run SHARP on each frame
Edit `SHARP_CMD_TEMPLATE` and `CONDA_ENV` at the top of `run_sharp_batch.py` to match however you currently call SHARP for a single image in LucidPast, then run it. `every_nth` lets you do an even cheaper first pass (e.g. `3` = every 3rd frame):
```bash
python run_sharp_batch.py ../../output/frames ../../output/splats 3
```
Watch `_spike_log.json` for any frame that times out or errors — worth checking whether those cluster around specific content (motion blur, damaged film).

### 4. View the raw sequence
```bash
cp ../viewer/view_sequence.html ../../output/splats/
cd ../../output/splats
python -m http.server 8000
```
Open `http://localhost:8000/view_sequence.html`. It fetches `manifest.json` (written by the batch script) and cycles through the raw, unsmoothed sequence at `PLAYBACK_FPS` — edit that constant in the file to match your extraction fps.

---

## What You're Looking For

1. Does the geometry visibly jump/flicker frame to frame, or is it stable enough to build the temporal-smoothing module on top of (Phase 3)?
2. Roughly how many Gaussians does a typical frame produce — this sets the real compression budget for Phase 4.
3. Do failures/anomalies in `_spike_log.json` cluster around particular kinds of frames?

---

## Viewer Note

Built on [Spark](https://github.com/sparkjsdev/spark) rather than `GaussianSplats3D` — Spark is the actively maintained one (the GaussianSplats3D maintainer now points people to Spark directly) and is built for good performance on lower-powered/mobile GPUs. It's intentionally bare-bones: no cleanup/disposal of old frame meshes between swaps, no compression — just enough to answer the flicker question. Don't build on this file for the real Phase 5 viewer.
