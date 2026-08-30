# LucidPast — Video Pipeline

Convert 2D archival video into frame-by-frame Gaussian Splat sequences and play them back in the browser or in VR via WebXR.

> **Status:** Phase 0 (Spike) — scripts written, not yet executed. No production pipeline code exists yet. The spike must be run and evaluated before building anything further.

---

## What This Is

The image pipeline in `../LucidPast/` turns a single archival photograph into a 3D Gaussian Splat. This module extends that idea to **video**: extract frames, reconstruct each one independently with Apple SHARP, and play the resulting splat sequence back in real time.

The approach is deliberately conservative — it is a per-frame "flipbook" of Gaussian Splats, **not** a full 4D Gaussian Splatting scene optimization. This keeps the pipeline deterministic, bounded, and runnable on a consumer RTX 3060.

---

## Locked Design Decisions

These were agreed upon before any code was written. They are **not open for re-evaluation** without explicit discussion.

| Decision | Rationale |
|---|---|
| **FOV: own the cone** | Reconstruct only what the camera saw. Void beyond the edges; no fill, no hallucination. |
| **Reconstruction: Apple ml-sharp, per frame** | Reuse the proven SHARP pipeline from the image work. No custom depth + calibration stack. |
| **Assembly: deterministic flipbook** | One `.ply` per frame, played back in sequence. No full 4DGS scene optimization. |
| **Playback: web-only** | Three.js / WebXR. Extend existing open-source viewers (Spark, splaTV, `.splat4d`). No Unity, no Unreal. |
| **Restoration: denoise/deflicker only** | Classical temporal filtering and stabilization. Never generative fill — the same ethical constraint as the image pipeline. |
| **Target hardware** | RTX 3060 12 GB (desktop) or 6 GB (mobile). |

---

## Directory Structure

```
Video/
│
├── README.md                   ← You are here
│
├── phase0-spike/               ← Phase 0: answer "does per-frame SHARP flicker?"
│   ├── scripts/
│   │   ├── extract_frames.sh   ← ffmpeg frame extraction (fps + resolution cap)
│   │   └── run_sharp_batch.py  ← Per-frame SHARP runner with timeout & logging
│   └── viewer/
│       └── view_sequence.html  ← Bare-bones Spark-based splat flipbook viewer
│
├── pipeline/                   ← Phases 1–5: production pipeline (TO BE BUILT)
│   ├── ingest/                 ← Phase 1: ffmpeg extraction, shot-cut detection, deflicker
│   ├── reconstruct/            ← Phase 2: SHARP batch wrapper with validation
│   ├── temporal/               ← Phase 3: optical-flow smoothing (conditional on Phase 0)
│   ├── compress/               ← Phase 4: keyframe + delta encoding, HTTP-streamable
│   └── viewer/                 ← Phase 5: production WebXR viewer with cone boundary
│
├── input/                      ← Drop source video clips here
│
└── output/                     ← Pipeline outputs land here
    ├── frames/                 ← Extracted PNG frames
    └── splats/                 ← Per-frame .ply files + manifest.json
```

---

## Phases

### Phase 0 — Spike ✦ CURRENT PHASE

> **Goal:** Answer one question — how much does per-frame SHARP flicker on video?

**What exists:** Three scripts in `phase0-spike/`. See the [Phase 0 README](phase0-spike/scripts/README.md) below for step-by-step instructions.

**Quick version:**
```bash
# 1. Extract frames from a 5–10s archival clip
bash phase0-spike/scripts/extract_frames.sh clip.mp4 output/frames 8 1024

# 2. Run SHARP on each frame (every 3rd frame for a cheap first pass)
python phase0-spike/scripts/run_sharp_batch.py output/frames output/splats 3

# 3. View the raw sequence
cp phase0-spike/viewer/view_sequence.html output/splats/
cd output/splats && python -m http.server 8000
# Open http://localhost:8000/view_sequence.html
```

**Exit criteria — answer all three before moving on:**
1. Is flicker bad enough to require Phase 3 (temporal smoothing)?
2. How many Gaussians does a typical frame produce? (Sets compression budget.)
3. Do failures cluster around specific frame types? (Motion blur, damaged film, etc.)

---

### Phase 1 — Ingest & Preprocess `pipeline/ingest/`

> **Status:** Not started. Build only after Phase 0 is complete.

- ffmpeg frame extraction (handle any common container/codec)
- Shot-cut detection — classical histogram/edge-based, not a learned model. Each shot becomes its own reconstruction run because a cut is a scene discontinuity.
- Non-generative deflicker (temporal filtering) + stabilization (feature-tracking based)

---

### Phase 2 — Reconstruct `pipeline/reconstruct/`

> **Status:** Not started.

- Production-grade batch wrapper: frame directory in → per-frame `.ply` out
- Handle SHARP's expected input resolution/aspect ratio explicitly — archival footage often deviates from modern norms
- Flag anomalous frames (depth-range outliers) for manual review

---

### Phase 3 — Temporal Consistency `pipeline/temporal/`

> **Status:** Not started. **Conditional on Phase 0 findings.**

- Only build if the spike shows real flicker
- Approach: optical-flow-guided smoothing across frame Gaussian positions — classical, deterministic, no diffusion video-depth models
- If Phase 0 looks acceptable, skip this and note it as a stretch goal

---

### Phase 4 — Compress `pipeline/compress/`

> **Status:** Not started.

- Keyframe + delta encoding across the splat sequence
- Study `.splat4d`'s format before designing anything custom
- Target HTTP-range-streamable output, not full-sequence download

---

### Phase 5 — Viewer `pipeline/viewer/`

> **Status:** Not started.

- Extend GaussianSplats3D's dynamic-scene mode, or fork splaTV / `.splat4d`'s viewer — don't write a WebGL/WebGPU renderer from scratch
- WebXR sitting-VR session — reuse the LucidPast WebXR setup where it fits
- Implement cone boundary: fade or soft-clamp at the edge of reconstructed geometry

---

### Phase 6 — Integration

> **Status:** Not started.

- Single CLI entrypoint: `video in → playable web build out`
- README + one example clip for open-source release

---

## Non-Negotiables (Apply to Every Phase)

- **No generative fill** anywhere in the geometry path
- **Deterministic per-frame reconstruction only**
- **Web-native playback only** — no Unity, no Unreal
- **Target hardware:** RTX 3060 12 GB (desktop baseline)

---

## Key References

| Project | What it is | Why it matters here |
|---|---|---|
| [Apple ml-sharp](https://github.com/apple/ml-sharp) | Single-image → Gaussian Splat | The reconstruction engine for every frame |
| [Spark (sparkjsdev)](https://github.com/sparkjsdev/spark) | WebGL Gaussian Splat renderer | Successor to GaussianSplats3D; used in Phase 0 viewer |
| [splaTV](https://github.com/nicmhd/splaTV) | 4D Gaussian Splat viewer | Reference for temporal playback of splat sequences |
| [.splat4d](https://github.com/nicmhd/splat4d) | Compressed 4D splat format | Reference for Phase 4 compression format |
| `../LucidPast/` | The image pipeline this extends | SHARP setup, WebXR code, config patterns to reuse |

---

## Relationship to the Image Pipeline

This module lives alongside the existing image pipeline:

```
LucidPast - Project/
├── LucidPast/         ← Image pipeline (2D photo → single Gaussian Splat)
├── Video/             ← Video pipeline (2D video → splat sequence) ← YOU ARE HERE
└── Documentation/     ← Design research, case study, mockups
```

The Video pipeline reuses:
- The same SHARP model and environment (`LucidPast/env_sharp/`, `LucidPast/ml-sharp/`)
- The same WebXR session setup patterns from `LucidPast/web/`
- The same ethical constraint: reconstruct only what the camera captured, never fabricate
