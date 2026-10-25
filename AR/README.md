# LucidPast — AR Android Viewer

A native Android AR viewer for the LucidPast Gaussian Splats and splat videos. Point the phone at a horizontal surface, pick an asset from the dock, and it materialises on that plane, locked in place — walk around it to see it from all angles. Adobe Aero for our splats, nothing more: this is a viewer, not an authoring tool.

> **Status:** M0 ✅ · M1 code complete — **M1.1 hotfix in verification** (crash + visual + perf fixes from first device run; see Fix Log). Branch `exploration/ar-android`. Nothing outside `AR/` is modified.

---

## User Flow (v1)

1. App opens → rear camera feed, ARCore scans for horizontal planes.
2. Detected planes are visualized; a 2D dock at the bottom of the screen shows thumbnail cards (10 static splats + 2 splat videos).
3. User taps a card, then taps a detected plane → the asset materialises (fade-in + rise, ~400 ms) on the plane and locks to an ARCore anchor.
4. User walks around the locked asset freely. Pinch to rescale within the volume clamp (see below).
5. Selecting another card replaces the current asset (one at a time).

## Locked Design Decisions

Agreed before any code was written. Not open for re-evaluation without explicit discussion.

| Decision | Choice | Rationale |
|---|---|---|
| **Stack** | Native Kotlin + ARCore + OpenGL ES 3.1 | No engine. Honors the project-wide no-Unity/no-Unreal ethos; plane detection and anchoring come from the ARCore SDK, not a game engine. |
| **Renderer** | Compute-shader counting-sort splat rasterizer | Port of the battle-tested `antimatter15/splat` (MIT) technique — depth-quantize, counting sort, instanced raster — proven on mobile-class GPUs. No renderer research risk. |
| **Assets** | The 10 curated static splats + Phase 2 flipbook videos (`testclip`, `testclip2`), read-only | As scoped by the user. No upload/authoring features in v1. |
| **Delivery** | Copy-to-device folder (`/sdcard/LucidPastAR/`) | Raw assets are multi-GB; APK bundling and LAN streaming were ruled out. The app scans the folder and reads `manifest.json`. |
| **Storage access** | `MANAGE_EXTERNAL_STORAGE` ("All files access") | `.splat` files aren't media — `READ_MEDIA_*` doesn't apply. Fine for a sideloaded exploration app; migrate to SAF (folder picker) only if this ever needs Play distribution. |
| **On-device format** | Standard 32-byte `.splat` records | Byte-for-byte GPU upload, no decoder to write. Decimation happens on PC (see below), not custom compression. Revisit only if storage/perf demands it. |
| **Scale** | Auto-normalized diorama; torso-volume default; pinch clamped to cricket-ball minimum | Explicit user volume spec — see below. |
| **Placement** | One asset at a time, session-scoped anchor | Keeps the render budget to one splat and the v1 scope tight. World-persistent placement is a later exploration. |

## Target Device Profile

| | |
|---|---|
| Phone | Motorola Moto G75 5G |
| SoC / GPU | Snapdragon 6 Gen 3 / Adreno 619 (mid-range) |
| ARCore | Requires Google Play Services for AR — verify certification on first run (the app checks and fails gracefully) |
| Perf budget | ≤ ~300K splats per frame at 30 fps; per-frame buffers ≤ ~10 MB for video swapping |

## Measured Findings (M0 — informs the packer)

SHARP emits a fixed **1,179,648-gaussian grid** per image (~36 MB as `.splat`). Measured alpha distribution on `testclip` frame 1:

| Alpha threshold | Splats kept | % kept |
|---|---|---|
| ≥ 1 | 1,177,199 | 99.8% |
| ≥ 32 | 1,156,054 | 98.0% |
| ≥ 128 | 1,111,457 | 94.2% |

**Opacity culling is a dead end** — unlike trained 3DGS scenes, SHARP's grid is dense and every splat is visible. The packer therefore uses **stride subsampling with scale-inflation compensation**: keep every Nth splat, multiply the remaining scales by N so coverage is preserved (slight softening, no pinholes). Profiles: statics at stride 2 (~590K splats, single render — quality first), videos at stride 4 (~294K splats/frame — swap budget for the mid-range GPU).

## Packed Asset Sizes (statics stride 2, videos stride 4)

| Asset class | Count | Per asset | Splats | Total |
|---|---|---|---|---|
| Static splats | 10 | ~18.9 MB | ~590K | ~189 MB |
| `testclip` video | 35 frames | ~9.4 MB/frame | ~294K | ~330 MB |
| `testclip2` video | 22 frames | ~9.4 MB/frame | ~294K | ~207 MB |
| **Total on device** | | | | **~725 MB** |

> Note: `testclip2` is moving-camera dashcam footage — its raw bbox spans ~55×27×70 m due to warp outliers. The manifest therefore also carries 1st/99th-percentile bboxes (`bbox_p01`/`bbox_p99`), which the app uses for robust diorama normalization.

## Volume Spec

A splat is a captured scene — placed in AR it reads as a miniature volumetric diorama.

| Bound | Definition |
|---|---|
| **Default (max): human torso** | Splat bbox normalized to fit a 0.45 m (W) × 0.30 m (D) × 0.45 m (H) envelope (~61 L, torso-presence scale) |
| **Minimum: cricket ball** | Pinch-down clamped so the envelope shrinks to a ~0.07 m box (cricket ball: d ≈ 0.072 m) |
| Interaction | Pinch scale factor clamped to [0.15, 1.0]× of the default envelope |

## Milestones

| ID | Scope | Status |
|---|---|---|
| **M0** | Asset pack: `pack_assets.py` (decimation, thumbnails, manifest) + format spec | ✅ Complete |
| **M1** | On-device splat renderer (GL ES 3.1, orbit camera, no AR yet) — proves Adreno 619 perf | ✅ Code complete — pending device verification |
| **M2** | ARCore: plane detection + visualization, 2D dock, tap-to-place, anchor lock, walk-around | Next |
| **M3** | Splat video playback: flipbook loop at manifest fps (frame-swap via io thread + GL queueEvent; pause-aware token cancels pending swaps). Ring-buffer double-buffering + compression = M3.1 | ✅ First cut |
| **M4** | Polish: materialise animation, pinch-scale clamp, error/permission states | |

## Directory Structure

```
AR/
├── README.md                    ← you are here
├── preprocessing/
│   ├── pack_assets.py           ← PC-side packer (numpy + optional Pillow)
│   └── output/
│       └── LucidPastAR/         ← generated; copy this folder to the phone
│           ├── manifest.json
│           ├── statics/<id>.splat
│           ├── videos/<clip>/frame_%05d.splat
│           └── thumbs/<id>.jpg
└── android/                     ← M1: native Kotlin app (Android Studio project)
    ├── app/src/main/java/com/lucidpast/ar/
    │   ├── MainActivity.kt      ← permission flow, dock, touch, debug overlay
    │   ├── Assets.kt            ← manifest parsing, .splat IO, diorama model matrix
    │   ├── ui/DockAdapter.kt    ← bottom thumbnail carousel
    │   └── render/
    │       ├── Shaders.kt       ← GLSL ES 3.1: counting-sort kernels + EWA quad splats
    │       ├── SplatRenderer.kt ← SSBO pipeline: count → scan → scatter → draw
    │       └── OrbitCamera.kt   ← drag/pinch camera (M2 swaps in the ARCore pose)
    └── ...
```

## Renderer Architecture (M1.1 / M1.2)

The splat pipeline per sort (only re-run when the view changes):

1. **Count** (compute): each splat → view-space distance → inverted 16-bit depth key → `atomicAdd` into 65,536 buckets.
2. **Scan** (compute ×3): two-level Hillis-Steele prefix sum over the buckets → exclusive prefix.
3. **Scatter** (compute): `slot = prefix[key] + atomicAdd(cursor)` → back-to-front index buffer (bounds-guarded).
4. **Prepare** (compute): per splat — project, transform covariance by `mat3(uModel)` (scale!), EWA to 2D pixel covariance, eigen-axes, conic — packed into a 64B record (4×vec4: center | axes | conic+alpha | rgb+alpha) in sorted order. Color travels as plain floats — no bit-trick payloads. Clamps splat screen radius to 256 px (overdraw guardrail).
5. **Draw** (vertex+frag): `glDrawArraysInstanced(TRIANGLE_STRIP, 0, 4, n)` — the vertex shader is a linear read + corner expansion; the fragment evaluates the gaussian falloff with premultiplied `over` blending.

Heavy per-splat math runs **once per sort in compute** (not 4× per frame in the vertex stage) — the M1.1 refactor that took diner from ~12 fps toward 30+. Technique follows the MIT-licensed `antimatter15/splat` counting-sort approach, implemented natively for ES 3.1.

## M1.1 Fix Log (first device run: Moto G75, first build)

Symptoms: only `01_diner_1940` and `03_radio_studio_1942` rendered (badly, ~10-12 fps); everything else appeared to crash while loading; content didn't look like the browser viewer.

Root causes found (measured via manifest percentile bboxes — SHARP's raw units are per-photo, so the diorama scale factor `s = 0.45/extent` varies by 400× across assets):

| Asset | Raw extent (p01-p99) | s | Effect of the bug |
|---|---|---|---|
| diner_1940 | 0.25 | 1.795 | Splats 1.8× undersized → mushy image |
| radio_studio | 0.44 | 1.016 | Nearly correct — why it "worked" |
| migrant | 84.8 (z) | 0.0053 | Splats ~190× oversized → monster quads |
| mlk | 102 (z) | 0.0044 | Same → GPU watchdog kill on first frame |
| aldrin / tesla / gandhi | 4–16 | 0.03–0.11 | Same, to a lesser degree |

1. **Covariance not transformed by the model matrix (the crash).** Positions got `uModel` (scale `s` + axis flip); covariances didn't. For `s` far from 1 the ellipses were wrong-sized by up to 200× — clamped to 512 px quads, 590K of them → multi-second frames → watchdog. Fixed in the Prepare pass: `Σ_world = M3 Σ M3ᵀ` with `M3 = mat3(uModel)`.
2. **Off-axis default camera (the "not understandable" look).** The orbit started at yaw 0.6 / pitch 0.3 — viewing the capture cone from the side, which is a smear by nature. The browser viewer starts at the capture viewpoint. Fixed: on load the camera resets to yaw=0/pitch=0 at the per-asset capture distance (`s · |center_z|`, all assets 0.23-0.41 m) — the photo view, exactly like the browser. Orbit/pinch out to inspect the diorama.
3. **Per-vertex covariance recomputation (the 10-12 fps).** 4 verts/splat × 2.36M invocations, each doing scattered SSBO reads + full EWA math. Fixed by the Prepare-pass refactor above.
4. **Hardening:** scatter writes are bounds-guarded (a key-recompute mismatch could have written past the order SSBO → native crash); quad radius clamped to 256 px (watchdog-proof); near plane 0.02 so the capture view isn't front-clipped; pinch min clamp lowered to 0.15 m.

## M1 device verification checklist (v2 — standard test: `02_migrant`, the previous failure case)

- [ ] Migrant Mother loads without crash and shows the actual photograph (not a smear) on load — this is the capture view
- [ ] All 10 statics switch cleanly; no crash on rapid switching (aldrin/tesla/mlk were the watchdog cases)
- [ ] One-finger drag orbits away from the photo view; pinch works; return-to-front still looks like the photo
- [ ] Debug readout: fps ≥ 30 static camera, sort ms < 25 while dragging, load ms < 600
- [ ] Video entries (`VID` badge) load frame 0; `testclip` reads as a whole diorama from the 1.2 m fit view at fps ≥ 30 (was 1 fps at M1.1); `testclip2` also frames cleanly (percentile bbox at work)
- [ ] Diner still renders sharper than the first build (s=1.795 now inflates its covariance correctly)

**Remaining known risk:** GLSL conventions (quaternion byte order, y-flip) are standard but only device-verifiable — a mirrored image means one sign flip in `SplatIO.modelMatrix`.

## M1.2 Fix Log (second device run: black screen, testclip 1 fps)

Symptoms: no crash; dock + debug overlay render; the GL surface shows only the clear color (black). Debug readout alive and plausible (fps/splats/sort/load all update). `testclip` at ~1 fps, statics at 58-60 fps.

Diagnosis: the fps signature matched the simulated raster cost exactly — `testclip` ≈ 1830× fullscreen overdraw (median splat ≈ 200 px, capture camera sits inside the flipbook cloud), diner ≈ 1.8×, migrant ≈ 22× (right at the Adreno 619 fill ceiling, hence 58 fps). So positions, sizes and alphas were being rasterized correctly on-device; only the color channel was black. The one non-portable construct in the pipeline: the packed RGBA uint32 was transported through the prepared SSBO as raw float bits (`uintBitsToFloat` → `floatBitsToUint`). For typical packed colors (alpha ≈ 255, blue ≥ 128) that bit pattern is a NaN / huge float — Adreno 6xx compilers flush such values in the store path, so `floatBitsToUint(0)` returned 0 → `rgb = (0,0,0)` with alpha intact → opaque black splats composited at full raster cost. Every symptom (black image, correct fps, no crash, all assets affected) follows from this.

Fixes:

1. **Prepared record 48B → 64B** (4×vec4): center | axes | conic+alpha | rgb+alpha. Color now travels as plain floats; no bit-trick payloads anywhere in the prepared stream. (`uintBitsToFloat` remains only on the RECORDS buffer where it decodes genuine float32 data.)
2. **`glMemoryBarrier(GL_ALL_BARRIER_BITS)`** after the Prepare dispatch (Adreno insurance for vertex-stage SSBO visibility; sort is rare so the cost is nil).
3. **Fragment gaussian cross-term** was half-strength (`x·y·c` instead of `2·x·y·c`) — splats rendered slightly boxy/wide along the minor axis.
4. **Video framing:** video assets now open at a diorama-fit distance (1.2 m) instead of the capture point. The capture point sits at the raw camera position — for flipbook frames the cloud front is only ~4.5 cm in front of it, so at the capture view every splat is ~200 px on a 2400 px screen (~1830× overdraw → ~1 fps). At 1.2 m: ~28× overdraw, ~45+ fps (meets the ≤300K splats @ 30 fps budget). Statics keep the capture view — locked design decision; their cloud depth is thin so the capture point is outside it (migrant 22×/58 fps is the ceiling case and acceptable).

Implication for M3 (video playback): the per-frame draw cost is now view-distance-bound, not data-bound. The remaining challenge is the swap: 294K splats × 48B ≈ 14 MB per frame at 10 fps ≈ 140 MB/s of SSBO uploads. That is where compression research pays off — temporal delta encoding between flipbook frames first, full 4DGS (per-splat motion vectors, interpolate ~35 frames from ~8 keyframes) as the research-grade option. Disk format stays as-is until then.

## Build & Run (M1)

```bash
# 1. Pack assets (if not already done — see "Running the Packer")
python AR/preprocessing/pack_assets.py

# 2. Copy to phone (USB, file transfer mode) OR:
adb push AR/preprocessing/output/LucidPastAR /sdcard/LucidPastAR

# 3. Open AR/android in Android Studio (Hedgehog or newer)
#    - let it sync (downloads Gradle 8.2 + SDK 34 on first sync)
#    - Build → Run on the Moto G75 (USB debugging on)

# CLI alternative (needs JDK 17 in JAVA_HOME):
cd AR/android && gradlew.bat assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

First launch: grant **All files access** when prompted (or Settings → All files access → LucidPast AR), then reopen the app. Verification checklist is in the M1.1 section above.

## Running the Packer

```bash
# Any python with numpy (Pillow optional, for thumbnails):
python AR/preprocessing/pack_assets.py

# Quick smoke test (1 static, 2 video frames, coarse stride):
python AR/preprocessing/pack_assets.py --limit-statics 1 --limit-frames 2 --stride 4

# Custom output location / playback rate:
python AR/preprocessing/pack_assets.py --out AR/preprocessing/output/LucidPastAR --fps 10
```

Reads (never writes): `output/gaussians/` (statics, via PLY), `Video/pipeline/reconstruct/output/` (Phase 2 frames, via `.splat`), `input/` + `Video/pipeline/ingest/output/` (thumbnails).

## Manifest Schema (v1)

```json
{
  "format_version": 1,
  "stride": { "static": 2, "video": 4 },
  "statics": [
    { "id": "01_diner_1940", "file": "statics/01_diner_1940.splat",
      "splat_count": 589824,
      "bbox_min": [x,y,z], "bbox_max": [x,y,z], "bbox_center": [x,y,z],
      "bbox_p01": [x,y,z], "bbox_p99": [x,y,z],
      "thumb": "thumbs/01_diner_1940.jpg" }
  ],
  "videos": [
    { "id": "testclip", "fps": 10.0, "frame_count": 35,
      "frame_pattern": "videos/testclip/frame_%05d.splat",
      "splat_count_per_frame": 294207,
      "bbox_min": [x,y,z], "bbox_max": [x,y,z], "bbox_center": [x,y,z],
      "bbox_p01": [x,y,z], "bbox_p99": [x,y,z],
      "thumb": "thumbs/testclip.jpg" }
  ]
}
```

The app centers each asset at the (percentile) bbox center — dropped to the plane — and scales by the percentile bbox extent per the volume spec.

## Relationship to Other Modules

```
LucidPast - Project/
├── LucidPast/    ← image pipeline (this module's parent repo)
├── Video/        ← video pipeline (source of the Phase 2 flipbook frames)
└── AR/           ← this module — branch exploration/ar-android
```

This module reuses existing pipeline **outputs only**. It touches no code in `web/`, `scripts/`, or `Video/`, and honors the same ethical constraint: it displays what was captured, inventing nothing.
