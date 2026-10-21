# Phase 3 — Temporal Consistency

> **Status: COMPLETE (v2).** Shadow-only mode is the recommended default. Color EMA available but disabled by default.

## Scope

Smooth Gaussian attributes across consecutive frames to reduce temporal flicker, and suppress transient shadow blob artifacts.

## Key Finding

**Frame density is the primary quality lever.** A/B comparison across v1 (full smoothing) and v2 (attribute-only) showed that:

1. **Position smoothing destroys geometry** — especially for moving cameras where each frame's Gaussians are in different camera-space coordinates
2. **Color EMA washes out saturation** — even at gentle α=0.3, colors become noticeably less vibrant
3. **More frames from Phase 2 (`--every-nth 1`) produces smoother playback** than any post-hoc temporal smoothing

The default configuration was changed to: **shadow blob suppression only, no color/opacity EMA**.

## Usage

```bash
# Default: shadow suppression only (recommended)
python temporal.py ../reconstruct/output/testclip2

# With gentle color EMA (if needed):
python temporal.py ../reconstruct/output/testclip2 --color-alpha 0.2 --opacity-alpha 0.1

# Skip Phase 3 entirely (if frame density is high enough):
# Just use Phase 2 output directly in the viewer
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--color-alpha` | `0.0` | EMA weight for color smoothing (0 = off) |
| `--opacity-alpha` | `0.0` | EMA weight for opacity smoothing (0 = off) |
| `--opacity-threshold` | `0.3` | Shadow suppression opacity deviation threshold |
| `--dark-threshold` | `0.15` | Shadow suppression luminance threshold |
| `--no-suppress` | off | Disable shadow blob suppression |
| `--no-convert` | off | Skip `.ply` → `.splat` conversion |
| `--output-dir` | auto | Override output directory |

## Version History

### v2 (current) — Attribute-Only EMA + Shadow Suppression
- Geometry (positions, scale, rotation) is **FROZEN** — passed through from SHARP unchanged
- Color/opacity EMA disabled by default (was washing out colors)
- Shadow blob suppression: detects transient dark+opaque Gaussians and clamps to temporal median

### v1 (deprecated) — Full Smoothing
- Averaged ALL attributes (positions, color, opacity, scale, rotation) across Gaussian-weighted temporal windows
- **Result:** Destroyed geometry for moving camera footage (testclip2 became unrecognizable)
- **Lesson:** Position smoothing is fundamentally wrong for per-frame single-image reconstruction where each frame is in its own camera coordinate system

## Phase 0 Findings (Decision Point — CLOSED)

Both test clips confirmed severe flicker. Phase 3 was mandatory.

- [x] **Flicker severity:** Three-tier instability observed in testclip2:
  - **Tier 1** — Edge scatter: present in both clips
  - **Tier 2** — Shadow instability: dark blobs on people/vehicles (color footage)
  - **Tier 3** — Full-scene warp: entire geometry shifts when camera moves
- [x] **Decision: BUILD** — Full-scene scope required, not edge-only
- [x] **Revised conclusion:** Shadow suppression addresses Tier 2. Tiers 1 & 3 are better addressed by higher frame density in Phase 2.

## Key Findings from Spike & Testing

| Metric | testclip (B&W, static) | testclip2 (color, moving) |
|---|---|---|
| Blobs suppressed (v2) | 60,097 | 80,505 |
| v1 quality | Slightly blurred | **Destroyed** — unrecognizable |
| v2 quality | Matches Phase 2 raw | Matches Phase 2 raw |
| Color EMA effect | Mild washout | Noticeable desaturation |

> [!IMPORTANT]
> Phase 3 scope was revised after testing. Shadow suppression only is the recommended default.
> The unsmoothed outputs in Phase 2 are preserved for comparison.
