"""
LucidPast AR — Asset Packer (Milestone 0)

Packs the existing LucidPast Gaussian Splat assets into a mobile-ready folder
for the AR Android viewer:

  - Static splats:  output/gaussians/<id>/<id>.ply      (10 curated assets)
  - Splat videos:   Video/pipeline/reconstruct/output/<clip>/splats/frame_*.splat
                    (Phase 2 flipbook frames: testclip, testclip2)

SHARP emits a fixed 1,179,648-gaussian grid per image (~36 MB as .splat), too
heavy for a mid-range phone GPU (Adreno 619) — and opacity culling does not
help (measured: 94% of splats have alpha >= 128). The packer therefore uses
stride subsampling with scale-inflation compensation: keep every Nth splat and
multiply the remaining scales by N so coverage is preserved.

Output (default): AR/preprocessing/output/LucidPastAR/
  manifest.json                  — asset index consumed by the Android app
  statics/<id>.splat            — standard 32-byte .splat records
  videos/<clip>/frame_%05d.splat
  thumbs/<id>.jpg               — dock thumbnails (source photos / first frames)

Copy the LucidPastAR folder to the phone (e.g. /sdcard/LucidPastAR); the app
scans it at startup. Existing pipeline outputs are read, never modified.

Usage:
  python AR/preprocessing/pack_assets.py [--stride 2] [--video-stride 4]
      [--fps 10.0] [--out AR/preprocessing/output/LucidPastAR]
      [--limit-statics N] [--limit-frames N] [--skip-thumbs]
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STATICS_DIR = ROOT / "output" / "gaussians"
VIDEO_DIR = ROOT / "Video" / "pipeline" / "reconstruct" / "output"
INGEST_FRAMES_DIR = ROOT / "Video" / "pipeline" / "ingest" / "output"
INPUT_DIR = ROOT / "input"

C0 = 0.28209479177387814
SPLAT_DTYPE = np.dtype([
    ("pos", "<f4", 3), ("scale", "<f4", 3),
    ("rgba", "u1", 4), ("quat", "u1", 4),
])


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def read_sharp_ply(ply_path: Path) -> np.ndarray:
    """Read a SHARP PLY file, extracting only the vertex (Gaussian) data.

    Assumes SHARP's fixed vertex layout: 14 float32 properties = 56 bytes.
    """
    with open(ply_path, "rb") as f:
        vertex_count = 0
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            if line == "end_header":
                break
        vertex_data = f.read(vertex_count * 56)

    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
        ("opacity", "<f4"),
        ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
        ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
    ])
    return np.frombuffer(vertex_data, dtype=dtype, count=vertex_count)


def ply_to_splat_records(vertices: np.ndarray) -> np.ndarray:
    """Convert SHARP PLY vertices to standard 32-byte .splat records.

    Same math as Video/pipeline/reconstruct/reconstruct.py (proven converter).
    """
    n = len(vertices)
    records = np.zeros(n, dtype=SPLAT_DTYPE)
    records["pos"] = np.column_stack(
        [vertices["x"], vertices["y"], vertices["z"]]
    ).astype("<f4")
    records["scale"] = np.column_stack([
        np.exp(vertices["scale_0"]),
        np.exp(vertices["scale_1"]),
        np.exp(vertices["scale_2"]),
    ]).astype("<f4")

    r = np.clip((0.5 + C0 * vertices["f_dc_0"]) * 255, 0, 255).astype(np.uint8)
    g = np.clip((0.5 + C0 * vertices["f_dc_1"]) * 255, 0, 255).astype(np.uint8)
    b = np.clip((0.5 + C0 * vertices["f_dc_2"]) * 255, 0, 255).astype(np.uint8)
    a = np.clip(_sigmoid(vertices["opacity"]) * 255, 0, 255).astype(np.uint8)
    records["rgba"] = np.column_stack([r, g, b, a])

    quats = np.column_stack([
        vertices["rot_0"], vertices["rot_1"],
        vertices["rot_2"], vertices["rot_3"],
    ])
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    records["quat"] = np.clip((quats / norms) * 128 + 128, 0, 255).astype(np.uint8)
    return records


def read_splat(splat_path: Path) -> np.ndarray:
    """Read a .splat file directly into structured records."""
    return np.fromfile(splat_path, dtype=SPLAT_DTYPE)


def decimate_records(records: np.ndarray, stride: int) -> np.ndarray:
    """Stride subsample + scale-inflation so coverage is preserved."""
    if stride <= 1:
        kept = records[records["rgba"][:, 3] > 0]
        return np.ascontiguousarray(kept)
    kept = records[::stride]
    kept = kept[kept["rgba"][:, 3] > 0]
    kept = np.ascontiguousarray(kept)
    kept["scale"] *= float(stride)
    return kept


def bbox_of(records: np.ndarray):
    pos = records["pos"]
    bmin = [round(float(v), 4) for v in pos.min(axis=0)]
    bmax = [round(float(v), 4) for v in pos.max(axis=0)]
    center = [round(float(v), 4) for v in (pos.min(axis=0) + pos.max(axis=0)) / 2.0]
    p01 = [round(float(v), 4) for v in np.percentile(pos, 1, axis=0)]
    p99 = [round(float(v), 4) for v in np.percentile(pos, 99, axis=0)]
    return bmin, bmax, center, p01, p99


def make_thumbnail(src: Path, dst: Path, size: int = 256) -> bool:
    """Downscale an image to a JPEG dock thumbnail. Returns False if unavailable."""
    try:
        from PIL import Image
    except ImportError:
        return False
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((size, size))
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, "JPEG", quality=82)
        return True
    except Exception as e:
        print(f"    thumbnail skipped ({e})")
        return False


def find_static_assets() -> list[Path]:
    """The 10 curated asset dirs (numbered), each containing <id>.ply."""
    assets = []
    if not STATICS_DIR.exists():
        return assets
    for d in sorted(STATICS_DIR.iterdir()):
        if not d.is_dir() or not re.match(r"^\d{2}_", d.name):
            continue
        ply = d / f"{d.name}.ply"
        if ply.exists():
            assets.append(ply)
    return assets


def find_video_clips() -> dict[str, list[Path]]:
    """Phase 2 clips: {clip_id: sorted frame .splat paths}."""
    clips = {}
    if not VIDEO_DIR.exists():
        return clips
    for d in sorted(VIDEO_DIR.iterdir()):
        if not d.is_dir():
            continue
        frames = sorted((d / "splats").glob("frame_*.splat"))
        if frames:
            clips[d.name] = frames
    return clips


def static_thumbnail_source(asset_id: str):
    for ext in (".tif", ".tiff", ".jpg", ".jpeg", ".png"):
        cand = INPUT_DIR / f"{asset_id}{ext}"
        if cand.exists():
            return cand
    return None


def video_thumbnail_source(clip_id: str):
    frames_dir = INGEST_FRAMES_DIR / clip_id / "frames"
    if not frames_dir.exists():
        return None
    pngs = sorted(frames_dir.glob("*.png"))
    return pngs[0] if pngs else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pack LucidPast splats into a mobile-ready AR asset folder."
    )
    parser.add_argument("--stride", type=int, default=2,
                        help="Subsampling stride for static splats (default: 2)")
    parser.add_argument("--video-stride", type=int, default=4,
                        help="Subsampling stride for video frames (default: 4)")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Splat video playback rate written to manifest (default: 10)")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "AR" / "preprocessing" / "output" / "LucidPastAR",
                        help="Output folder (copied to the phone)")
    parser.add_argument("--limit-statics", type=int, default=0,
                        help="Pack only the first N statics (0 = all)")
    parser.add_argument("--limit-frames", type=int, default=0,
                        help="Pack only the first N frames per video (0 = all)")
    parser.add_argument("--skip-thumbs", action="store_true",
                        help="Skip thumbnail generation")
    args = parser.parse_args()

    video_stride = args.video_stride if args.video_stride is not None else args.stride
    if args.stride < 1 or video_stride < 1:
        parser.error("stride must be >= 1")

    statics = find_static_assets()
    clips = find_video_clips()
    if not statics and not clips:
        print("No assets found. Run this from the LucidPast repo root.")
        return 1
    if args.limit_statics > 0:
        statics = statics[: args.limit_statics]
    if args.limit_frames > 0:
        clips = {cid: frames[: args.limit_frames] for cid, frames in clips.items()}

    statics_out = args.out / "statics"
    videos_out = args.out / "videos"
    thumbs_out = args.out / "thumbs"
    statics_out.mkdir(parents=True, exist_ok=True)
    videos_out.mkdir(parents=True, exist_ok=True)

    manifest_statics = []
    manifest_videos = []
    total_bytes = 0

    print(f"LucidPast AR packer - stride {args.stride} (video {video_stride})")
    print(f"Output: {args.out}\n")

    print("== Static splats ==")
    for ply in statics:
        asset_id = ply.parent.name
        print(f"  {asset_id}: reading PLY ...", end=" ", flush=True)
        records = ply_to_splat_records(read_sharp_ply(ply))
        kept = decimate_records(records, args.stride)
        out_path = statics_out / f"{asset_id}.splat"
        out_path.write_bytes(kept.tobytes())
        total_bytes += out_path.stat().st_size
        bmin, bmax, center, p01, p99 = bbox_of(kept)
        thumb = None
        if not args.skip_thumbs:
            src = static_thumbnail_source(asset_id)
            if src is not None:
                tpath = thumbs_out / f"{asset_id}.jpg"
                if make_thumbnail(src, tpath):
                    thumb = f"thumbs/{tpath.name}"
        manifest_statics.append({
            "id": asset_id,
            "file": f"statics/{out_path.name}",
            "splat_count": int(len(kept)),
            "bbox_min": bmin, "bbox_max": bmax, "bbox_center": center,
            "bbox_p01": p01, "bbox_p99": p99,
            "thumb": thumb,
        })
        print(f"{len(records):,} -> {len(kept):,} splats "
              f"({out_path.stat().st_size / 1e6:.1f} MB)")

    print("\n== Splat videos (Phase 2 frames) ==")
    for clip_id, frames in clips.items():
        print(f"  {clip_id}: {len(frames)} frames, stride {video_stride}")
        clip_dir = videos_out / clip_id
        clip_dir.mkdir(parents=True, exist_ok=True)
        per_frame = None
        bmin = bmax = center = None
        for i, frame_path in enumerate(frames):
            records = read_splat(frame_path)
            kept = decimate_records(records, video_stride)
            out_path = clip_dir / (f"frame_{i:05d}.splat")
            out_path.write_bytes(kept.tobytes())
            total_bytes += out_path.stat().st_size
            if i == 0:
                per_frame = len(kept)
                bmin, bmax, center, p01, p99 = bbox_of(kept)
            print(f"    {frame_path.name} -> {out_path.name} "
                  f"({len(kept):,} splats, {out_path.stat().st_size / 1e6:.1f} MB)")
        thumb = None
        if not args.skip_thumbs:
            src = video_thumbnail_source(clip_id)
            if src is not None:
                tpath = thumbs_out / f"{clip_id}.jpg"
                if make_thumbnail(src, tpath):
                    thumb = f"thumbs/{tpath.name}"
        manifest_videos.append({
            "id": clip_id,
            "fps": args.fps,
            "frame_count": len(frames),
            "frame_pattern": f"videos/{clip_id}/frame_%05d.splat",
            "splat_count_per_frame": per_frame,
            "bbox_min": bmin, "bbox_max": bmax, "bbox_center": center,
            "bbox_p01": p01, "bbox_p99": p99,
            "thumb": thumb,
        })

    manifest = {
        "format_version": 1,
        "created": datetime.now().isoformat(timespec="seconds"),
        "stride": {"static": args.stride, "video": video_stride},
        "statics": manifest_statics,
        "videos": manifest_videos,
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nDone: {len(manifest_statics)} statics, {len(manifest_videos)} videos, "
          f"{total_bytes / 1e6:.0f} MB total")
    print(f"Manifest: {manifest_path}")
    print(f"Next: copy '{args.out}' to the phone (e.g. /sdcard/LucidPastAR)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
