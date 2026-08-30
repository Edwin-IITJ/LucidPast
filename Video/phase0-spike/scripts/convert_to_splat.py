#!/usr/bin/env python3
"""
Convert SHARP's custom PLY Gaussian Splat files to the standard .splat
format (antimatter15/splat) that web viewers like Spark can load.

SHARP PLY has extra metadata elements (extrinsic, intrinsic, etc.) that
confuse most web-based splat renderers. The .splat format is a compact
32-byte-per-gaussian binary that every major web viewer supports.

Usage:
    python convert_to_splat.py <input_dir> <output_dir>

Converts all .ply files in input_dir to .splat files in output_dir,
and writes an updated manifest.json.
"""

import json
import struct
import sys
import numpy as np
from pathlib import Path


# SH0 coefficient for converting f_dc to RGB
C0 = 0.28209479177387814


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def read_sharp_ply(ply_path: Path) -> dict:
    """Read a SHARP PLY file, extracting only the vertex (Gaussian) data."""
    with open(ply_path, "rb") as f:
        # Parse header
        header_lines = []
        while True:
            line = f.readline().decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        # Find vertex count
        vertex_count = 0
        for line in header_lines:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
                break

        if vertex_count == 0:
            raise ValueError(f"No vertices found in {ply_path}")

        # Each vertex has 14 float32 properties = 56 bytes
        # x, y, z, f_dc_0, f_dc_1, f_dc_2, opacity,
        # scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3
        vertex_size = 14 * 4  # 14 floats × 4 bytes
        vertex_data = f.read(vertex_count * vertex_size)

    # Parse into structured array
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
        ("opacity", "<f4"),
        ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
        ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
    ])

    vertices = np.frombuffer(vertex_data, dtype=dtype, count=vertex_count)
    return vertices


def convert_to_splat(vertices: np.ndarray) -> bytes:
    """
    Convert parsed Gaussian vertices to the .splat binary format.

    .splat format per Gaussian (32 bytes):
        position:   3 × float32  (12 bytes)
        scale:      3 × float32  (12 bytes)
        rgba:       4 × uint8    (4 bytes)
        quaternion: 4 × uint8    (4 bytes)
    """
    n = len(vertices)
    buf = bytearray(n * 32)

    # Positions (float32)
    positions = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype("<f4")

    # Scales — apply exp (SHARP stores log-scale)
    scales = np.column_stack([
        np.exp(vertices["scale_0"]),
        np.exp(vertices["scale_1"]),
        np.exp(vertices["scale_2"]),
    ]).astype("<f4")

    # Colors — convert SH DC coefficients to RGB [0..255]
    r = np.clip((0.5 + C0 * vertices["f_dc_0"]) * 255, 0, 255).astype(np.uint8)
    g = np.clip((0.5 + C0 * vertices["f_dc_1"]) * 255, 0, 255).astype(np.uint8)
    b = np.clip((0.5 + C0 * vertices["f_dc_2"]) * 255, 0, 255).astype(np.uint8)

    # Opacity — apply sigmoid, then to [0..255]
    a = np.clip(sigmoid(vertices["opacity"]) * 255, 0, 255).astype(np.uint8)

    rgba = np.column_stack([r, g, b, a])

    # Quaternion — normalize then map to [0..255] centered at 128
    quats = np.column_stack([
        vertices["rot_0"], vertices["rot_1"],
        vertices["rot_2"], vertices["rot_3"],
    ])
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    quats = quats / norms
    quat_u8 = np.clip((quats * 128) + 128, 0, 255).astype(np.uint8)

    # Pack into buffer
    for i in range(n):
        offset = i * 32
        struct.pack_into("<3f", buf, offset, *positions[i])
        struct.pack_into("<3f", buf, offset + 12, *scales[i])
        buf[offset + 24:offset + 28] = rgba[i].tobytes()
        buf[offset + 28:offset + 32] = quat_u8[i].tobytes()

    return bytes(buf)


def convert_to_splat_fast(vertices: np.ndarray) -> bytes:
    """Vectorized version — much faster for 1M+ Gaussians."""
    n = len(vertices)

    positions = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype("<f4")

    scales = np.column_stack([
        np.exp(vertices["scale_0"]),
        np.exp(vertices["scale_1"]),
        np.exp(vertices["scale_2"]),
    ]).astype("<f4")

    r = np.clip((0.5 + C0 * vertices["f_dc_0"]) * 255, 0, 255).astype(np.uint8)
    g = np.clip((0.5 + C0 * vertices["f_dc_1"]) * 255, 0, 255).astype(np.uint8)
    b = np.clip((0.5 + C0 * vertices["f_dc_2"]) * 255, 0, 255).astype(np.uint8)
    a = np.clip(sigmoid(vertices["opacity"]) * 255, 0, 255).astype(np.uint8)
    rgba = np.column_stack([r, g, b, a])

    quats = np.column_stack([
        vertices["rot_0"], vertices["rot_1"],
        vertices["rot_2"], vertices["rot_3"],
    ])
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    quats = quats / norms
    quat_u8 = np.clip((quats * 128) + 128, 0, 255).astype(np.uint8)

    # Build the output as a structured array for zero-copy packing
    out_dtype = np.dtype([
        ("pos", "<f4", 3),
        ("scale", "<f4", 3),
        ("rgba", "u1", 4),
        ("quat", "u1", 4),
    ])
    out = np.zeros(n, dtype=out_dtype)
    out["pos"] = positions
    out["scale"] = scales
    out["rgba"] = rgba
    out["quat"] = quat_u8

    return out.tobytes()


def main():
    if len(sys.argv) < 3:
        print("Usage: convert_to_splat.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    ply_files = sorted(input_dir.glob("frame_*.ply"))
    if not ply_files:
        print(f"No frame_*.ply files found in {input_dir}")
        sys.exit(1)

    print(f"Converting {len(ply_files)} PLY files to .splat format...")

    manifest = []
    for i, ply_path in enumerate(ply_files):
        splat_name = f"{ply_path.stem}.splat"
        splat_path = output_dir / splat_name

        print(f"  [{i+1}/{len(ply_files)}] {ply_path.name} -> {splat_name}...", end=" ", flush=True)

        vertices = read_sharp_ply(ply_path)
        splat_data = convert_to_splat_fast(vertices)
        splat_path.write_bytes(splat_data)

        size_mb = len(splat_data) / (1024 * 1024)
        print(f"ok ({len(vertices)} gaussians, {size_mb:.1f} MB)")
        manifest.append(splat_name)

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDone. manifest.json written with {len(manifest)} entries.")


if __name__ == "__main__":
    main()
