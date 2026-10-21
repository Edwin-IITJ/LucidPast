# Phase 3 steps — temporal consistency
import sys
import io

# Force UTF-8 output on Windows terminals that default to cp1252
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
from pathlib import Path

# ─── Shared PLY I/O ──────────────────────────────────────────────────────────
# SHARP PLY format: 14 float32 properties per Gaussian + metadata elements

GAUSSIAN_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
    ("opacity", "<f4"),
    ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
    ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
])

VERTEX_SIZE = 14 * 4  # 56 bytes per Gaussian


def read_ply(ply_path: Path) -> tuple[np.ndarray, bytes]:
    """
    Read a SHARP PLY file.

    Returns
    -------
    vertices : np.ndarray
        Structured array of Gaussians.
    metadata : bytes
        Everything after the vertex data (extrinsic, intrinsic, etc.)
        preserved verbatim so we can write it back.
    """
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

        # Read vertex data
        vertex_data = f.read(vertex_count * VERTEX_SIZE)
        # Read remaining metadata
        metadata = f.read()

    vertices = np.frombuffer(vertex_data, dtype=GAUSSIAN_DTYPE, count=vertex_count).copy()
    return vertices, metadata


def write_ply(ply_path: Path, vertices: np.ndarray, metadata: bytes) -> None:
    """
    Write a SHARP-compatible PLY file.

    Reconstructs the original header format with the same metadata elements
    so the file can be consumed by downstream tools and the .splat converter.
    """
    n = len(vertices)

    # Build header identical to SHARP's output format
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        # Metadata element declarations — match SHARP's format
        "element extrinsic 16",
        "property float extrinsic",
        "element intrinsic 9",
        "property float intrinsic",
        "element image_size 2",
        "property uint image_size",
        "element frame 2",
        "property int frame",
        "element disparity 2",
        "property float disparity",
        "element color_space 1",
        "property uchar color_space",
        "element version 3",
        "property uchar version",
        "end_header",
    ]

    header = "\n".join(header_lines) + "\n"

    with open(ply_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertices.tobytes())
        f.write(metadata)
