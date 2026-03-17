"""
batch_sharp_gaussian.py
-----------------------
Batch-processes images from SourcePhotos/ through Apple's SHARP model
(Gaussian Splatting) using the SHARP CLI. Outputs one subfolder per image
inside Gaussians-SHARP/.

Usage:
    python Scripts/batch_sharp_gaussian.py

All paths are resolved relative to __file__, so the script works regardless
of the current working directory.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path constants — all relative to the project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR   = PROJECT_ROOT / "SourcePhotos"
OUTPUT_DIR   = PROJECT_ROOT / "Gaussians-SHARP"
CHECKPOINT   = PROJECT_ROOT / "ml-sharp" / "sharp_model.pt"
SHARP_EXE    = PROJECT_ROOT / "env_sharp" / "Scripts" / "sharp.exe"

# ---------------------------------------------------------------------------
# Supported input image extensions
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def collect_images(source_dir: Path) -> list[Path]:
    """Return a sorted list of supported image files in *source_dir*."""
    return sorted(
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def is_already_processed(output_subdir: Path) -> bool:
    """Return True if *output_subdir* exists and contains at least one file."""
    return output_subdir.exists() and any(output_subdir.iterdir())


def build_sharp_command(
    input_image: Path,
    output_subdir: Path,
    checkpoint: Path,
) -> list[str]:
    """Construct the SHARP CLI command as a list of strings."""
    return [
        str(SHARP_EXE), "predict",
        "-i", str(input_image),
        "-o", str(output_subdir),
        "-c", str(checkpoint),
    ]


def run_sharp(command: list[str]) -> subprocess.CompletedProcess:
    """
    Execute *command* via subprocess.

    Captures stdout and stderr; raises CalledProcessError on non-zero exit.
    """
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )


def process_image(
    image_path: Path,
    index: int,
    total: int,
    output_dir: Path,
    checkpoint: Path,
) -> str:
    """
    Process a single image through SHARP.

    Returns one of: "skipped", "success", "failed".
    """
    output_subdir = output_dir / image_path.stem

    if is_already_processed(output_subdir):
        print(f"  Skipping {image_path.name} — already processed")
        return "skipped"

    print(f"  Processing {index}/{total}: {image_path.name}...")

    output_subdir.mkdir(parents=True, exist_ok=True)
    command = build_sharp_command(image_path, output_subdir, checkpoint)

    start = time.perf_counter()
    try:
        run_sharp(command)
        elapsed = time.perf_counter() - start
        print(f"  ✓ Done in {elapsed:.1f}s")
        return "success"

    except subprocess.CalledProcessError as exc:
        elapsed = time.perf_counter() - start
        # Clean up partial output so re-runs can retry this image
        shutil.rmtree(output_subdir, ignore_errors=True)
        error_detail = exc.stderr.strip() if exc.stderr and exc.stderr.strip() else str(exc)
        print(f"  ✗ Failed: {image_path.name} — {error_detail}")
        return "failed"

    except FileNotFoundError:
        # SHARP executable not on PATH or not found at SHARP_EXE
        shutil.rmtree(output_subdir, ignore_errors=True)
        print(
            f"  ✗ Failed: {image_path.name} — "
            f"'sharp' executable not found at {SHARP_EXE}. "
            "Make sure env_sharp is set up correctly."
        )
        return "failed"


def print_summary(
    total: int,
    succeeded: int,
    skipped: int,
    failed: int,
    wall_time: float,
) -> None:
    """Print a final batch-run summary."""
    print("\n" + "=" * 50)
    print("Batch complete")
    print("=" * 50)
    print(f"  Total attempted : {total}")
    print(f"  Succeeded       : {succeeded}")
    print(f"  Skipped         : {skipped}")
    print(f"  Failed          : {failed}")
    print(f"  Wall-clock time : {wall_time:.1f}s")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Validate prerequisites
    if not SOURCE_DIR.exists():
        raise SystemExit(f"[Error] Source directory not found: {SOURCE_DIR}")
    if not CHECKPOINT.exists():
        raise SystemExit(f"[Error] SHARP checkpoint not found: {CHECKPOINT}")
    if not SHARP_EXE.exists():
        raise SystemExit(
            f"[Error] SHARP executable not found: {SHARP_EXE}\n"
            "Make sure env_sharp is set up correctly."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = collect_images(SOURCE_DIR)
    if not images:
        print(f"No supported images found in {SOURCE_DIR}")
        return

    total = len(images)
    print(f"\nFound {total} image(s) in {SOURCE_DIR}")
    print(f"Output directory : {OUTPUT_DIR}")
    print(f"Checkpoint       : {CHECKPOINT}")
    print(f"SHARP executable : {SHARP_EXE}\n")

    succeeded = 0
    skipped   = 0
    failed    = 0

    batch_start = time.perf_counter()

    for idx, image_path in enumerate(images, start=1):
        result = process_image(image_path, idx, total, OUTPUT_DIR, CHECKPOINT)
        if result == "success":
            succeeded += 1
        elif result == "skipped":
            skipped += 1
        else:
            failed += 1

    wall_time = time.perf_counter() - batch_start
    print_summary(total, succeeded, skipped, failed, wall_time)


if __name__ == "__main__":
    main()
