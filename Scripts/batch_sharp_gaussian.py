"""
batch_sharp_gaussian.py
-----------------------
Batch-processes images from input/ through Apple's SHARP model
(Gaussian Splatting) using the SHARP CLI. Outputs one subfolder per image
inside output/gaussians/.

Usage:
    python scripts/batch_sharp_gaussian.py
    python scripts/batch_sharp_gaussian.py --config path/to/config.yaml

All paths are resolved relative to the project root (parent of this script's
directory) unless overridden via config.yaml.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults — used when config.yaml is absent or incomplete
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

_DEFAULT_SOURCE_DIR = PROJECT_ROOT / "input"
_DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "gaussians"
_DEFAULT_CHECKPOINT = PROJECT_ROOT / "ml-sharp" / "sharp_model.pt"
_DEFAULT_SHARP_EXE  = PROJECT_ROOT / "env_sharp" / "Scripts" / "sharp.exe"

_DEFAULT_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}
)

# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def load_config(config_path: Path | None = None) -> dict:
    """
    Load configuration from a YAML file.

    Falls back to hardcoded defaults if the file is missing or if ``pyyaml``
    is not installed.  Individual keys can be absent — only present keys
    override the defaults.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"

    config: dict = {}
    if config_path.exists():
        try:
            import yaml  # noqa: WPS433 — optional import

            with config_path.open("r", encoding="utf-8") as fh:
                config = yaml.safe_load(fh) or {}
            print(f"[OK] Loaded config from {config_path}")
        except ImportError:
            print(
                "[WARN] pyyaml not installed -- using default configuration. "
                "Install with: pip install pyyaml"
            )
    else:
        print(f"[INFO] No config file at {config_path} -- using defaults")

    return config


def resolve_paths(config: dict) -> tuple[Path, Path, Path, Path]:
    """Return (source_dir, output_dir, checkpoint, sharp_exe) from *config*."""
    paths = config.get("paths", {})
    source_dir = PROJECT_ROOT / paths.get("input_dir", _DEFAULT_SOURCE_DIR.relative_to(PROJECT_ROOT))
    output_dir = PROJECT_ROOT / paths.get("output_dir", _DEFAULT_OUTPUT_DIR.relative_to(PROJECT_ROOT))
    checkpoint = PROJECT_ROOT / paths.get("checkpoint", _DEFAULT_CHECKPOINT.relative_to(PROJECT_ROOT))
    sharp_exe  = PROJECT_ROOT / paths.get("sharp_exe", _DEFAULT_SHARP_EXE.relative_to(PROJECT_ROOT))
    return source_dir, output_dir, checkpoint, sharp_exe


def resolve_extensions(config: dict) -> frozenset[str]:
    """Return the set of supported image extensions from *config*."""
    sharp_cfg = config.get("sharp", {})
    ext_list = sharp_cfg.get("supported_extensions")
    if ext_list:
        return frozenset(ext_list)
    return _DEFAULT_EXTENSIONS


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def collect_images(source_dir: Path, extensions: frozenset[str]) -> list[Path]:
    """Return a sorted list of supported image files in *source_dir*."""
    return sorted(
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )


def is_already_processed(output_subdir: Path) -> bool:
    """Return True if *output_subdir* exists and contains at least one file."""
    return output_subdir.exists() and any(output_subdir.iterdir())


def build_sharp_command(
    input_image: Path,
    output_subdir: Path,
    checkpoint: Path,
    sharp_exe: Path,
) -> list[str]:
    """Construct the SHARP CLI command as a list of strings."""
    return [
        str(sharp_exe), "predict",
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
    sharp_exe: Path,
) -> str:
    """
    Process a single image through SHARP.

    Returns one of: "skipped", "success", "failed".
    """
    output_subdir = output_dir / image_path.stem

    if is_already_processed(output_subdir):
        print(f"  Skipping {image_path.name} -- already processed")
        return "skipped"

    print(f"  Processing {index}/{total}: {image_path.name}...")

    output_subdir.mkdir(parents=True, exist_ok=True)
    command = build_sharp_command(image_path, output_subdir, checkpoint, sharp_exe)

    start = time.perf_counter()
    try:
        run_sharp(command)
        elapsed = time.perf_counter() - start
        print(f"  [OK] Done in {elapsed:.1f}s")
        return "success"

    except subprocess.CalledProcessError as exc:
        elapsed = time.perf_counter() - start
        # Clean up partial output so re-runs can retry this image
        shutil.rmtree(output_subdir, ignore_errors=True)
        error_detail = exc.stderr.strip() if exc.stderr and exc.stderr.strip() else str(exc)
        print(f"  [FAIL] Failed: {image_path.name} -- {error_detail}")
        return "failed"

    except FileNotFoundError:
        # SHARP executable not on PATH or not found at sharp_exe
        shutil.rmtree(output_subdir, ignore_errors=True)
        print(
            f"  [FAIL] Failed: {image_path.name} -- "
            f"'sharp' executable not found at {sharp_exe}. "
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
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch-process images through Apple SHARP (Gaussian Splatting)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: <project_root>/config.yaml)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    source_dir, output_dir, checkpoint, sharp_exe = resolve_paths(config)
    extensions = resolve_extensions(config)

    # Validate prerequisites
    if not source_dir.exists():
        raise SystemExit(f"[Error] Source directory not found: {source_dir}")
    if not checkpoint.exists():
        raise SystemExit(f"[Error] SHARP checkpoint not found: {checkpoint}")
    if not sharp_exe.exists():
        raise SystemExit(
            f"[Error] SHARP executable not found: {sharp_exe}\n"
            "Make sure env_sharp is set up correctly."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(source_dir, extensions)
    if not images:
        print(f"No supported images found in {source_dir}")
        return

    total = len(images)
    print(f"\nFound {total} image(s) in {source_dir}")
    print(f"Output directory : {output_dir}")
    print(f"Checkpoint       : {checkpoint}")
    print(f"SHARP executable : {sharp_exe}\n")

    succeeded = 0
    skipped   = 0
    failed    = 0

    batch_start = time.perf_counter()

    for idx, image_path in enumerate(images, start=1):
        result = process_image(image_path, idx, total, output_dir, checkpoint, sharp_exe)
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
