#!/usr/bin/env python3
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
"""
LucidPast Phase 4 — Compress Pipeline Orchestrator
=====================================================
Run the full compression pipeline: validate → fit → TRBF → .splat4d → serve.

Usage:
    python compress.py <reconstruct_output_dir> [options]

    # Full pipeline:
    python compress.py ../reconstruct/output/testclip2

    # Skip validation (already done):
    python compress.py ../reconstruct/output/testclip2 --skip-validate

    # Serve the result for viewing:
    python compress.py ../reconstruct/output/testclip2 --serve --port 8002

Examples:
    # Quick test with default settings:
    python compress.py ../reconstruct/output/testclip2 --serve

    # High-quality dashcam (higher polynomial degree):
    python compress.py ../reconstruct/output/testclip2 --pos-degree 3
"""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent


def compress(
    reconstruct_dir: Path,
    output_dir: Path | None = None,
    pos_degree: int = 3,
    color_degree: int = 1,
    opacity_degree: int = 1,
    scale_degree: int = 1,
    rot_degree: int = 0,
    fps: float = 8.0,
    skip_validate: bool = False,
    no_trbf: bool = False,
    serve: bool = False,
    port: int = 8002,
) -> dict:
    """Run the full Phase 4 compression pipeline."""
    reconstruct_dir = Path(reconstruct_dir).resolve()
    clip_name = reconstruct_dir.name

    if output_dir is None:
        output_dir = _SCRIPT_DIR / "output" / clip_name
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  LucidPast Phase 4 — Compress Pipeline")
    print(f"  Clip:            {clip_name}")
    print(f"  Position degree: {pos_degree}")
    print(f"  Color degree:    {color_degree}")
    print(f"  Opacity degree:  {opacity_degree}")
    print(f"  Scale degree:    {scale_degree}")
    print(f"  Rotation degree: {rot_degree}")
    print(f"  TRBF:            {'OFF' if no_trbf else 'ON'}")
    print(f"  Output:          {output_dir}")
    print(f"{'='*60}")

    t_start = time.time()
    results = {}

    # ── Step 1: Validate correspondence ──
    if not skip_validate:
        print(f"\n{'─'*60}")
        print(f"  Step 1/4 — Validate Correspondence")
        print(f"{'─'*60}")
        from validate_correspondence import validate_correspondence
        val_result = validate_correspondence(reconstruct_dir, output_dir)
        results["correspondence"] = val_result["verdict"]
        if val_result["verdict"] == "REJECTED":
            print(f"\n✗  Correspondence rejected. Cannot proceed with polynomial compression.")
            print(f"   Paper 1 (Dynamic 3DGS) tracking would be needed.")
            sys.exit(1)
    else:
        print(f"\n⟳  Skipping correspondence validation")
        results["correspondence"] = "SKIPPED"

    # ── Step 2: Polynomial motion fitting ──
    print(f"\n{'─'*60}")
    print(f"  Step 2/4 — Polynomial Motion Fitting")
    print(f"{'─'*60}")
    from fit_motion import fit_motion
    fit_result = fit_motion(
        reconstruct_dir, output_dir,
        pos_degree=pos_degree,
        color_degree=color_degree,
        opacity_degree=opacity_degree,
        scale_degree=scale_degree,
        rot_degree=rot_degree,
        fps=fps,
    )
    results["fit"] = {
        "compression_ratio": fit_result["compression"]["ratio"],
        "position_rmse_median": fit_result["residuals"]["position"]["rmse_median"],
    }

    # ── Step 3: TRBF temporal opacity ──
    if not no_trbf:
        print(f"\n{'─'*60}")
        print(f"  Step 3/4 — TRBF Temporal Opacity")
        print(f"{'─'*60}")
        from fit_trbf import fit_trbf
        trbf_result = fit_trbf(reconstruct_dir, output_dir, fps=fps)
        results["trbf"] = {
            "stable": trbf_result["classification"]["stable"],
            "transient": trbf_result["classification"]["transient"],
        }
    else:
        print(f"\n⟳  Skipping TRBF fitting")
        results["trbf"] = "SKIPPED"

    # ── Step 4: Build .splat4d ──
    print(f"\n{'─'*60}")
    print(f"  Step 4/4 — Build .splat4d")
    print(f"{'─'*60}")
    from splat4d import build_splat4d
    splat4d_result = build_splat4d(reconstruct_dir, output_dir, no_trbf=no_trbf)
    results["splat4d"] = {
        "file_size_mb": splat4d_result["file_size_mb"],
        "compression_vs_splat": splat4d_result["compression_vs_splat"],
        "compression_vs_ply": splat4d_result["compression_vs_ply"],
    }

    # ── Copy viewer ──
    viewer_src = _SCRIPT_DIR / "view_4d.html"
    if viewer_src.exists():
        viewer_dst = output_dir / "view_4d.html"
        shutil.copy2(viewer_src, viewer_dst)

        # Create a convenience redirect that auto-sets the ?file= parameter
        splat4d_filename = f"{clip_name}.splat4d"
        redirect_html = output_dir / "index.html"
        redirect_html.write_text(
            f'<!doctype html><meta http-equiv="refresh" '
            f'content="0;url=view_4d.html?file={splat4d_filename}">'
        )

    # ── Summary ──
    total_time = round(time.time() - t_start, 2)
    results["total_seconds"] = total_time

    summary_path = output_dir / "compress_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))

    splat4d_path = output_dir / f"{clip_name}.splat4d"

    print(f"\n{'='*60}")
    print(f"  ✅ Phase 4 Compression Complete — {total_time}s")
    print(f"     .splat4d:      {splat4d_path.name} ({splat4d_result['file_size_mb']} MB)")
    print(f"     Compression:   {splat4d_result['compression_vs_splat']}× vs .splat flipbook")
    print(f"     Compression:   {splat4d_result['compression_vs_ply']}× vs .ply flipbook")
    print(f"     Output:        {output_dir}")
    print(f"{'='*60}")

    # ── Serve ──
    if serve:
        url = f"http://localhost:{port}/index.html"
        print(f"\n  Starting viewer at {url}")
        print(f"  Press Ctrl+C to stop the server.\n")
        import webbrowser
        try:
            webbrowser.open(url)
        except Exception:
            pass
        try:
            subprocess.run(
                [sys.executable, "-m", "http.server", str(port)],
                cwd=output_dir,
            )
        except KeyboardInterrupt:
            print("\n  Server stopped.")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="LucidPast Phase 4: Full compression pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("reconstruct_dir", type=Path,
                        help="Phase 2 output directory (contains splats/*.ply)")
    parser.add_argument("--pos-degree", type=int, default=3,
                        help="Position polynomial degree (default: 3)")
    parser.add_argument("--color-degree", type=int, default=1,
                        help="Color polynomial degree (default: 1)")
    parser.add_argument("--opacity-degree", type=int, default=1,
                        help="Opacity polynomial degree (default: 1)")
    parser.add_argument("--scale-degree", type=int, default=1,
                        help="Scale polynomial degree (default: 1)")
    parser.add_argument("--rot-degree", type=int, default=0,
                        help="Rotation polynomial degree (default: 0)")
    parser.add_argument("--fps", type=float, default=8.0,
                        help="Frame rate (default: 8.0)")
    parser.add_argument("--skip-validate", action="store_true",
                        help="Skip correspondence validation")
    parser.add_argument("--no-trbf", action="store_true",
                        help="Skip TRBF temporal opacity")
    parser.add_argument("--serve", action="store_true",
                        help="Start HTTP server after building")
    parser.add_argument("--port", type=int, default=8002,
                        help="HTTP server port (default: 8002)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    compress(
        reconstruct_dir=args.reconstruct_dir,
        output_dir=args.output_dir,
        pos_degree=args.pos_degree,
        color_degree=args.color_degree,
        opacity_degree=args.opacity_degree,
        scale_degree=args.scale_degree,
        rot_degree=args.rot_degree,
        fps=args.fps,
        skip_validate=args.skip_validate,
        no_trbf=args.no_trbf,
        serve=args.serve,
        port=args.port,
    )


if __name__ == "__main__":
    main()

