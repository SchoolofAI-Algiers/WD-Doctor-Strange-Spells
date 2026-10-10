"""Entry point: python -m src.pipeline.main

Keys: q / ESC = quit, s = screenshot, d = toggle the debug overlay.
"""

from __future__ import annotations

import argparse

from src.pipeline.config import PipelineConfig
from src.pipeline.pipeline import Pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Dr. Strange Spells: hand tracking + AR spells.")
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--width", type=int, default=640, help="requested frame width")
    ap.add_argument("--height", type=int, default=480, help="requested frame height")
    ap.add_argument("--fps", type=int, default=30, help="requested camera fps")
    ap.add_argument("--model", default=None, help="MediaPipe model file (default: project root)")
    ap.add_argument("--no-mirror", action="store_true", help="do not flip the camera image")
    ap.add_argument("--no-debug", action="store_true", help="start with the debug overlay off")
    ap.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = PipelineConfig(
        camera_id=args.camera,
        frame_width=args.width,
        frame_height=args.height,
        target_fps=args.fps,
        model_path=args.model,
        mirror_view=not args.no_mirror,
        show_debug=not args.no_debug,
    )

    try:
        pipeline = Pipeline(config)
    except FileNotFoundError as exc:  # model file missing: the message has the download link
        raise SystemExit(str(exc)) from exc
    except RuntimeError as exc:  # camera can't open
        raise SystemExit(f"{exc}. Is another program using it?") from exc

    with pipeline:
        pipeline.run(max_frames=args.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
