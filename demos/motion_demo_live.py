"""demo_live.py - the Motion Analysis subsystem on a REAL camera.

Pipeline (every stage is an existing module, this file only wires them together):

    Capture.read()  ->  HandTracker.process()  ->  GeometryCalculator.compute()
                    ->  MotionAnalyzer.update()  ->  drawing

For every hand it draws the landmarks, the palm trail, a velocity arrow, the fingertip
trail (recorded only while pointing), MOVING / STATIONARY / TOWARD / AWAY labels and
a text line with speed, direction, acceleration and a LEFT/RIGHT/UP/DOWN word.
While pointing, the only labels shown are POINTING and (if the hand moves vertically)
UP or DOWN.

The picture is mirrored (selfie view): moving your hand to YOUR right moves it to the
right of the screen, and the arrow points right (vx > 0).

Run from the project root (the folder that contains src/):

    python -m src.motion.demo_live

Keys:  q or ESC = quit    c = clear all trails    p = pause    s = save a screenshot
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from src.capture.capture import Capture
from src.geometry.calculator import GeometryCalculator, HandGeometry
from src.motion.analyzer import INDEX_TIP, MotionAnalyzer, MotionState
from src.tracking.hand_tracker import HandLandmarks, HandTracker

ARROW_SECONDS = 0.25  # the arrow shows where the hand will be in this many seconds
LOST_FRAMES = 5  # consecutive frames with no hand before all tracks are dropped
BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]  # fmt: skip
# One color per hand id (BGR): 0 = Left hand, 1 = Right hand, 2 = a duplicate
COLORS = {0: (255, 160, 60), 1: (80, 220, 80), 2: (200, 80, 255)}

# What one frame gives back for each hand
HandResult = tuple[int, HandLandmarks, HandGeometry, bool, MotionState]


# Pipeline glue


def looks_like_pointing(lm_px: np.ndarray) -> bool:
    """DEMO-ONLY stand-in for the gesture layer: index out, other fingers folded.

    Replace this call with the real gesture classifier from src/gestures when it is
    ready. A finger is "out" when its tip is farther from the wrist than its PIP joint.
    """

    def out(tip: int, pip: int) -> bool:
        return bool(
            np.linalg.norm(lm_px[tip] - lm_px[0]) > 1.1 * np.linalg.norm(lm_px[pip] - lm_px[0])
        )

    return out(8, 6) and not (out(12, 10) or out(16, 14) or out(20, 18))


def process_frame(
    frame: np.ndarray,
    tracker: HandTracker,
    calc: GeometryCalculator,
    analyzer: MotionAnalyzer,
    dt: float,
) -> list[HandResult]:
    """One frame through Tracker -> Geometry -> MotionAnalyzer."""
    results: list[HandResult] = []
    used: set[int] = set()
    for hand in tracker.process(frame):
        hand_id = 0 if hand.handedness == "Left" else 1
        if hand_id in used:  # MediaPipe called both hands the same side
            hand_id = 2
        used.add(hand_id)

        geometry = calc.compute(hand)
        lm_px = hand.landmarks_px[:, :2]
        tip = (float(lm_px[INDEX_TIP][0]), float(lm_px[INDEX_TIP][1]))
        pointing = looks_like_pointing(lm_px)
        state = analyzer.update(hand_id, geometry, dt, tip_px=tip, is_pointing=pointing)
        results.append((hand_id, hand, geometry, pointing, state))

    if used:
        analyzer.retain_only(used)  # drop tracks of hands not seen this frame
    return results


# Drawing


def _pt(p: np.ndarray | tuple[float, float]) -> tuple[int, int]:
    return int(p[0]), int(p[1])


def direction_word(state: MotionState, min_speed: float = 60.0) -> str:
    """LEFT / RIGHT / UP / DOWN from the velocity (screen coordinates, y grows downward)."""
    vx, vy = state.velocity_px_s
    if state.speed_px_s < min_speed:
        return "-"
    if abs(vx) >= abs(vy):
        return "RIGHT" if vx > 0 else "LEFT"
    return "DOWN" if vy > 0 else "UP"


def draw_hand(frame: np.ndarray, result: HandResult, row: int) -> None:
    hand_id, hand, geometry, pointing, state = result
    color = COLORS[hand_id]
    cx, cy = _pt(geometry.palm_center_px)

    # landmarks: the tracker already gives them in pixels
    px = hand.landmarks_px[:, :2]
    for a, b in BONES:
        cv2.line(frame, _pt(px[a]), _pt(px[b]), color, 2)
    for p in px:
        cv2.circle(frame, _pt(p), 3, (255, 255, 255), -1)

    # palm trail, fading with age
    pts = state.trajectory
    for k, (p0, p1) in enumerate(pairwise(pts)):
        fade = 0.25 + 0.75 * k / max(len(pts) - 1, 1)
        shade = tuple(int(c * fade) for c in color)
        cv2.line(frame, _pt(p0), _pt(p1), shade, 2)

    # fingertip trail (only while pointing)
    if len(state.fingertip_trajectory) > 1:
        path = np.array(state.fingertip_trajectory, np.int32)
        cv2.polylines(frame, [path], False, (0, 215, 255), 4)

    # palm center and velocity arrow
    cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
    vx, vy = state.velocity_px_s
    end = (int(cx + vx * ARROW_SECONDS), int(cy + vy * ARROW_SECONDS))
    cv2.arrowedLine(frame, (cx, cy), end, (0, 0, 255), 4, tipLength=0.3)

    # labels above the hand
    word = direction_word(state)
    if pointing:
        # pointing: only POINTING, plus UP or DOWN when the hand moves vertically
        tags = [("POINTING", (0, 215, 255))]
        if word in ("UP", "DOWN"):
            tags.append((word, (255, 200, 0)))
    else:
        tags = [("STATIONARY", (0, 255, 255)) if state.is_stationary else ("MOVING", (0, 140, 255))]
        if state.is_moving_toward_camera:
            tags.append(("TOWARD", (0, 255, 0)))
        if state.is_moving_away_from_camera:
            tags.append(("AWAY", (255, 120, 0)))
    y = max(cy - int(geometry.hand_size_px) - 10 - 26 * (len(tags) - 1), 70)
    for text, col in tags:
        cv2.putText(frame, text, (max(cx - 70, 5), y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        y += 26

    # one text line per hand, top left
    ax, ay = state.acceleration_px_s2
    line = (
        f"{hand.handedness:5s} speed {state.speed_px_s:5.0f} px/s  "
        f"dir {math.degrees(state.direction_rad):4.0f} deg  "
        f"acc {math.hypot(ax, ay):6.0f} px/s2  {word}"
    )
    cv2.putText(frame, line, (10, 28 + 26 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_help(frame: np.ndarray, fps: float, paused: bool) -> None:
    h = frame.shape[0]
    text = "q quit | c clear trails | p pause | s screenshot"
    cv2.putText(frame, text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(
        frame,
        f"{fps:4.1f} fps",
        (10, h - 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )
    if paused:
        cv2.putText(frame, "PAUSED", (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


# Main loop


def run(args: argparse.Namespace, capture: Capture, tracker: HandTracker) -> int:
    analyzer = MotionAnalyzer()
    calc: GeometryCalculator | None = None
    writer: cv2.VideoWriter | None = None
    results: list[HandResult] = []
    frame: np.ndarray | None = None
    paused = False
    missed = 0  # consecutive frames with no hand detected
    n = shots = 0
    last = time.perf_counter()

    while True:
        if not paused:
            try:
                frame = capture.read()
            except TimeoutError as exc:
                print(exc)
                break
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)  # selfie view: your right = screen right
            h, w = frame.shape[:2]
            if calc is None:
                calc = GeometryCalculator(w, h)
            if args.save and writer is None:
                writer = cv2.VideoWriter(
                    args.save, cv2.VideoWriter.fourcc(*"mp4v"), args.fps, (w, h)
                )

            now = time.perf_counter()
            dt, last = now - last, now
            results = process_frame(frame, tracker, calc, analyzer, dt)

            # grace period: one missed detection should not wipe the trails
            if results:
                missed = 0
            else:
                missed += 1
                if missed >= LOST_FRAMES:
                    analyzer.retain_only(())
            n += 1
        elif frame is None:
            break

        shown = frame.copy()
        for row, result in enumerate(results):
            draw_hand(shown, result, row)
        draw_help(shown, capture.get_fps(), paused)

        if writer is not None and not paused:
            writer.write(shown)
        if not args.no_window:
            cv2.imshow("Motion Analysis - live", shown)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("c"):
                for hand_id in COLORS:
                    analyzer.clear(hand_id)
                results = []
            if key == ord("p"):
                paused = not paused
                last = time.perf_counter()  # no huge dt after the pause
            if key == ord("s"):
                shots += 1
                cv2.imwrite(f"motion_demo_{shots}.png", shown)
        if args.max_frames and n >= args.max_frames:
            break

    if writer is not None:
        writer.release()
        print(f"saved {args.save}")
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Motion Analysis on a real camera.")
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--width", type=int, default=640, help="requested frame width")
    ap.add_argument("--height", type=int, default=480, help="requested frame height")
    ap.add_argument("--fps", type=int, default=30, help="requested camera fps")
    ap.add_argument("--model", default=None, help="MediaPipe model file (default: project root)")
    ap.add_argument("--save", metavar="FILE.mp4", help="record the annotated video")
    ap.add_argument("--no-window", action="store_true", help="do not open a window")
    ap.add_argument("--no-mirror", action="store_true", help="do not flip the camera image")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = never)")
    args = ap.parse_args()

    try:
        capture = Capture(args.camera, args.width, args.height, args.fps)
    except RuntimeError as exc:
        raise SystemExit(f"{exc}. Is another program using it?") from exc
    try:
        tracker = HandTracker(args.model)
    except FileNotFoundError as exc:
        capture.release()
        raise SystemExit(str(exc)) from exc

    with capture, tracker:
        return run(args, capture, tracker)


if __name__ == "__main__":
    raise SystemExit(main())
