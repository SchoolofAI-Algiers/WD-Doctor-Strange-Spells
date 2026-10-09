from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from src.geometry.calculator import HandGeometry
from src.motion.analyzer import MotionAnalyzer, MotionState

W, H, FPS = 960, 540, 30
ARROW_SCALE = 0.25  # seconds of travel the arrow represents

# Synthetic hand: 12 s loop that exercises every detector


def synthetic_hand(t: float):
    """Return (center, size, tip, is_pointing) at time t (seconds)."""
    t = t % 12.0
    cx, cy, size, pointing = W / 2, H / 2, 100.0, False
    if t < 3:  # moving in a circle
        a = t / 3 * 2 * math.pi
        cx += 220 * math.cos(a)
        cy += 120 * math.sin(a)
    elif t < 5:  # hold still (jitter only)
        cx += 220 + 0.5 * math.sin(t * 40)
    elif t < 7:  # grow -> TOWARD
        size = 100 + (t - 5) / 2 * 70
        cx += 220
    elif t < 9:  # shrink -> AWAY
        size = 170 - (t - 7) / 2 * 70
        cx += 220
    else:  # point and draw a loop with the fingertip
        pointing = True
        cx += 220 - (t - 9) / 3 * 440
    center = (cx, cy)
    if pointing:
        a = (t - 9) / 3 * 2 * math.pi * 1.5
        tip = (
            cx + 0.9 * size * math.cos(a),
            cy - 0.9 * size + 0.5 * size * math.sin(a),
        )
    else:
        tip = (cx, cy - 0.9 * size)
    return center, size, tip, pointing


def to_geometry(center: tuple[float, float], size: float, tip: tuple[float, float]) -> HandGeometry:
    """The HandGeometry the real pipeline would give for this synthetic hand.

    The analyzer rebuilds the fingertip from landmarks_norm, so landmark 8 is
    placed to land exactly on `tip`. Landmark 9 sets the palm length (half the
    hand size), which is what the depth check follows."""

    lm = np.zeros((21, 2))
    lm[9] = (0.0, -0.5)
    mean = lm[[0, 5, 9, 13, 17]].mean(axis=0)
    lm[8] = mean + ((tip[0] - center[0]) / size, (tip[1] - center[1]) / size)
    return HandGeometry(
        palm_center_px=center,
        palm_center_norm=(center[0] / W, center[1] / H),
        hand_size_px=size,
        hand_size_norm=size / max(W, H),
        palm_angle_rad=0.0,
        landmarks_norm=lm,
    )


# Webcam hands: Capture -> Tracker -> Geometry (the project's own classes)


class WebcamHands:
    def __init__(self) -> None:
        raise NotImplementedError("webcam mode: Capture/Tracker wiring not written yet")

    def read(self):
        """Return (frame, hands) with hands = [(hand_id, HandGeometry, pointing)]."""
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


# Drawing


def draw(frame, hand_id: int, center, size, state: MotionState, pointing: bool) -> None:
    # palm trajectory (fading)
    pts = state.trajectory
    for k, (a, b) in enumerate(zip(pts, pts[1:], strict=False)):
        shade = int(60 + 150 * k / max(len(pts) - 1, 1))
        cv2.line(frame, tuple(map(int, a)), tuple(map(int, b)), (shade, shade, 0), 2)

    # fingertip trail
    ft = state.fingertip_trajectory
    if len(ft) > 1:
        cv2.polylines(frame, [np.array(ft, np.int32)], False, (0, 165, 255), 3)

    # palm marker + hand-size ring
    c = (int(center[0]), int(center[1]))
    cv2.circle(frame, c, 6, (0, 255, 0), -1)
    cv2.circle(frame, c, int(size), (0, 255, 0), 1)

    # velocity arrow
    vx, vy = state.velocity_px_s
    end = (int(c[0] + vx * ARROW_SCALE), int(c[1] + vy * ARROW_SCALE))
    cv2.arrowedLine(frame, c, end, (0, 0, 255), 3, tipLength=0.25)

    # labels
    tags = []
    if state.is_stationary:
        tags.append(("STATIONARY", (0, 255, 255)))
    if state.is_moving_toward_camera:
        tags.append(("TOWARD", (0, 255, 0)))
    if pointing:
        tags.append(("POINTING", (0, 165, 255)))
    y = c[1] - int(size) - 12 - 22 * (len(tags) - 1)
    for text, col in tags:
        cv2.putText(
            frame,
            text,
            (c[0] - 60, max(y, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            col,
            2,
        )
        y += 22
    cv2.putText(
        frame,
        f"hand {hand_id}  speed {state.speed_px_s:6.0f} px/s  ({state.speed_px_s / max(size, 1):.2f} hs/s)",
        (10, 25 + 22 * hand_id),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--webcam", action="store_true", help="use camera + MediaPipe")
    ap.add_argument(
        "--frames", type=int, default=0, help="stop after N frames (0 = run until quit)"
    )
    ap.add_argument("--save-frame", help="write the last frame to this path on exit")
    ap.add_argument("--no-window", action="store_true", help="don't open a window (headless)")
    args = ap.parse_args()

    analyzer = MotionAnalyzer()
    cam = WebcamHands() if args.webcam else None
    n, t_sim, last = 0, 0.0, time.perf_counter()
    frame = None

    while True:
        if cam:
            frame, hands = cam.read()
            if frame is None:
                break
            now = time.perf_counter()
            dt, last = now - last, now
        else:
            frame = np.full((H, W, 3), 25, np.uint8)
            c, s, tip, ptg = synthetic_hand(t_sim)
            hands = [(0, to_geometry(c, s, tip), ptg)]
            dt, t_sim = 1.0 / FPS, t_sim + 1.0 / FPS

        if not hands:
            analyzer.tick(dt)
        for hand_id, geometry, pointing in hands:
            state = analyzer.update(
                hand_id,
                geometry,
                dt,
                tip_px=tip,
                is_pointing=pointing,
            )
            draw(
                frame,
                hand_id,
                geometry.palm_center_px,
                geometry.hand_size_px,
                state,
                pointing,
            )

        n += 1
        if not args.no_window:
            cv2.imshow("motion demo", frame)
            if cv2.waitKey(int(1000 / FPS) if not cam else 1) & 0xFF in (27, ord("q")):
                break
        if args.frames and n >= args.frames:
            break

    if args.save_frame and frame is not None:
        cv2.imwrite(args.save_frame, frame)
    if cam:
        cam.close()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
