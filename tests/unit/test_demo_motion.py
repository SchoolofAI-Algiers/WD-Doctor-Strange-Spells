from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.motion.Analyzer import MotionAnalyzer, MotionState  # noqa: E402
from src.motion.motion import DepthTrend  # noqa: E402

W, H, FPS = 960, 540, 30
ARROW_SCALE = 0.25  # seconds of travel the arrow represents


# --------------------------------------------------------------------------
# Synthetic hand: 12 s loop that exercises every detector
# --------------------------------------------------------------------------
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
        tip = (cx + 0.9 * size * math.cos(a), cy - 0.9 * size + 0.5 * size * math.sin(a))
    else:
        tip = (cx, cy - 0.9 * size)
    return center, size, tip, pointing


# --------------------------------------------------------------------------
# Webcam hand (MediaPipe) -- computes what the analyzer needs directly
# --------------------------------------------------------------------------
class WebcamHands:
    def __init__(self) -> None:
        import mediapipe as mp  # imported lazily: synthetic mode needs no mediapipe

        self._mp = mp
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2, model_complexity=0,
            min_detection_confidence=0.6, min_tracking_confidence=0.5,
        )
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("cannot open webcam (index 0)")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None, []
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        res = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        out = []
        for i, lm in enumerate(res.multi_hand_landmarks or []):
            p = [(l.x * w, l.y * h) for l in lm.landmark]
            idx = (0, 5, 9, 13, 17)
            center = (sum(p[k][0] for k in idx) / 5, sum(p[k][1] for k in idx) / 5)
            size = math.dist(p[0], p[12])
            # crude "pointing": index extended, middle/ring/pinky curled
            def ext(tip, pip):  # farther from wrist than its PIP joint
                return math.dist(p[tip], p[0]) > math.dist(p[pip], p[0]) * 1.1
            pointing = ext(8, 6) and not (ext(12, 10) or ext(16, 14) or ext(20, 18))
            out.append((i, center, size, p[8], pointing))
        return frame, out

    def close(self) -> None:
        self.cap.release()
        self.hands.close()


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def draw(frame, hand_id: int, center, size, state: MotionState, pointing: bool) -> None:
    # palm trajectory (fading)
    pts = state.trajectory
    for a, b, k in zip(pts, pts[1:], range(len(pts))):
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
    if state.depth_trend is DepthTrend.TOWARD:
        tags.append(("TOWARD", (0, 255, 0)))
    elif state.depth_trend is DepthTrend.AWAY:
        tags.append(("AWAY", (255, 128, 0)))
    if pointing:
        tags.append(("POINTING", (0, 165, 255)))
    y = c[1] - int(size) - 12 - 22 * (len(tags) - 1)
    for text, col in tags:
        cv2.putText(frame, text, (c[0] - 60, max(y, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        y += 22
    cv2.putText(
        frame, f"hand {hand_id}  speed {state.speed_px_s:6.0f} px/s  ({state.speed_px_s / max(size, 1):.2f} hs/s)",
        (10, 25 + 22 * hand_id), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
    )


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--webcam", action="store_true", help="use camera + MediaPipe")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = run until quit)")
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
            hands = [(0, c, s, tip, ptg)]
            dt, t_sim = 1.0 / FPS, t_sim + 1.0 / FPS

        for hand_id, center, size, tip, pointing in hands:
            state = analyzer.update_raw(hand_id, center, size, dt, tip, pointing)
            draw(frame, hand_id, center, size, state, pointing)
        analyzer.retain_only(h[0] for h in hands)

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