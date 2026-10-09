"""Live demo: Capture -> Tracker (MediaPipe) -> Geometry -> Renderer (shield).

Open palm facing the camera = shield. Press q to quit.
Needs: pip install mediapipe opencv-python numpy
First run downloads hand_landmarker.task (~8 MB) next to this file.
"""

import math
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from src.gestures.recognizer import PoseState
from src.rendering.spell_renderer import SpellRenderer
from src.rendering.spells import make_spells

CAM_INDEX = 0
SHOW_CENTER = True  # draw a dot on the palm center
PALM_POINTS = (0, 5, 9, 13, 17)  # wrist + knuckles
FINGERS = ((8, 6), (12, 10), (16, 14), (20, 18))  # (tip, pip), thumb ignored

MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def build_geometry(lm, w: int, h: int) -> SimpleNamespace:
    pts = [(p.x * w, p.y * h) for p in lm]
    cx = sum(pts[i][0] for i in PALM_POINTS) / len(PALM_POINTS)
    cy = sum(pts[i][1] for i in PALM_POINTS) / len(PALM_POINTS)
    wrist, mid_mcp = pts[0], pts[9]
    dx, dy = mid_mcp[0] - wrist[0], mid_mcp[1] - wrist[1]

    # size follows the whole hand: shrinks when fingers curl or the hand moves away
    spread = sum(math.hypot(x - cx, y - cy) for x, y in pts) / len(pts)

    return SimpleNamespace(
        palm_center_px=(cx, cy),
        hand_size_px=spread,
        palm_angle_rad=math.atan2(dy, dx) + math.pi / 2,  # 0 when the hand points up
    )


def build_gesture(lm, score: float) -> SimpleNamespace:
    def d(i: int) -> float:
        return math.hypot(lm[i].x - lm[0].x, lm[i].y - lm[0].y)

    extended = sum(d(tip) > d(pip) for tip, pip in FINGERS)
    state = PoseState.ACTIVE if extended >= 3 else PoseState.IDLE
    return SimpleNamespace(state=state, confidence=score)


def draw_palm_center(frame, geometry, color=(0, 255, 0)) -> None:
    cx, cy = (int(v) for v in geometry.palm_center_px)
    cv2.circle(frame, (cx, cy), 6, color, -1, cv2.LINE_AA)  # filled dot
    cv2.circle(frame, (cx, cy), 12, color, 2, cv2.LINE_AA)  # outer ring
    cv2.putText(frame, f"({cx}, {cy})", (cx + 16, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def make_tracker() -> vision.HandLandmarker:
    if not MODEL_PATH.exists():
        print("Downloading hand model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def main() -> None:
    cap = cv2.VideoCapture(CAM_INDEX)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read from the webcam")
    h, w = frame.shape[:2]

    renderer = SpellRenderer(make_spells(), (h, w))
    tracker = make_tracker()

    t0 = time.perf_counter()
    prev = t0
    last_ts = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror, feels natural
        now = time.perf_counter()
        anim_time = now - t0

        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        last_ts = max(last_ts + 1, int(anim_time * 1000))  # must strictly increase
        res = tracker.detect_for_video(mp_img, last_ts)

        spells = []
        geometries = []
        seen: set[int] = set()
        for lm, handed in zip(res.hand_landmarks, res.handedness, strict=True):
            cat = handed[0]
            hand_id = 0 if cat.category_name == "Left" else 1
            geometry = build_geometry(lm, w, h)
            geometries.append(geometry)
            gesture = build_gesture(lm, cat.score)
            t = renderer.compute_transform(
                geometry, gesture, None, anim_time, hand_id=hand_id, spell_id="shield"
            )
            seen.add(hand_id)
            spells.append((t, gesture))

        spells += renderer.ghost_spells(seen)
        renderer.render_all(frame, spells, anim_time)

        if SHOW_CENTER:  # drawn after the spell so the dot stays on top
            for geometry in geometries:
                draw_palm_center(frame, geometry)

        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(frame, f"{fps:.0f} fps", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("shield demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()