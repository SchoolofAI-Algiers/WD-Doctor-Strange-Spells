"""Live test: Capture -> MediaPipe -> Geometry -> Recognizer + Motion -> Renderer.

- Open palm (confirmed by the recognizer, ~8 frames) -> shield on that hand.
- Mirror / Ruby / Portal are only DETECTED here; their status is printed on screen
  (no art presets exist for them yet).
Press q to quit.

Run from the project root:   python DEMO.py
Needs: pip install mediapipe opencv-python numpy
First run downloads hand_landmarker.task (~8 MB) next to this file.

Requires in src/gestures/recognizer.py:
  - mirror_metrics() and the new is_mirror() (MIRROR_WINDOW / MIRROR_MIN_APART)
  - TwoHandContext, SpellDetector, is_ruby, is_portal
"""

import math
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from src.gestures.recognizer import (
    Gesture,
    GestureConfig,
    GestureRecognizer,
    PoseState,
    SpellDetector,
    StabilityConfig,
    TwoHandContext,
    extended_fingers,
    is_mirror,
    is_portal,
    is_ruby,
    mirror_metrics,
)
from src.motion.analyzer import MotionAnalyzer
from src.rendering.spell_renderer import SPELL_FOR, SpellRenderer, to_active
from src.rendering.spells import make_spells

CAM_INDEX = 0
SHOW_CENTER = True
SHOW_DEBUG = True  # raw pose / extended fingers / mirror numbers on screen
PALM_POINTS = (0, 5, 9, 13, 17)  # wrist + knuckles
INDEX_TIP = 8

MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def make_geometry(lm, w: int, h: int) -> SimpleNamespace:
    """Stand-in for your calculator's HandGeometry builder.

    If your src/geometry/calculator.py has a real function that builds a HandGeometry
    from MediaPipe landmarks, call that here instead and keep the rest unchanged.
    """
    pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float64)
    scale = float(max(w, h))
    center = pts[list(PALM_POINTS)].mean(axis=0)
    wrist, mid = pts[0], pts[9]
    rigid = float(
        np.linalg.norm(mid - wrist)
    )  # wrist -> middle knuckle, stays put when fingers curl
    spread = float(np.mean(np.linalg.norm(pts - center, axis=1)))  # shrinks when the hand closes
    dx, dy = mid - wrist
    return SimpleNamespace(
        landmarks=pts,
        landmarks_norm=pts / scale,
        hand_size_norm=rigid / scale,
        hand_size_px=spread,
        rigid_size_px=rigid,
        palm_center_px=(float(center[0]), float(center[1])),
        palm_angle_rad=math.atan2(dy, dx) + math.pi / 2,  # 0 when the hand points up
    )


def draw_palm_center(frame, geometry, color=(0, 255, 0)) -> None:
    cx, cy = (int(v) for v in geometry.palm_center_px)
    cv2.circle(frame, (cx, cy), 6, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 12, color, 2, cv2.LINE_AA)


def put(frame, text: str, x: int, y: int, color=(255, 255, 255), scale: float = 0.6) -> None:
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


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
    recognizer = GestureRecognizer()
    analyzer = MotionAnalyzer()

    # "together" is more generous than the default (palm centers within 2.5 hand-spreads)
    ctx = TwoHandContext(GestureConfig(together_max_ratio=2.5))

    fast = GestureConfig(stability=StabilityConfig(enter_frames=2, confirm_frames=3, exit_frames=3))
    # Mirror is a short burst: 2 frames to turn on, stays on ~8 frames after
    burst = GestureConfig(
        stability=StabilityConfig(enter_frames=1, confirm_frames=1, exit_frames=8)
    )
    detectors = {
        "MIRROR": SpellDetector(lambda g, c: is_mirror(ctx), burst),
        "RUBY": SpellDetector(lambda g, c: is_ruby(ctx), fast),
        "PORTAL": SpellDetector(lambda g, c: is_portal(ctx), fast),
    }

    t0 = time.perf_counter()
    prev = t0
    last_ts = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror view
        now = time.perf_counter()
        anim_time = now - t0
        dt = now - prev
        prev = now

        # ---- tracker ----
        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        last_ts = max(last_ts + 1, int(anim_time * 1000))  # must strictly increase
        res = tracker.detect_for_video(mp_img, last_ts)

        # ---- geometry + ids (Left = 0, Right = 1) ----
        geometries = [make_geometry(lm, w, h) for lm in res.hand_landmarks]
        ids = [0 if hd[0].category_name == "Left" else 1 for hd in res.handedness]
        if len(set(ids)) != len(ids):  # both hands got the same label: use list positions
            ids = list(range(len(geometries)))
        by_id = dict(zip(ids, geometries, strict=True))

        # ---- recognizer + motion ----
        states = recognizer.update(geometries, ids)
        motion = {}
        for hid, geo in zip(ids, geometries, strict=True):
            tip = (float(geo.landmarks[INDEX_TIP][0]), float(geo.landmarks[INDEX_TIP][1]))
            pointing = recognizer.raw.get(hid) is Gesture.POINTING
            motion[hid] = analyzer.update_raw(
                hid, geo.palm_center_px, geo.rigid_size_px, dt, tip, pointing
            )
        analyzer.retain_only(by_id)

        # ---- two-hand spells (call every frame, even with no hands) ----
        ctx.observe(anim_time, by_id, recognizer.raw, motion)
        active = {
            name: d.update(geometries).state is PoseState.ACTIVE for name, d in detectors.items()
        }

        # ---- renderer: shield on every confirmed open palm ----
        spells = []
        seen: set[int] = set()
        for g, geo in zip(states, geometries, strict=True):
            spell_id = SPELL_FOR.get(g.gesture)
            if spell_id is None:
                continue
            t = renderer.compute_transform(
                geo,
                to_active(g),
                motion.get(g.hand_id),
                anim_time,
                hand_id=g.hand_id,
                spell_id=spell_id,
            )
            seen.add(g.hand_id)
            spells.append((t, g))
        spells += renderer.ghost_spells(seen)
        renderer.render_all(frame, spells, anim_time)

        # ---- overlay ----
        if SHOW_CENTER:
            for geo in geometries:
                draw_palm_center(frame, geo)
        for g, geo in zip(states, geometries, strict=True):
            cx, cy = (int(v) for v in geo.palm_center_px)
            side = "L" if g.hand_id == 0 else "R"
            label = f"{side}: {g.gesture.value} f={g.frames_held} c={g.confidence:.2f}"
            put(frame, label, max(cx - 60, 5), max(cy - 90, 20), (0, 255, 255))
            m = motion.get(g.hand_id)
            if m is not None:
                put(
                    frame,
                    f"{'still' if m.is_stationary else 'moving'} {m.depth_trend.name.lower()}",
                    max(cx - 60, 5),
                    max(cy - 70, 40),
                    (200, 200, 200),
                    0.5,
                )
            if SHOW_DEBUG:
                raw = recognizer.raw.get(g.hand_id)
                ext = (
                    ",".join(sorted(extended_fingers(geo.landmarks_norm, recognizer.config))) or "-"
                )
                put(
                    frame,
                    f"raw={raw.value if raw else '-'} ext={ext} size={geo.hand_size_norm:.2f}",
                    max(cx - 60, 5),
                    max(cy - 50, 60),
                    (255, 200, 0),
                    0.5,
                )

        put(frame, f"{1.0 / max(dt, 1e-6):.0f} fps", 10, 28, scale=0.8)
        y = 56
        for name in ("MIRROR", "RUBY", "PORTAL"):
            on = active.get(name, False)
            put(
                frame,
                f"{name}: {'ACTIVE' if on else '-'}",
                10,
                y,
                (0, 255, 0) if on else (160, 160, 160),
            )
            y += 24

        if SHOW_DEBUG:
            mm = mirror_metrics(ctx)
            dbg = f"together={ctx.together_now} armed_ago={anim_time - ctx.together_active_t:.2f}s"
            if mm:
                dbg += f" apart={mm[0]:.2f} opposite={mm[1]} horiz={mm[2]}"
            else:
                dbg += " (need both hands)"
            put(frame, dbg, 10, y + 6, (255, 200, 0), 0.5)

        cv2.imshow("spell demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
