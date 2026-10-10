"""Drawing helpers for the debug overlay. Pure OpenCV, no pipeline logic."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from src.geometry.calculator import HandGeometry
from src.gestures.recognizer import GestureConfig, extended_fingers
from src.pipeline.state import PipelineState
from src.tracking.hand_tracker import HandLandmarks

Frame = NDArray[np.uint8]
Color = tuple[int, int, int]

BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]  # fmt: skip

HAND_COLORS: dict[int, Color] = {0: (255, 160, 60), 1: (80, 220, 80)}  # BGR: Left, Right
DEFAULT_COLOR: Color = (200, 80, 255)
SPELL_NAMES = ("MIRROR", "RUBY", "PORTAL")


def put(
    frame: Frame, text: str, x: int, y: int, color: Color = (255, 255, 255), scale: float = 0.6
) -> None:
    """Text with a black outline so it stays readable on any background."""
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_landmarks(frame: Frame, hand: HandLandmarks, color: Color) -> None:
    px = hand.landmarks_px[:, :2]
    for a, b in BONES:
        cv2.line(frame, (int(px[a][0]), int(px[a][1])), (int(px[b][0]), int(px[b][1])), color, 2)
    for p in px:
        cv2.circle(frame, (int(p[0]), int(p[1])), 3, (255, 255, 255), -1)


def draw_palm_center(frame: Frame, geometry: HandGeometry, color: Color = (0, 255, 0)) -> None:
    cx, cy = int(geometry.palm_center_px[0]), int(geometry.palm_center_px[1])
    cv2.circle(frame, (cx, cy), 6, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 12, color, 2, cv2.LINE_AA)


def draw_debug_overlay(frame: Frame, state: PipelineState) -> None:
    """Landmarks, palm center, gesture and motion labels, FPS and two-hand spell status."""
    gestures = {g.hand_id: g for g in state.gestures}

    for hid, hand, geo in zip(state.hand_ids, state.hands, state.geometries, strict=True):
        color = HAND_COLORS.get(hid, DEFAULT_COLOR)
        draw_landmarks(frame, hand, color)
        draw_palm_center(frame, geo)

        cx, cy = int(geo.palm_center_px[0]), int(geo.palm_center_px[1])
        x = max(cx - 60, 5)
        side = "L" if hid == 0 else "R"

        g = gestures.get(hid)
        if g is not None:
            label = f"{side}: {g.gesture.value} f={g.frames_held} c={g.confidence:.2f}"
            put(frame, label, x, max(cy - 90, 20), (0, 255, 255))

        m = state.motions.get(hid)
        if m is not None:
            motion_text = f"{'still' if m.is_stationary else 'moving'} {m.depth_trend.name.lower()}"
            put(frame, motion_text, x, max(cy - 70, 40), (200, 200, 200), 0.5)

        raw = state.raw_gestures.get(hid)
        ext = "".join(
            n[0].upper() if n in extended_fingers(geo.landmarks_norm, GestureConfig()) else "-"
            for n in ("thumb", "index", "middle", "ring", "pinky")
        )
        put(
            frame,
            f"raw={raw.value if raw else '-'} ext={ext} size={geo.hand_size_norm:.2f}",
            x,
            max(cy - 50, 60),
            (255, 200, 0),
            0.5,
        )

    put(frame, f"{state.fps_render:.0f} fps", 10, 28, scale=0.8)
    y = 56
    for name in SPELL_NAMES:
        on = state.spells_active.get(name, False)
        put(
            frame,
            f"{name}: {'ACTIVE' if on else '-'}",
            10,
            y,
            (0, 255, 0) if on else (160, 160, 160),
        )
        y += 24

    dbg = f"together={state.together_now}"
    if state.mirror_metrics is not None:
        apart, opposite, horizontal = state.mirror_metrics
        dbg += f" apart={apart:.2f} opposite={opposite} horiz={horizontal}"
    else:
        dbg += " (need both hands)"
    put(frame, dbg, 10, y + 6, (255, 200, 0), 0.5)
