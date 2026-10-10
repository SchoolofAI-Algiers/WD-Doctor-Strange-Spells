"""PipelineState: everything computed during one frame, in one place."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from src.geometry.calculator import HandGeometry
from src.gestures.recognizer import Gesture, GestureState
from src.motion.analyzer import MotionState
from src.tracking.hand_tracker import HandLandmarks


@dataclass
class PipelineState:
    """Snapshot of the last processed frame.

    The per-hand lists (hands, hand_ids, geometries, gestures) all have the same
    length and order. The dicts are keyed by hand id (0 = Left, 1 = Right), because
    with a single hand the id can be 1, so a list index would not match it.
    """

    frame: NDArray[np.uint8] | None = None  # the output image (spells + overlay)
    timestamp: float = 0.0  # seconds, from the pipeline clock
    dt: float = 0.0  # seconds since the previous frame
    animation_time: float = 0.0  # seconds since the pipeline started

    hands: list[HandLandmarks] = field(default_factory=list)
    hand_ids: list[int] = field(default_factory=list)
    geometries: list[HandGeometry] = field(default_factory=list)
    gestures: list[GestureState] = field(default_factory=list)  # stabilized
    raw_gestures: dict[int, Gesture] = field(default_factory=dict)  # not stabilized
    motions: dict[int, MotionState] = field(default_factory=dict)

    spells_active: dict[str, bool] = field(default_factory=dict)  # MIRROR / RUBY / PORTAL
    together_now: bool = False
    mirror_metrics: tuple[float, bool, bool] | None = None  # (apart, opposite, horizontal)

    fps_detection: float = 0.0
    fps_render: float = 0.0
