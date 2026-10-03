"""Velocity, smoothing, and acceleration — the part of MotionAnalyzer
that turns "position now" + "position before" into speed, direction,
and acceleration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class _HandVelocityHistory:
    """What we need to remember about ONE hand, between frames."""

    prev_position: tuple[float, float] | None = None
    prev_velocity_smooth: tuple[float, float] | None = None


@dataclass
class VelocityResult:
    """What this piece hands back to the rest of MotionAnalyzer."""

    velocity_px_s: tuple[float, float]
    speed_px_s: float
    direction_rad: float
    acceleration_px_s2: tuple[float, float]


class VelocityTracker:
    """Computes smoothed velocity and acceleration for each tracked hand."""

    def __init__(self, velocity_ema_alpha: float = 0.3) -> None:
        self._alpha = velocity_ema_alpha
        self._history: dict[int, _HandVelocityHistory] = {}

    def update(
        self, hand_id: int, position_px: tuple[float, float], dt: float
    ) -> VelocityResult:
        """Call this once per frame, per hand, with its current palm center.

        Args:
            hand_id: Which hand this is (so two hands don't share history).
            position_px: Current (x, y) palm center, in pixels.
            dt: Seconds elapsed since the last frame.

        Returns:
            VelocityResult with velocity, speed, direction, acceleration.
        """
        history = self._history.setdefault(hand_id, _HandVelocityHistory())

        # First time we ever see this hand: no "previous" to compare
        # against, so velocity/acceleration are both zero this frame.
        if history.prev_position is None:
            history.prev_position = position_px
            history.prev_velocity_smooth = (0.0, 0.0)
            return VelocityResult(
                velocity_px_s=(0.0, 0.0),
                speed_px_s=0.0,
                direction_rad=0.0,
                acceleration_px_s2=(0.0, 0.0),
            )

        # --- raw velocity: change in position / change in time ---
        prev_x, prev_y = history.prev_position
        curr_x, curr_y = position_px
        vx = (curr_x - prev_x) / dt
        vy = (curr_y - prev_y) / dt

        # --- smoothing (EMA): blend new reading with recent history ---
        # prev_velocity_smooth is always set together with prev_position
        # (see the first-frame branch above), so this is never None here.
        assert history.prev_velocity_smooth is not None
        prev_vx_smooth, prev_vy_smooth = history.prev_velocity_smooth
        vx_smooth = self._alpha * vx + (1 - self._alpha) * prev_vx_smooth
        vy_smooth = self._alpha * vy + (1 - self._alpha) * prev_vy_smooth

        # --- derived from the smoothed velocity ---
        speed = math.hypot(vx_smooth, vy_smooth)
        direction = math.atan2(vy_smooth, vx_smooth)

        # --- acceleration: change in (smoothed) velocity / change in time ---
        ax = (vx_smooth - prev_vx_smooth) / dt
        ay = (vy_smooth - prev_vy_smooth) / dt

        # --- save this frame's values for next frame's calculation ---
        history.prev_position = position_px
        history.prev_velocity_smooth = (vx_smooth, vy_smooth)

        return VelocityResult(
            velocity_px_s=(vx_smooth, vy_smooth),
            speed_px_s=speed,
            direction_rad=direction,
            acceleration_px_s2=(ax, ay),
        )

    def clear(self, hand_id: int) -> None:
        """Forget a hand's history (call this when the hand disappears)."""
        self._history.pop(hand_id, None)
