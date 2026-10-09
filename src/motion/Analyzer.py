
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from src.motion.motion import (
    DepthTrend,
    DepthTrendDetector,
    FingertipTrail,
    StationaryDetector,
    TrajectoryBuffer,
    VelocityTracker,
)

if TYPE_CHECKING:  # typing only, keeps update_raw() usable without geometry
    from src.geometry.calculator import HandGeometry

Point = tuple[float, float]
INDEX_TIP = 8  # MediaPipe landmark id for the index fingertip


@dataclass
class MotionConfig:
    # trajectory
    max_trajectory_len: int = 60
    # velocity
    velocity_ema_alpha: float = 0.3
    # stationary (thresholds are in hand-sizes per second)
    stationary_enter: float = 0.10
    stationary_exit: float = 0.25
    stationary_min_frames: int = 5
    # depth trend
    depth_alpha: float = 0.4
    depth_window: int = 5
    depth_ratio_thresh: float = 1.02
    # fingertip trail
    fingertip_grace_frames: int = 3
    # a gap longer than this (s) means the hand was lost: reset velocity,
    # stationary and depth state instead of computing a huge fake velocity
    max_dt: float = 0.25


@dataclass
class MotionState:
    velocity_px_s: Point
    speed_px_s: float
    direction_rad: float
    acceleration_px_s2: Point
    trajectory: list[Point]
    fingertip_trajectory: list[Point]
    is_stationary: bool
    depth_trend: DepthTrend

    @property
    def is_moving_toward_camera(self) -> bool:
        return self.depth_trend is DepthTrend.TOWARD

    @property
    def is_moving_away_from_camera(self) -> bool:
        return self.depth_trend is DepthTrend.AWAY


def _read_geometry(geometry: "HandGeometry") -> tuple[Point, float, Point | None]:
    """Return (palm_center_px, hand_size_px, index_tip_px).
    """
    from src.geometry.calculator import hand_size, palm_center

    center = palm_center(geometry)
    size = hand_size(geometry)
    try:
        lm = geometry.landmarks[INDEX_TIP]
        tip: Point | None = (float(lm[0]), float(lm[1]))
    except (AttributeError, IndexError, TypeError):
        tip = None
    return (float(center[0]), float(center[1])), float(size), tip

@dataclass
class _HandTrack:
    trajectory: TrajectoryBuffer
    stationary: StationaryDetector
    depth: DepthTrendDetector
    fingertip: FingertipTrail
    clock: float = 0.0  # seconds accumulated from dt, used as timestamps


class MotionAnalyzer:
    def __init__(self, config: MotionConfig | None = None) -> None:
        self.cfg = config or MotionConfig()
        self._velocity = VelocityTracker(self.cfg.velocity_ema_alpha)
        self._tracks: dict[int, _HandTrack] = {}

    #main entry point
    def update(
        self,
        hand_id: int,
        geometry: "HandGeometry",
        dt: float,
        is_pointing: bool = False,
    ) -> MotionState:
        """is_pointing comes from the gesture layer (fingertip trail only
        records while it is True)."""
        center, size, tip = _read_geometry(geometry)
        return self.update_raw(hand_id, center, size, dt, tip, is_pointing)

    def update_raw(
        self,
        hand_id: int,
        center: Point,
        size: float,
        dt: float,
        tip: Point | None = None,
        is_pointing: bool = False,
    ) -> MotionState:
        cfg = self.cfg
        track = self._tracks.get(hand_id)
        if track is None:
            track = self._tracks[hand_id] = self._new_track()

        if dt > cfg.max_dt:  # hand was lost for a while: start fresh
            self._velocity.clear(hand_id)
            track.stationary.reset()
            track.depth.reset()

        track.clock += max(dt, 0.0)

        vel = self._velocity.update(hand_id, center, dt)
        track.trajectory.append(center[0], center[1], track.clock)
        stationary = track.stationary.update(vel.speed_px_s, size)
        depth = track.depth.update(size)
        fingertip = track.fingertip.update(tip, track.clock, is_pointing)

        return MotionState(
            velocity_px_s=vel.velocity_px_s,
            speed_px_s=vel.speed_px_s,
            direction_rad=vel.direction_rad,
            acceleration_px_s2=vel.acceleration_px_s2,
            trajectory=track.trajectory.points(),
            fingertip_trajectory=fingertip,
            is_stationary=stationary,
            depth_trend=depth,
        )

    # per-hand access / cleanup 
    def get_trajectory(self, hand_id: int) -> list[Point]:
        track = self._tracks.get(hand_id)
        return track.trajectory.points() if track else []

    def get_fingertip_trajectory(self, hand_id: int) -> list[Point]:
        track = self._tracks.get(hand_id)
        return track.fingertip.buffer.points() if track else []

    def clear(self, hand_id: int) -> None:
        self._velocity.clear(hand_id)
        self._tracks.pop(hand_id, None)

    def retain_only(self, visible_ids: Iterable[int]) -> None:
        """Call once per frame with the ids seen this frame to drop the rest."""
        keep = set(visible_ids)
        for hand_id in [h for h in self._tracks if h not in keep]:
            self.clear(hand_id)

    # internals
    def _new_track(self) -> _HandTrack:
        c = self.cfg
        return _HandTrack(
            trajectory=TrajectoryBuffer(c.max_trajectory_len),
            stationary=StationaryDetector(
                c.stationary_enter, c.stationary_exit, c.stationary_min_frames
            ),
            depth=DepthTrendDetector(c.depth_alpha, c.depth_window, c.depth_ratio_thresh),
            fingertip=FingertipTrail(c.max_trajectory_len, c.fingertip_grace_frames),
        )