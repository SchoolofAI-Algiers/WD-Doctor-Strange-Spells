from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from src.geometry.calculator import FloatArray, HandGeometry, distance

if TYPE_CHECKING:  # typing only, avoids a circular import
    from src.motion.analyzer import MotionState


class Gesture(Enum):
    NONE = "none"
    OPEN_PALM = "open_palm"
    CLOSED_FIST = "closed_fist"
    POINTING = "pointing"
    PEACE = "peace"
    TWO_HANDS_TOGETHER = "two_hands_together"
    THUMB_MIDDLE_PINCH = "thumb_middle_pinch"


@dataclass
class GestureState:
    gesture: Gesture
    confidence: float
    frames_held: int
    hand_id: int


@dataclass(frozen=True)
class StabilityConfig:
    enter_frames: int = 3
    confirm_frames: int = 5
    exit_frames: int = 3
    ema_alpha: float = 0.3


@dataclass(frozen=True)
class GestureConfig:
    min_hand_size_norm: float = 0.05  # avoids classifying a non-existent hand as a closed fist
    thumb_threshold: float = 1.3
    finger_threshold: float = 1.5
    # Ring/pinky may read partially extended when the hand is tilted
    # (PIP-MCP foreshortened). Peace allows them up to this ratio, while a
    # true open palm still needs all four clearly out. Slightly above
    # finger_threshold on purpose so Peace is easier than Open.
    peace_curl_max_ratio: float = 1.65
    # An edge-on hand (palm seen from the side) collapses the knuckle span:
    # width(5-17) / length(0-9) drops toward 0. Below this it is not an open
    # palm no matter how straight the fingers read.
    palm_min_facing_ratio: float = 0.45
    # Actual clap = palms touching/overlapping, much tighter than "together"
    # (which is just nearby). Ruby re-arms on a clap, not on proximity.
    clap_max_ratio: float = 0.7
    together_max_ratio: float = 1.0
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    pinch_max_ratio: float = 0.3


@dataclass(frozen=True)
class FingerSpec:
    name: str
    tip: int
    pip: int
    mcp: int


FINGERS = (
    FingerSpec("thumb", tip=4, pip=3, mcp=2),
    FingerSpec("index", tip=8, pip=6, mcp=5),
    FingerSpec("middle", tip=12, pip=10, mcp=9),
    FingerSpec("ring", tip=16, pip=14, mcp=13),
    FingerSpec("pinky", tip=20, pip=18, mcp=17),
)


def is_finger_extended(landmarks_norm: FloatArray, finger: FingerSpec, threshold: float) -> bool:
    tip_to_mcp = distance(landmarks_norm[finger.tip], landmarks_norm[finger.mcp])
    pip_to_mcp = distance(landmarks_norm[finger.pip], landmarks_norm[finger.mcp])

    return tip_to_mcp > threshold * pip_to_mcp


def finger_ratio(landmarks_norm: FloatArray, finger: FingerSpec) -> float:
    """tip-MCP distance relative to pip-MCP distance (same ratio as above)."""
    pip_to_mcp = distance(landmarks_norm[finger.pip], landmarks_norm[finger.mcp])
    if pip_to_mcp < 1e-9:
        return float("inf")
    return distance(landmarks_norm[finger.tip], landmarks_norm[finger.mcp]) / pip_to_mcp


def is_peace_pose(landmarks_norm: FloatArray, config: GestureConfig) -> bool:
    """Index+middle out, ring+pinky folded (with tilt margin, thumb ignored)."""
    by_name = {f.name: f for f in FINGERS}
    if finger_ratio(landmarks_norm, by_name["index"]) <= config.finger_threshold:
        return False
    if finger_ratio(landmarks_norm, by_name["middle"]) <= config.finger_threshold:
        return False
    if finger_ratio(landmarks_norm, by_name["ring"]) >= config.peace_curl_max_ratio:
        return False
    if finger_ratio(landmarks_norm, by_name["pinky"]) >= config.peace_curl_max_ratio:
        return False
    return True


def extended_fingers(landmarks_norm: FloatArray, config: GestureConfig) -> frozenset[str]:
    return frozenset(
        finger.name
        for finger in FINGERS
        if is_finger_extended(
            landmarks_norm,
            finger,
            config.thumb_threshold if finger.name == "thumb" else config.finger_threshold,
        )
    )


def palm_facing_ratio(landmarks_norm: FloatArray) -> float:
    """Knuckle span vs palm length, scale-invariant (1 = frontal, 0 = edge-on)."""
    width = distance(landmarks_norm[5], landmarks_norm[17])
    length = distance(landmarks_norm[0], landmarks_norm[9])
    if length < 1e-9:
        return 0.0
    return width / length


def is_palm_frontal(landmarks_norm: FloatArray, config: GestureConfig) -> bool:
    """True when the palm (or back of hand) faces the camera, not edge-on."""
    return palm_facing_ratio(landmarks_norm) >= config.palm_min_facing_ratio


def is_thumb_middle_pinch(geometry: HandGeometry, config: GestureConfig | None = None) -> bool:
    config = config or GestureConfig()
    if geometry.hand_size_norm < config.min_hand_size_norm:
        return False

    lm = geometry.landmarks_norm
    # hand size is distorted by the pose itself, so use wrist -> middle MCP as scale
    scale = distance(lm[0], lm[9])
    if scale < 1e-6:
        return False
    return distance(lm[4], lm[12]) / scale < config.pinch_max_ratio


def classify_hand(geometry: HandGeometry, config: GestureConfig | None = None) -> Gesture:
    config = config or GestureConfig()

    if geometry.hand_size_norm < config.min_hand_size_norm:
        return Gesture.NONE

    fingers = extended_fingers(geometry.landmarks_norm, config)
    others = fingers - {"thumb"}  # the thumb test is unreliable, so poses use the 4 fingers

    if not others:
        return Gesture.CLOSED_FIST
    if others >= {"index", "ring", "pinky"} and is_thumb_middle_pinch(geometry, config):
        return Gesture.THUMB_MIDDLE_PINCH
    # Peace before Open: a tilted peace can read ring/pinky slightly extended,
    # which must not flip it into an open palm + shield.
    if is_peace_pose(geometry.landmarks_norm, config):
        return Gesture.PEACE
    if len(others) == 4:
        # Four straight fingers only count when the palm faces the camera:
        # edge-on hands read extended but must not trigger a shield.
        if is_palm_frontal(geometry.landmarks_norm, config):
            return Gesture.OPEN_PALM
        return Gesture.NONE
    if others == frozenset({"index"}):
        return Gesture.POINTING
    if others == frozenset({"index", "middle"}):
        return Gesture.PEACE

    return Gesture.NONE


# Two hands gesture
def hands_together(geometries: list[HandGeometry], config: GestureConfig | None = None) -> bool:
    config = config or GestureConfig()
    if len(geometries) != 2:
        return False

    a, b = geometries
    mean_size = (a.hand_size_px + b.hand_size_px) / 2
    if mean_size < 1e-6:
        return False

    d = distance(np.array(a.palm_center_px), np.array(b.palm_center_px))
    return d / mean_size < config.together_max_ratio


# The shield of seraphim
def is_shield(geometries: list[HandGeometry], config: GestureConfig | None = None) -> bool:
    if len(geometries) != 2:  # exactly 2 hands
        return False

    gestures = [classify_hand(geometry, config) for geometry in geometries]
    return gestures[0] == gestures[1] and gestures[0] in (
        Gesture.CLOSED_FIST,
        Gesture.OPEN_PALM,
    )


# The images of ikkon
def is_ikkon(geometries: list[HandGeometry], config: GestureConfig | None = None) -> bool:
    config = config or GestureConfig()
    return len(geometries) == 2 and all(
        classify_hand(g, config) is Gesture.THUMB_MIDDLE_PINCH for g in geometries
    )


# State machine
class PoseState(Enum):
    IDLE = "idle"
    CANDIDATE = "candidate"
    ACTIVE = "active"


@dataclass(frozen=True)
class PoseStatus:
    state: PoseState
    confidence: float
    frames_held: int


class PoseStabilizer:
    def __init__(self, config: StabilityConfig | None = None) -> None:
        self.config = config or StabilityConfig()
        self._state = PoseState.IDLE
        self._seen_streak = 0
        self._lost_streak = 0
        self._frames_held = 0
        self._confidence = 0.0

    def update(self, pose_seen: bool) -> PoseStatus:
        cfg = self.config
        observation = 1.0 if pose_seen else 0.0
        self._confidence = cfg.ema_alpha * observation + (1 - cfg.ema_alpha) * self._confidence

        if self._state is PoseState.ACTIVE:
            self._frames_held += 1
            if pose_seen:
                self._lost_streak = 0
            else:
                self._lost_streak += 1
                if self._lost_streak >= cfg.exit_frames:
                    self._go_idle()

        elif pose_seen:
            self._seen_streak += 1
            if self._seen_streak >= cfg.enter_frames + cfg.confirm_frames:
                self._state = PoseState.ACTIVE
                self._lost_streak = 0
                self._frames_held = 1
            elif self._seen_streak >= cfg.enter_frames:
                self._state = PoseState.CANDIDATE

        else:
            self._go_idle()

        return PoseStatus(self._state, self._confidence, self._frames_held)

    def reset(self) -> None:
        self._go_idle()
        self._confidence = 0.0

    def _go_idle(self) -> None:
        self._state = PoseState.IDLE
        self._seen_streak = 0
        self._lost_streak = 0
        self._frames_held = 0


HANDEDNESS_ID = {"Left": 0, "Right": 1}


def hand_ids_from_handedness(count: int, handedness: list[str] | None) -> list[int]:
    positional = list(range(count))
    if handedness is None or len(handedness) != count:
        return positional
    ids = [HANDEDNESS_ID.get(label, -1) for label in handedness]
    if -1 in ids or len(set(ids)) != count:
        return positional
    return ids


class GestureRecognizer:
    def __init__(self, config: GestureConfig | None = None) -> None:
        self.config = config or GestureConfig()
        self._stabilizers: dict[int, dict[Gesture, PoseStabilizer]] = {}
        self.raw: dict[int, Gesture] = {}  # raw pose per hand id, read by the two-hand spells

    def update(
        self, geometries: list[HandGeometry], handedness: list[str] | None = None
    ) -> list[GestureState]:
        ids = hand_ids_from_handedness(len(geometries), handedness)
        for lost_id in [h for h in self._stabilizers if h not in ids]:
            del self._stabilizers[lost_id]

        self.raw = {}
        states: list[GestureState] = []
        for hand_id, geometry in zip(ids, geometries, strict=True):
            raw = classify_hand(geometry, self.config)
            self.raw[hand_id] = raw
            states.append(self._update_hand(hand_id, raw))
        return states

    def reset(self) -> None:
        self._stabilizers.clear()
        self.raw = {}

    def _update_hand(self, hand_id: int, raw: Gesture) -> GestureState:
        stabilizers = self._stabilizers.setdefault(hand_id, {})

        best: tuple[Gesture, PoseStatus] | None = None
        for gesture in Gesture:
            if gesture in (Gesture.NONE, Gesture.TWO_HANDS_TOGETHER):
                continue
            if gesture not in stabilizers:
                stabilizers[gesture] = PoseStabilizer(self.config.stability)
            status = stabilizers[gesture].update(raw is gesture)
            if status.state is PoseState.ACTIVE and (
                best is None or status.confidence > best[1].confidence
            ):
                best = (gesture, status)

        if best is None:
            return GestureState(Gesture.NONE, 0.0, 0, hand_id)
        gesture, status = best
        return GestureState(gesture, status.confidence, status.frames_held, hand_id)


class SpellDetector:
    def __init__(
        self,
        predicate: Callable[[list[HandGeometry], GestureConfig], bool],
        config: GestureConfig | None = None,
    ) -> None:
        self.config = config or GestureConfig()
        self._predicate = predicate
        self._stabilizer = PoseStabilizer(self.config.stability)

    def update(self, geometries: list[HandGeometry]) -> PoseStatus:
        return self._stabilizer.update(self._predicate(geometries, self.config))

    def reset(self) -> None:
        self._stabilizer.reset()


class TwoHandContext:
    def __init__(self, config: GestureConfig | None = None) -> None:
        self.config = config or GestureConfig()
        # Reuses PoseStabilizer with the default config: 3 + 5 = 8 frames to confirm TOGETHER.
        self._together = PoseStabilizer(self.config.stability)
        self.t = 0.0
        self.together_now = False
        self.together_seen_t = -math.inf  # history: last time the hands were close (clap burst)
        self.together_active_t = -math.inf  # last time TOGETHER was confirmed (8 frames)
        self.clap_now = False  # actual touch: palms overlapping, not just nearby
        self.clap_seen_t = -math.inf  # last time a real clap touched
        self.mirror_t = -math.inf  # last time Mirror was detected (Ruby exclusion)
        self.ruby_latched = False  # stays True after clap->open until fists/loss
        self.geom: dict[int, HandGeometry] = {}
        self.pose: dict[int, Gesture] = {}
        self.motion: dict[int, MotionState] = {}

    def observe(
        self,
        t: float,
        geom: dict[int, HandGeometry],
        pose: dict[int, Gesture],
        motion: dict[int, MotionState],
    ) -> None:
        self.t, self.geom, self.pose, self.motion = t, geom, pose, motion
        self.together_now = hands_together(list(geom.values()), self.config)
        if self.together_now:
            self.together_seen_t = t
        self.clap_now = hands_together(
            list(geom.values()), replace(self.config, together_max_ratio=self.config.clap_max_ratio)
        )
        if self.clap_now:
            self.clap_seen_t = t
        if self._together.update(self.together_now).state is PoseState.ACTIVE:
            self.together_active_t = t
        # Ruby latch: set on clap->both-open, cleared on fists or hand loss.
        if not self.both():
            self.ruby_latched = False
        elif self.pose.get(0) is Gesture.CLOSED_FIST and self.pose.get(1) is Gesture.CLOSED_FIST:
            self.ruby_latched = False
        elif (
            not self.clap_now
            and self.t - self.clap_seen_t < RUBY_CLAP_WINDOW
            and self.t - self.mirror_t > RUBY_MIRROR_COOLDOWN
            and self.pose.get(0) is Gesture.OPEN_PALM
            and self.pose.get(1) is Gesture.OPEN_PALM
        ):
            self.ruby_latched = True
        if is_mirror(self):  # remember Mirror so Ruby doesn't fire right after it
            self.mirror_t = t

    def both(self) -> bool:
        return all(i in self.geom and i in self.motion and i in self.pose for i in (0, 1))


# Mirror Dimension
# TOGETHER (8 frames) -> hands FAR apart on the x axis + moving apart horizontally.
# A small parting (1-2 widths, Ruby's range) must never trigger it.
MIRROR_WINDOW = 0.8  # s after a confirmed TOGETHER during which the pull can start
MIRROR_MIN_APART = 0.8  # hand-sizes per second the distance must grow
MIRROR_MIN_DIST = 2.5  # palms must be this many hand-sizes apart (wide pull)


def mirror_metrics(ctx: TwoHandContext) -> tuple[float, bool, bool] | None:
    """(apart speed, opposite x directions, mostly horizontal) - also used for the on-screen debug."""
    if not ctx.both():
        return None
    a, b = ctx.geom[0], ctx.geom[1]
    va, vb = ctx.motion[0].velocity_px_s, ctx.motion[1].velocity_px_s
    size = (a.hand_size_px + b.hand_size_px) / 2
    if size < 1e-6:
        return None
    ux = b.palm_center_px[0] - a.palm_center_px[0]
    uy = b.palm_center_px[1] - a.palm_center_px[1]
    n = math.hypot(ux, uy) or 1.0
    apart = ((vb[0] - va[0]) * ux + (vb[1] - va[1]) * uy) / n / size  # hand-sizes/s
    opposite = va[0] * vb[0] < 0
    horizontal = all(abs(v[1]) < 0.6 * abs(v[0]) for v in (va, vb))
    return apart, opposite, horizontal


def is_mirror(ctx: TwoHandContext) -> bool:
    # Ruby owns the both-palms-open pull-apart: without this guard the same
    # motion satisfies both detectors and (with arbitration) Ruby would still
    # lose whenever its pose confirms a frame later than Mirror's motion.
    if ctx.pose.get(0) is Gesture.OPEN_PALM and ctx.pose.get(1) is Gesture.OPEN_PALM:
        return False
    # A latched Ruby owns both open hands until fists/loss: never steal it.
    if ctx.ruby_latched:
        return False
    if ctx.t - ctx.together_active_t > MIRROR_WINDOW:  # TOGETHER must have just ended
        return False
    m = mirror_metrics(ctx)
    if m is None:
        return False
    apart, opposite, horizontal = m
    if not (apart > MIRROR_MIN_APART and opposite and horizontal):
        return False
    # Wide pull only: palms must actually be far apart on screen.
    a, b = ctx.geom[0], ctx.geom[1]
    size = (a.hand_size_px + b.hand_size_px) / 2
    if size < 1e-6:
        return False
    dist = distance(np.array(a.palm_center_px), np.array(b.palm_center_px)) / size
    return dist > MIRROR_MIN_DIST


# Actual clap: palms touching/overlapping (tighter than "together").


def is_clap(ctx: TwoHandContext) -> bool:
    return ctx.clap_now


# Ruby Rings
# real clap -> both hands OPEN_PALM within < 1 s
RUBY_CLAP_WINDOW = 1.0  # s: the clap must have happened less than this ago
RUBY_MIRROR_COOLDOWN = 1.5  # s: no Ruby right after a Mirror (same hand motion)


def is_ruby(ctx: TwoHandContext) -> bool:
    # Latched in observe(): on after clap->both-open, off on both fists/loss.
    # While touching (clap) report False so the CLAP flag owns the screen.
    return bool(ctx.ruby_latched and not ctx.clap_now)


# Dr Strange Portal
# 1 hand PEACE and stationary + the other hand moving faster than the threshold
# (rotation cut: approximation)
PORTAL_MIN_SPEED = 1.0  # hand-sizes/s


def portal_roles(ctx: TwoHandContext) -> tuple[int, int] | None:
    if not ctx.both():
        return None
    for peace, mover in ((0, 1), (1, 0)):
        size = ctx.geom[mover].hand_size_px
        if size < 1e-6:  # avoids ZeroDivisionError
            continue
        if (
            ctx.pose.get(peace) is Gesture.PEACE
            and ctx.motion[peace].is_stationary
            and ctx.motion[mover].speed_px_s / size > PORTAL_MIN_SPEED
        ):
            return peace, mover
    return None


def is_portal(ctx: TwoHandContext) -> bool:
    return portal_roles(ctx) is not None
