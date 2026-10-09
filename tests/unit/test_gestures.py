"""Unit tests for the gesture subsystem.

Pure logic: hands are built from synthetic landmarks, no camera, no MediaPipe.
Adjust the import below if your module lives elsewhere.
"""

import random
from collections.abc import Iterable

import numpy as np
import pytest
from src.geometry.calculator import FloatArray, HandGeometry
from src.gestures.recognizer import (
    FINGERS,
    Gesture,
    GestureConfig,
    GestureRecognizer,
    PoseStabilizer,
    PoseState,
    SpellDetector,
    StabilityConfig,
    classify_hand,
    extended_fingers,
    hands_together,
    is_ikkon,
    is_shield,
    is_thumb_middle_pinch,
)

IMAGE_W, IMAGE_H = 640, 480
ALL_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
PINCH_FINGERS = ("index", "ring", "pinky")  # fingers that stay straight in a thumb-middle pinch

# (first landmark index of the finger, x position of the finger column)
_FINGER_COLUMNS = {
    "index": (5, -0.10),
    "middle": (9, 0.00),
    "ring": (13, 0.10),
    "pinky": (17, 0.20),
}


# Synthetic hand builders
def make_landmarks(extended: Iterable[str], *, pinch: bool = False) -> FloatArray:
    """Build 21 wrist-centered landmarks where only `extended` fingers are straight.

    Straight finger: tip is far from the MCP (ratio ~2.5 for fingers, ~2 for the thumb).
    Curled finger: tip folds back close to the MCP (ratio well below the thresholds).
    With `pinch=True`, the thumb tip and the middle tip touch each other.
    """
    ext = set(extended)
    lm = np.zeros((21, 2), dtype=np.float64)

    # Thumb: 1 = CMC, 2 = MCP, 3 = IP, 4 = tip
    lm[1] = (-0.15, -0.05)
    lm[2] = (-0.25, -0.15)
    lm[3] = (-0.40, -0.25)
    lm[4] = (-0.55, -0.35) if "thumb" in ext else (-0.20, -0.30)

    # Other fingers: MCP, PIP, DIP, tip
    for name, (first, x) in _FINGER_COLUMNS.items():
        tip_y = -0.90 if name in ext else -0.45
        lm[first] = (x, -0.40)
        lm[first + 1] = (x, -0.60)
        lm[first + 2] = (x, -0.75)
        lm[first + 3] = (x, tip_y)

    if pinch:
        lm[4] = (-0.05, -0.50)  # thumb tip
        lm[12] = (-0.03, -0.50)  # middle tip, ~0.02 away (ratio ~0.05 of the palm length)
    return lm


def make_geometry(
    extended: Iterable[str] = (),
    *,
    center: tuple[float, float] = (320.0, 240.0),
    size_px: float = 100.0,
    scale: float = 1.0,
    pinch: bool = False,
    angle_deg: float = 0.0,
) -> HandGeometry:
    """Build a HandGeometry for a hand with the given extended fingers.

    `angle_deg` rotates the hand in the image plane (0 = fingers pointing up).
    """
    landmarks = make_landmarks(extended, pinch=pinch) * scale
    if angle_deg:
        theta = np.radians(angle_deg)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        landmarks = landmarks @ rot.T
    return HandGeometry(
        palm_center_px=center,
        palm_center_norm=(center[0] / IMAGE_W, center[1] / IMAGE_H),
        hand_size_px=size_px,
        hand_size_norm=size_px / max(IMAGE_W, IMAGE_H),
        palm_angle_rad=0.0,
        landmarks_norm=landmarks,
    )


LEFT = (150.0, 240.0)
RIGHT = (500.0, 240.0)


def palm(center: tuple[float, float] = LEFT, angle_deg: float = 0.0) -> HandGeometry:
    return make_geometry(ALL_FINGERS, center=center, angle_deg=angle_deg)


def fist(center: tuple[float, float] = LEFT) -> HandGeometry:
    return make_geometry((), center=center)


def pointing(center: tuple[float, float] = LEFT) -> HandGeometry:
    return make_geometry(("index",), center=center)


def pinch_hand(center: tuple[float, float] = LEFT, angle_deg: float = 0.0) -> HandGeometry:
    return make_geometry(PINCH_FINGERS, center=center, pinch=True, angle_deg=angle_deg)


def far_pair(a: HandGeometry, b: HandGeometry) -> list[HandGeometry]:
    """Two hands far apart (not 'together')."""
    return [a, b]


def close_pair() -> list[HandGeometry]:
    """Two pinch hands almost touching (20px apart, hand size 100px)."""
    return [pinch_hand((310.0, 240.0)), pinch_hand((330.0, 240.0))]


def feed_recognizer(
    recognizer: GestureRecognizer, geometries: list[HandGeometry], n: int
) -> list[list[Gesture]]:
    """Feed the same frame n times, return the gestures reported at each frame."""
    return [[s.gesture for s in recognizer.update(geometries)] for _ in range(n)]


def activate(stabilizer: PoseStabilizer) -> None:
    cfg = stabilizer.config
    for _ in range(cfg.enter_frames + cfg.confirm_frames):
        status = stabilizer.update(True)
    assert status.state is PoseState.ACTIVE


# Defaults match gestures.md
def test_default_stability_matches_spec() -> None:
    cfg = StabilityConfig()
    assert (cfg.enter_frames, cfg.confirm_frames, cfg.exit_frames) == (3, 5, 3)
    assert cfg.ema_alpha == pytest.approx(0.3)


def test_default_thresholds_match_spec() -> None:
    cfg = GestureConfig()
    assert cfg.thumb_threshold == pytest.approx(1.3)
    assert cfg.finger_threshold == pytest.approx(1.5)


def test_default_spell_thresholds() -> None:
    cfg = GestureConfig()
    assert cfg.together_max_ratio == pytest.approx(1.0)
    assert cfg.pinch_max_ratio == pytest.approx(0.3)


def test_finger_landmark_indices_match_spec() -> None:
    expected = {
        "thumb": (4, 3, 2),
        "index": (8, 6, 5),
        "middle": (12, 10, 9),
        "ring": (16, 14, 13),
        "pinky": (20, 18, 17),
    }
    assert {f.name: (f.tip, f.pip, f.mcp) for f in FINGERS} == expected


# Single-hand classification
@pytest.mark.parametrize(
    ("extended", "expected"),
    [
        ((), Gesture.CLOSED_FIST),
        (ALL_FINGERS, Gesture.OPEN_PALM),
        (("index",), Gesture.POINTING),
        (("index", "middle"), Gesture.PEACE),
        (("thumb",), Gesture.NONE),  # THUMBS_UP no longer exists
        (("ring",), Gesture.NONE),
        (("thumb", "index"), Gesture.NONE),
        (("index", "middle", "ring"), Gesture.NONE),
        (("index", "middle", "ring", "pinky"), Gesture.NONE),  # 4 fingers, no thumb
    ],
)
def test_classify_hand(extended: tuple[str, ...], expected: Gesture) -> None:
    assert classify_hand(make_geometry(extended)) is expected


@pytest.mark.parametrize(
    "extended",
    [(), ("index",), ("index", "middle"), ("thumb",), ALL_FINGERS],
)
def test_extended_fingers_reports_exact_set(extended: tuple[str, ...]) -> None:
    result = extended_fingers(make_landmarks(extended), GestureConfig())
    assert result == frozenset(extended)


def test_classification_is_scale_invariant() -> None:
    for scale in (0.2, 1.0, 5.0):
        assert classify_hand(make_geometry(ALL_FINGERS, scale=scale)) is Gesture.OPEN_PALM
        assert classify_hand(make_geometry((), scale=scale)) is Gesture.CLOSED_FIST
        assert classify_hand(make_geometry(PINCH_FINGERS, pinch=True, scale=scale)) is (
            Gesture.THUMB_MIDDLE_PINCH
        )


@pytest.mark.parametrize("angle", [0.0, 45.0, 90.0, 180.0, -135.0])
def test_classification_is_rotation_invariant(angle: float) -> None:
    assert classify_hand(make_geometry(ALL_FINGERS, angle_deg=angle)) is Gesture.OPEN_PALM
    assert classify_hand(make_geometry((), angle_deg=angle)) is Gesture.CLOSED_FIST
    assert classify_hand(pinch_hand(angle_deg=angle)) is Gesture.THUMB_MIDDLE_PINCH


def test_tiny_hand_is_none_not_fist() -> None:
    ghost = make_geometry((), size_px=5.0)  # hand_size_norm < min_hand_size_norm
    assert classify_hand(ghost) is Gesture.NONE


def test_classify_hand_uses_default_config_when_none_given() -> None:
    geometry = make_geometry(ALL_FINGERS)
    assert classify_hand(geometry, None) is classify_hand(geometry, GestureConfig())


def test_thresholds_are_configurable() -> None:
    strict = GestureConfig(thumb_threshold=10.0, finger_threshold=10.0)
    # Nothing can reach such a ratio: an open hand now reads as a fist.
    assert classify_hand(make_geometry(ALL_FINGERS), strict) is Gesture.CLOSED_FIST


def test_min_hand_size_is_configurable() -> None:
    permissive = GestureConfig(min_hand_size_norm=0.0)
    assert classify_hand(make_geometry((), size_px=5.0), permissive) is Gesture.CLOSED_FIST


# Thumb-middle pinch
def test_pinch_hand_is_classified_as_pinch() -> None:
    assert classify_hand(pinch_hand()) is Gesture.THUMB_MIDDLE_PINCH


def test_open_palm_is_not_a_pinch() -> None:
    # All five fingers straight, thumb far from the middle tip: palm wins.
    assert not is_thumb_middle_pinch(palm())
    assert classify_hand(palm()) is Gesture.OPEN_PALM


def test_pinch_requires_index_ring_and_pinky_extended() -> None:
    # Tips touch, but the index is curled: not an Ikkon pinch.
    hand = make_geometry(("ring", "pinky"), pinch=True)
    assert is_thumb_middle_pinch(hand)  # the raw distance test still passes
    assert classify_hand(hand) is Gesture.NONE


def test_pinch_distance_threshold_is_configurable() -> None:
    strict = GestureConfig(pinch_max_ratio=0.01)  # the synthetic pinch is ~0.05
    assert not is_thumb_middle_pinch(pinch_hand(), strict)
    assert classify_hand(pinch_hand(), strict) is Gesture.NONE


def test_pinch_rejects_tiny_ghost_hand() -> None:
    ghost = make_geometry(PINCH_FINGERS, pinch=True, size_px=5.0)
    assert not is_thumb_middle_pinch(ghost)


def test_pinch_rejects_degenerate_landmarks() -> None:
    # All landmarks on the wrist: palm length is zero, nothing to compare against.
    flat = make_geometry(PINCH_FINGERS, pinch=True, scale=0.0)
    assert not is_thumb_middle_pinch(flat)


def test_pinch_uses_default_config_when_none_given() -> None:
    assert is_thumb_middle_pinch(pinch_hand(), None)


# Two-hand gestures
def test_hands_together_when_close() -> None:
    hands = [palm((300.0, 240.0)), palm((350.0, 240.0))]  # 50px apart, size 100px
    assert hands_together(hands)


def test_hands_not_together_when_far() -> None:
    assert not hands_together(far_pair(palm(LEFT), palm(RIGHT)))


def test_hands_together_boundary_is_strict() -> None:
    # ratio == 1.0 is NOT below the threshold.
    hands = [palm((300.0, 240.0)), palm((400.0, 240.0))]
    assert not hands_together(hands)


def test_hands_together_depends_on_hand_size_not_pixels() -> None:
    # Same pixel distance, but bigger hands (closer to the camera) -> together.
    small = [
        make_geometry(center=(300.0, 240.0), size_px=50.0),
        make_geometry(center=(400.0, 240.0), size_px=50.0),
    ]
    big = [
        make_geometry(center=(300.0, 240.0), size_px=300.0),
        make_geometry(center=(400.0, 240.0), size_px=300.0),
    ]
    assert not hands_together(small)
    assert hands_together(big)


@pytest.mark.parametrize("count", [0, 1, 3])
def test_hands_together_needs_exactly_two_hands(count: int) -> None:
    hands = [palm((300.0 + i, 240.0)) for i in range(count)]
    assert not hands_together(hands)


def test_hands_together_zero_size_is_false() -> None:
    hands = [make_geometry(size_px=0.0), make_geometry(size_px=0.0)]
    assert not hands_together(hands)


def test_together_threshold_is_configurable() -> None:
    hands = [palm((300.0, 240.0)), palm((400.0, 240.0))]  # ratio 1.0
    assert hands_together(hands, GestureConfig(together_max_ratio=2.0))


# The shield of Seraphim
def test_shield_two_open_palms() -> None:
    assert is_shield(far_pair(palm(LEFT), palm(RIGHT)))


def test_shield_two_fists() -> None:
    assert is_shield(far_pair(fist(LEFT), fist(RIGHT)))


def test_shield_rejects_mixed_palm_and_fist() -> None:
    assert not is_shield(far_pair(palm(LEFT), fist(RIGHT)))


def test_shield_rejects_other_gestures() -> None:
    assert not is_shield(far_pair(pointing(LEFT), pointing(RIGHT)))


def test_shield_rejects_pinch_hands() -> None:
    assert not is_shield(far_pair(pinch_hand(LEFT), pinch_hand(RIGHT)))


@pytest.mark.parametrize("count", [0, 1, 3])
def test_shield_needs_exactly_two_hands(count: int) -> None:
    assert not is_shield([palm(LEFT) for _ in range(count)])


def test_shield_rejects_tiny_ghost_hands() -> None:
    ghosts = [
        make_geometry((), size_px=5.0, center=LEFT),
        make_geometry((), size_px=5.0, center=RIGHT),
    ]
    assert not is_shield(ghosts)


# The images of Ikkon
def test_ikkon_two_pinch_hands() -> None:
    assert is_ikkon(far_pair(pinch_hand(LEFT), pinch_hand(RIGHT)))


@pytest.mark.parametrize(
    "hands",
    [
        far_pair(pinch_hand(LEFT), palm(RIGHT)),
        far_pair(pinch_hand(LEFT), fist(RIGHT)),
        far_pair(palm(LEFT), palm(RIGHT)),
        far_pair(fist(LEFT), fist(RIGHT)),
    ],
)
def test_ikkon_rejects_non_pinch_hands(hands: list[HandGeometry]) -> None:
    assert not is_ikkon(hands)


@pytest.mark.parametrize("count", [0, 1, 3])
def test_ikkon_needs_exactly_two_hands(count: int) -> None:
    assert not is_ikkon([pinch_hand(LEFT) for _ in range(count)])


def test_ikkon_does_not_depend_on_hand_distance() -> None:
    assert is_ikkon(close_pair())
    assert is_ikkon(far_pair(pinch_hand(LEFT), pinch_hand(RIGHT)))


@pytest.mark.parametrize("angle", [0.0, 45.0, -90.0, 180.0])
def test_ikkon_does_not_depend_on_hand_rotation(angle: float) -> None:
    hands = far_pair(pinch_hand(LEFT, angle), pinch_hand(RIGHT, -angle))
    assert is_ikkon(hands)


def test_ikkon_rejects_tiny_ghost_hands() -> None:
    ghosts = [
        make_geometry(PINCH_FINGERS, pinch=True, size_px=5.0, center=LEFT),
        make_geometry(PINCH_FINGERS, pinch=True, size_px=5.0, center=RIGHT),
    ]
    assert not is_ikkon(ghosts)


def test_ikkon_uses_config_pinch_threshold() -> None:
    strict = GestureConfig(pinch_max_ratio=0.01)  # the synthetic pinch is ~0.05
    assert not is_ikkon(far_pair(pinch_hand(LEFT), pinch_hand(RIGHT)), strict)


# PoseStabilizer (state machine)
def test_stabilizer_starts_idle() -> None:
    status = PoseStabilizer().update(False)
    assert status.state is PoseState.IDLE
    assert status.frames_held == 0


def test_stabilizer_idle_candidate_active_timeline() -> None:
    stab = PoseStabilizer()  # enter=3, confirm=5
    states = [stab.update(True).state for _ in range(8)]
    assert states == [
        PoseState.IDLE,
        PoseState.IDLE,
        PoseState.CANDIDATE,
        PoseState.CANDIDATE,
        PoseState.CANDIDATE,
        PoseState.CANDIDATE,
        PoseState.CANDIDATE,
        PoseState.ACTIVE,
    ]


def test_stabilizer_frames_held_counts_from_one() -> None:
    stab = PoseStabilizer()
    activate(stab)
    assert stab.update(True).frames_held == 2
    assert stab.update(True).frames_held == 3


def test_stabilizer_tolerates_short_dropouts_when_active() -> None:
    stab = PoseStabilizer()  # exit=3
    activate(stab)
    assert stab.update(False).state is PoseState.ACTIVE
    assert stab.update(False).state is PoseState.ACTIVE


def test_stabilizer_exits_after_exit_frames_lost() -> None:
    stab = PoseStabilizer()
    activate(stab)
    stab.update(False)
    stab.update(False)
    status = stab.update(False)
    assert status.state is PoseState.IDLE
    assert status.frames_held == 0


def test_stabilizer_lost_streak_resets_when_pose_returns() -> None:
    stab = PoseStabilizer()
    activate(stab)
    for seen in (False, False, True, False, False):
        status = stab.update(seen)
    assert status.state is PoseState.ACTIVE


def test_stabilizer_candidate_drops_on_single_miss_and_restarts() -> None:
    stab = PoseStabilizer()
    for _ in range(3):
        status = stab.update(True)
    assert status.state is PoseState.CANDIDATE
    assert stab.update(False).state is PoseState.IDLE
    # The streak restarted from zero.
    assert stab.update(True).state is PoseState.IDLE
    assert stab.update(True).state is PoseState.IDLE
    assert stab.update(True).state is PoseState.CANDIDATE


def test_stabilizer_reactivation_after_exit_needs_full_streak() -> None:
    stab = PoseStabilizer()
    activate(stab)
    for _ in range(3):
        stab.update(False)
    states = [stab.update(True).state for _ in range(8)]
    assert states[-1] is PoseState.ACTIVE
    assert PoseState.ACTIVE not in states[:-1]


def test_stabilizer_confidence_follows_ema() -> None:
    stab = PoseStabilizer()  # alpha = 0.3
    assert stab.update(True).confidence == pytest.approx(0.3)
    assert stab.update(True).confidence == pytest.approx(0.51)
    assert stab.update(False).confidence == pytest.approx(0.357)


def test_stabilizer_confidence_stays_in_unit_interval() -> None:
    stab = PoseStabilizer()
    rng = random.Random(0)
    for _ in range(300):
        confidence = stab.update(rng.random() > 0.5).confidence
        assert 0.0 <= confidence <= 1.0


def test_stabilizer_reset_clears_everything() -> None:
    stab = PoseStabilizer()
    activate(stab)
    stab.reset()
    status = stab.update(False)
    assert status.state is PoseState.IDLE
    assert status.frames_held == 0
    assert status.confidence == pytest.approx(0.0)


def test_stabilizer_custom_config() -> None:
    stab = PoseStabilizer(StabilityConfig(enter_frames=1, confirm_frames=1, exit_frames=1))
    assert stab.update(True).state is PoseState.CANDIDATE
    assert stab.update(True).state is PoseState.ACTIVE
    assert stab.update(False).state is PoseState.IDLE


# Flicker resistance
def test_flicker_alternating_never_activates() -> None:
    stab = PoseStabilizer()
    for i in range(200):
        assert stab.update(i % 2 == 0).state is not PoseState.ACTIVE


def test_flicker_short_streaks_never_activate() -> None:
    stab = PoseStabilizer()  # needs 8 consecutive frames
    for _ in range(30):
        for _ in range(7):
            assert stab.update(True).state is not PoseState.ACTIVE
        assert stab.update(False).state is PoseState.IDLE


def test_flicker_single_dropouts_never_deactivate() -> None:
    stab = PoseStabilizer()
    activate(stab)
    for i in range(300):
        assert stab.update(i % 3 != 0).state is PoseState.ACTIVE


def test_flicker_random_dropouts_shorter_than_exit_never_deactivate() -> None:
    stab = PoseStabilizer()
    activate(stab)
    rng = random.Random(1234)
    lost_run = 0
    for _ in range(1000):
        # Force a hit before the lost run reaches exit_frames.
        seen = rng.random() > 0.4 or lost_run >= stab.config.exit_frames - 1
        lost_run = 0 if seen else lost_run + 1
        assert stab.update(seen).state is PoseState.ACTIVE


# GestureRecognizer
def test_recognizer_no_hands_returns_empty_list() -> None:
    assert GestureRecognizer().update([]) == []


def test_recognizer_uses_default_config() -> None:
    assert GestureRecognizer().config == GestureConfig()


def test_recognizer_confirms_gesture_after_enter_plus_confirm_frames() -> None:
    rec = GestureRecognizer()
    history = feed_recognizer(rec, [palm()], 8)
    assert [h[0] for h in history[:7]] == [Gesture.NONE] * 7
    assert history[7][0] is Gesture.OPEN_PALM


def test_recognizer_state_fields() -> None:
    rec = GestureRecognizer()
    for _ in range(7):
        rec.update([palm()])
    state = rec.update([palm()])[0]
    assert state.gesture is Gesture.OPEN_PALM
    assert state.hand_id == 0
    assert state.frames_held == 1
    assert state.confidence == pytest.approx(1 - 0.7**8)
    assert rec.update([palm()])[0].frames_held == 2


def test_recognizer_unconfirmed_state_has_zero_confidence_and_frames() -> None:
    state = GestureRecognizer().update([palm()])[0]
    assert state.gesture is Gesture.NONE
    assert state.confidence == 0.0
    assert state.frames_held == 0


def test_recognizer_tracks_each_hand_independently() -> None:
    rec = GestureRecognizer()
    frame = [palm(LEFT), fist(RIGHT)]
    history = feed_recognizer(rec, frame, 8)
    assert history[-1] == [Gesture.OPEN_PALM, Gesture.CLOSED_FIST]
    states = rec.update(frame)
    assert [s.hand_id for s in states] == [0, 1]


def test_recognizer_survives_single_misclassified_frame() -> None:
    rec = GestureRecognizer()
    feed_recognizer(rec, [palm()], 8)
    glitch = rec.update([fist()])[0]  # one wrong frame
    back = rec.update([palm()])[0]
    assert glitch.gesture is Gesture.OPEN_PALM
    assert back.gesture is Gesture.OPEN_PALM


def test_recognizer_gesture_switch_is_clean() -> None:
    rec = GestureRecognizer()
    feed_recognizer(rec, [palm()], 8)
    history = [h[0] for h in feed_recognizer(rec, [fist()], 8)]
    # Palm survives exit_frames-1 frames, then nothing is reported until
    # the fist is confirmed. Two gestures are never reported at once.
    assert history == [
        Gesture.OPEN_PALM,
        Gesture.OPEN_PALM,
        Gesture.NONE,
        Gesture.NONE,
        Gesture.NONE,
        Gesture.NONE,
        Gesture.NONE,
        Gesture.CLOSED_FIST,
    ]


def test_recognizer_resets_state_when_hand_is_lost() -> None:
    rec = GestureRecognizer()
    feed_recognizer(rec, [palm()], 8)
    assert rec.update([]) == []
    # Hand is back: it must ramp up again from scratch.
    assert rec.update([palm()])[0].gesture is Gesture.NONE


def test_recognizer_resets_only_the_lost_hand() -> None:
    rec = GestureRecognizer()
    frame = [palm(LEFT), palm(RIGHT)]
    feed_recognizer(rec, frame, 8)
    rec.update([palm(LEFT)])  # second hand disappears
    states = rec.update(frame)  # and comes back
    assert states[0].gesture is Gesture.OPEN_PALM
    assert states[1].gesture is Gesture.NONE


def test_recognizer_reset_forgets_everything() -> None:
    rec = GestureRecognizer()
    feed_recognizer(rec, [palm()], 8)
    rec.reset()
    assert rec.update([palm()])[0].gesture is Gesture.NONE


def test_recognizer_two_hands_together() -> None:
    rec = GestureRecognizer()
    frame = [fist((300.0, 240.0)), fist((340.0, 240.0))]
    history = feed_recognizer(rec, frame, 8)
    assert history[-1] == [Gesture.TWO_HANDS_TOGETHER, Gesture.TWO_HANDS_TOGETHER]


def test_recognizer_custom_stability_config() -> None:
    stability = StabilityConfig(enter_frames=1, confirm_frames=1)
    rec = GestureRecognizer(GestureConfig(stability=stability))
    history = feed_recognizer(rec, [palm()], 2)
    assert history[0][0] is Gesture.NONE
    assert history[1][0] is Gesture.OPEN_PALM


def test_recognizer_reports_pinch_on_each_hand() -> None:
    rec = GestureRecognizer()
    frame = far_pair(pinch_hand(LEFT), pinch_hand(RIGHT))
    history = feed_recognizer(rec, frame, 8)
    assert history[-1] == [Gesture.THUMB_MIDDLE_PINCH, Gesture.THUMB_MIDDLE_PINCH]


def test_recognizer_reports_together_instead_of_pinch_when_hands_touch() -> None:
    # Documented behaviour: TWO_HANDS_TOGETHER overrides single-hand gestures.
    # Spell detectors (is_ikkon, ...) are not affected: they read the raw geometries.
    rec = GestureRecognizer()
    history = feed_recognizer(rec, close_pair(), 8)
    assert history[-1] == [Gesture.TWO_HANDS_TOGETHER, Gesture.TWO_HANDS_TOGETHER]


# SpellDetector (stabilised spells)
def run_spell(det: SpellDetector, frame: list[HandGeometry], n: int) -> PoseState:
    status = det.update(frame)
    for _ in range(n - 1):
        status = det.update(frame)
    return status.state


def test_spell_detector_uses_default_config() -> None:
    assert SpellDetector(is_shield).config == GestureConfig()


def test_spell_detector_passes_geometries_and_config_to_predicate() -> None:
    calls: list[tuple[list[HandGeometry], GestureConfig]] = []

    def predicate(geometries: list[HandGeometry], config: GestureConfig) -> bool:
        calls.append((geometries, config))
        return True

    cfg = GestureConfig(together_max_ratio=2.0)
    frame = [palm()]
    SpellDetector(predicate, cfg).update(frame)
    assert len(calls) == 1
    assert calls[0][0] is frame
    assert calls[0][1] is cfg


def test_spell_detector_accepts_any_predicate() -> None:
    always = SpellDetector(lambda geometries, config: True)
    assert run_spell(always, [], 8) is PoseState.ACTIVE


def test_spell_detector_respects_custom_stability() -> None:
    cfg = GestureConfig(stability=StabilityConfig(enter_frames=1, confirm_frames=1))
    det = SpellDetector(is_shield, cfg)
    assert run_spell(det, far_pair(palm(LEFT), palm(RIGHT)), 2) is PoseState.ACTIVE


# Shield through SpellDetector
def test_shield_detector_activates_with_two_palms() -> None:
    det = SpellDetector(is_shield)
    assert run_spell(det, far_pair(palm(LEFT), palm(RIGHT)), 8) is PoseState.ACTIVE


def test_shield_detector_activates_with_two_fists() -> None:
    det = SpellDetector(is_shield)
    assert run_spell(det, far_pair(fist(LEFT), fist(RIGHT)), 8) is PoseState.ACTIVE


def test_shield_detector_not_active_before_confirmation() -> None:
    det = SpellDetector(is_shield)
    assert run_spell(det, far_pair(palm(LEFT), palm(RIGHT)), 7) is PoseState.CANDIDATE


def test_shield_detector_never_activates_for_mixed_hands() -> None:
    det = SpellDetector(is_shield)
    assert run_spell(det, far_pair(palm(LEFT), fist(RIGHT)), 50) is PoseState.IDLE


def test_shield_detector_survives_one_hand_flicker() -> None:
    det = SpellDetector(is_shield)
    shield = far_pair(palm(LEFT), palm(RIGHT))
    run_spell(det, shield, 8)
    assert det.update([palm(LEFT)]).state is PoseState.ACTIVE  # one hand lost for a frame
    assert det.update(shield).state is PoseState.ACTIVE


def test_shield_detector_deactivates_when_hands_leave() -> None:
    det = SpellDetector(is_shield)
    run_spell(det, far_pair(palm(LEFT), palm(RIGHT)), 8)
    assert run_spell(det, [], 3) is PoseState.IDLE


def test_shield_detector_alternating_frames_never_activate() -> None:
    det = SpellDetector(is_shield)
    shield = far_pair(fist(LEFT), fist(RIGHT))
    for i in range(100):
        frame = shield if i % 2 == 0 else []
        assert det.update(frame).state is not PoseState.ACTIVE


def test_shield_detector_reset() -> None:
    det = SpellDetector(is_shield)
    run_spell(det, far_pair(palm(LEFT), palm(RIGHT)), 8)
    det.reset()
    assert det.update(far_pair(palm(LEFT), palm(RIGHT))).state is PoseState.IDLE


# Images of Ikkon through SpellDetector
def test_ikkon_detector_activates_with_two_pinches() -> None:
    det = SpellDetector(is_ikkon)
    assert run_spell(det, far_pair(pinch_hand(LEFT), pinch_hand(RIGHT)), 8) is PoseState.ACTIVE


def test_ikkon_detector_never_activates_for_a_shield_pose() -> None:
    det = SpellDetector(is_ikkon)
    assert run_spell(det, far_pair(palm(LEFT), palm(RIGHT)), 50) is PoseState.IDLE


def test_shield_detector_never_activates_for_an_ikkon_pose() -> None:
    det = SpellDetector(is_shield)
    assert run_spell(det, far_pair(pinch_hand(LEFT), pinch_hand(RIGHT)), 50) is PoseState.IDLE


def test_ikkon_detector_survives_single_frame_dropouts() -> None:
    det = SpellDetector(is_ikkon)
    pose = far_pair(pinch_hand(LEFT), pinch_hand(RIGHT))
    run_spell(det, pose, 8)
    for _ in range(20):
        assert det.update([]).state is PoseState.ACTIVE
        assert det.update(pose).state is PoseState.ACTIVE
