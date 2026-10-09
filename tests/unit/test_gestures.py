"""Unit tests for the gesture subsystem.

Pure logic: hands are built from synthetic landmarks, no camera, no MediaPipe.

"""

import random
from collections.abc import Iterable

import numpy as np
import pytest
from src.geometry.calculator import FloatArray, HandGeometry
from src.gestures.recognizer import (
    FINGERS,
    HANDEDNESS_ID,
    Gesture,
    GestureConfig,
    GestureRecognizer,
    PoseStabilizer,
    PoseState,
    SpellDetector,
    StabilityConfig,
    classify_hand,
    extended_fingers,
    hand_ids_from_handedness,
    hands_together,
    is_ikkon,
    is_shield,
    is_thumb_middle_pinch,
)

IMAGE_W, IMAGE_H = 640, 480
ALL_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
FOUR_FINGERS = ("index", "middle", "ring", "pinky")  # everything but the thumb
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


def ghost_fist(center: tuple[float, float] = LEFT) -> HandGeometry:
    """A 'hand' that is too small to be real (5px -> hand_size_norm ~0.008)."""
    return make_geometry((), size_px=5.0, center=center)


def far_pair(a: HandGeometry, b: HandGeometry) -> list[HandGeometry]:
    """Two hands far apart (not 'together')."""
    return [a, b]


def close_pair() -> list[HandGeometry]:
    """Two pinch hands almost touching (20px apart, hand size 100px)."""
    return [pinch_hand((310.0, 240.0)), pinch_hand((330.0, 240.0))]


def feed_recognizer(
    recognizer: GestureRecognizer,
    geometries: list[HandGeometry],
    n: int,
    handedness: list[str] | None = None,
) -> list[list[Gesture]]:
    """Feed the same frame n times, return the gestures reported at each frame."""
    return [[s.gesture for s in recognizer.update(geometries, handedness)] for _ in range(n)]


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
    assert cfg.min_hand_size_norm == pytest.approx(0.05)


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


# Single-hand classification (the thumb is ignored, only the 4 other fingers decide)
@pytest.mark.parametrize(
    ("extended", "expected"),
    [
        ((), Gesture.CLOSED_FIST),
        (ALL_FINGERS, Gesture.OPEN_PALM),
        (("index",), Gesture.POINTING),
        (("index", "middle"), Gesture.PEACE),
        (("thumb",), Gesture.CLOSED_FIST),  # thumb alone does not count: still a fist
        (("thumb", "index"), Gesture.POINTING),  # "L" shape is still pointing
        (("thumb", "index", "middle"), Gesture.PEACE),
        (("ring",), Gesture.NONE),
        (("index", "middle", "ring"), Gesture.NONE),
        (("thumb", "index", "middle", "ring"), Gesture.NONE),
        (FOUR_FINGERS, Gesture.OPEN_PALM),  # 4 fingers, no thumb: palm
    ],
)
def test_classify_hand(extended: tuple[str, ...], expected: Gesture) -> None:
    assert classify_hand(make_geometry(extended)) is expected


@pytest.mark.parametrize(
    "others",
    [(), ("index",), ("index", "middle"), ("ring",), ("index", "middle", "ring"), FOUR_FINGERS],
)
def test_classification_ignores_thumb(others: tuple[str, ...]) -> None:
    without_thumb = classify_hand(make_geometry(others))
    with_thumb = classify_hand(make_geometry(("thumb", *others)))
    assert with_thumb is without_thumb


@pytest.mark.parametrize(
    "extended",
    [(), ("index",), ("index", "middle"), ("thumb",), ALL_FINGERS],
)
def test_extended_fingers_reports_exact_set(extended: tuple[str, ...]) -> None:
    # extended_fingers still reports the thumb: only classify_hand ignores it.
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
    assert classify_hand(ghost_fist()) is Gesture.NONE


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


def test_shield_ignores_thumb_position() -> None:
    # Fist with the thumb sticking out / four-finger palm: still a shield.
    thumb_out_fists = far_pair(
        make_geometry(("thumb",), center=LEFT), make_geometry(("thumb",), center=RIGHT)
    )
    four_finger_palms = far_pair(
        make_geometry(FOUR_FINGERS, center=LEFT), make_geometry(FOUR_FINGERS, center=RIGHT)
    )
    assert is_shield(thumb_out_fists)
    assert is_shield(four_finger_palms)


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
    assert not is_shield([ghost_fist(LEFT), ghost_fist(RIGHT)])


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


# hand_ids_from_handedness
def test_handedness_id_mapping() -> None:
    assert HANDEDNESS_ID == {"Left": 0, "Right": 1}


@pytest.mark.parametrize(
    ("count", "handedness", "expected"),
    [
        (0, None, []),
        (0, [], []),
        (1, None, [0]),
        (2, None, [0, 1]),
        (1, ["Left"], [0]),
        (1, ["Right"], [1]),  # a single right hand keeps id 1
        (2, ["Left", "Right"], [0, 1]),
        (2, ["Right", "Left"], [1, 0]),  # ids follow the label, not the position
    ],
)
def test_hand_ids_follow_handedness(
    count: int, handedness: list[str] | None, expected: list[int]
) -> None:
    assert hand_ids_from_handedness(count, handedness) == expected


@pytest.mark.parametrize(
    ("count", "handedness"),
    [
        (2, ["Left"]),  # length mismatch
        (1, ["Left", "Right"]),  # length mismatch
        (2, ["Left", "Left"]),  # duplicate labels
        (2, ["Right", "Right"]),
        (2, ["Left", "Unknown"]),  # unknown label
        (1, ["left"]),  # labels are case-sensitive
    ],
)
def test_hand_ids_fall_back_to_positional_when_handedness_is_unusable(
    count: int, handedness: list[str]
) -> None:
    assert hand_ids_from_handedness(count, handedness) == list(range(count))


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


def test_recognizer_thumb_wobble_does_not_break_a_fist() -> None:
    # The thumb flips in/out every frame: since it is ignored, the fist stays stable.
    rec = GestureRecognizer()
    folded, out = make_geometry(()), make_geometry(("thumb",))
    last: list[Gesture] = []
    for i in range(20):
        last = [s.gesture for s in rec.update([folded if i % 2 else out])]
    assert last == [Gesture.CLOSED_FIST]


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


def test_recognizer_never_emits_two_hands_together() -> None:
    # TWO_HANDS_TOGETHER exists in the enum and hands_together() works, but the recognizer
    # does not emit it: close hands still report their own single-hand gesture.
    # Spell detectors (is_ikkon, ...) read the raw geometries, so they are unaffected.
    pair = close_pair()
    assert hands_together(pair)
    history = feed_recognizer(GestureRecognizer(), pair, 8)
    assert history[-1] == [Gesture.THUMB_MIDDLE_PINCH, Gesture.THUMB_MIDDLE_PINCH]
    assert all(Gesture.TWO_HANDS_TOGETHER not in frame for frame in history)

    fists = [fist((300.0, 240.0)), fist((340.0, 240.0))]
    history = feed_recognizer(GestureRecognizer(), fists, 8)
    assert history[-1] == [Gesture.CLOSED_FIST, Gesture.CLOSED_FIST]


# GestureRecognizer with handedness
def test_recognizer_without_handedness_uses_positional_ids() -> None:
    states = GestureRecognizer().update([palm(LEFT), palm(RIGHT)])
    assert [s.hand_id for s in states] == [0, 1]


def test_recognizer_single_right_hand_gets_id_1() -> None:
    states = GestureRecognizer().update([palm(RIGHT)], ["Right"])
    assert [s.hand_id for s in states] == [1]


def test_recognizer_ids_follow_handedness_when_list_order_swaps() -> None:
    rec = GestureRecognizer()
    for _ in range(8):
        rec.update([palm(LEFT), fist(RIGHT)], ["Left", "Right"])
    # The detector now lists the right hand first: the state must follow the hand.
    states = rec.update([fist(RIGHT), palm(LEFT)], ["Right", "Left"])
    assert [(s.hand_id, s.gesture) for s in states] == [
        (1, Gesture.CLOSED_FIST),
        (0, Gesture.OPEN_PALM),
    ]
    assert [s.frames_held for s in states] == [2, 2]  # no ramp-up restart


def test_recognizer_positional_ids_mix_hands_when_list_order_swaps() -> None:
    # Documents what handedness fixes: without it, a swapped order feeds the wrong hand.
    rec = GestureRecognizer()
    for _ in range(8):
        rec.update([palm(LEFT), fist(RIGHT)])
    states = rec.update([fist(RIGHT), palm(LEFT)])
    # Slot 0 now receives a fist but still reports the old palm (it survives a short dropout).
    assert states[0].gesture is Gesture.OPEN_PALM


def test_recognizer_keeps_remaining_hand_when_other_one_disappears() -> None:
    rec = GestureRecognizer()
    both = [palm(LEFT), palm(RIGHT)]
    feed_recognizer(rec, both, 8, ["Left", "Right"])
    # Left hand leaves: the right hand keeps id 1 and its confirmed gesture.
    states = rec.update([palm(RIGHT)], ["Right"])
    assert [(s.hand_id, s.gesture) for s in states] == [(1, Gesture.OPEN_PALM)]
    # Left hand comes back: it ramps up from scratch, the right one is untouched.
    states = rec.update(both, ["Left", "Right"])
    assert states[0].gesture is Gesture.NONE
    assert states[1].gesture is Gesture.OPEN_PALM


def test_recognizer_falls_back_to_positional_ids_on_bad_handedness() -> None:
    rec = GestureRecognizer()
    states = rec.update([palm(LEFT), palm(RIGHT)], ["Left", "Left"])  # duplicate labels
    assert [s.hand_id for s in states] == [0, 1]
    states = rec.update([palm(LEFT), palm(RIGHT)], ["Left"])  # wrong length
    assert [s.hand_id for s in states] == [0, 1]


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



# when MediaPipe loses the hand, it can still output a few collapsed
# landmarks. All fingertips then sit on top of their MCPs, so every finger is
# "curled" and the hand would be classified as CLOSED_FIST. The size gate
# rejects those detections BEFORE looking at the fingers.
GHOST_MIN_PX = 0.05 * max(IMAGE_W, IMAGE_H)  # 32 px for a 640 px wide image


def test_ghost_without_size_gate_would_be_a_fist() -> None:
    # Proves the gate is what protects us: without it the ghost IS a fist.
    gate_off = GestureConfig(min_hand_size_norm=0.0)
    assert classify_hand(ghost_fist(), gate_off) is Gesture.CLOSED_FIST
    assert classify_hand(ghost_fist()) is Gesture.NONE


@pytest.mark.parametrize("size_px", [0.0, 1.0, 5.0, 20.0, GHOST_MIN_PX - 1.0])
def test_size_gate_rejects_every_size_below_threshold(size_px: float) -> None:
    for extended in ((), ("index",), ALL_FINGERS):
        assert classify_hand(make_geometry(extended, size_px=size_px)) is Gesture.NONE


def test_size_gate_boundary_is_strict() -> None:
    # hand_size_norm == 0.05 is NOT below the threshold -> the hand is real.
    assert classify_hand(make_geometry((), size_px=GHOST_MIN_PX)) is Gesture.CLOSED_FIST
    assert classify_hand(make_geometry((), size_px=GHOST_MIN_PX + 1.0)) is Gesture.CLOSED_FIST
    assert classify_hand(make_geometry((), size_px=GHOST_MIN_PX - 1.0)) is Gesture.NONE


def test_ghost_hand_never_reaches_a_confirmed_fist_through_the_recognizer() -> None:
    rec = GestureRecognizer()
    history = feed_recognizer(rec, [ghost_fist()], 200)
    assert all(frame == [Gesture.NONE] for frame in history)


def test_ghost_hands_never_trigger_a_shield() -> None:
    det = SpellDetector(is_shield)
    for _ in range(200):
        assert det.update([ghost_fist(LEFT), ghost_fist(RIGHT)]).state is PoseState.IDLE


def test_ghost_hand_between_real_frames_does_not_create_a_fist() -> None:
    # A real palm, then the detector hallucinates a ghost: no fist ever appears.
    rec = GestureRecognizer()
    feed_recognizer(rec, [palm()], 8)
    seen: set[Gesture] = set()
    for _ in range(50):
        seen.update(s.gesture for s in rec.update([ghost_fist()]))
    assert Gesture.CLOSED_FIST not in seen




#a pose only becomes ACTIVE after enter_frames (3) + confirm_frames (5) = 8
# CONSECUTIVE frames. Any miss before that resets the streak, so a signal that
# flips every frame (or every few frames) never gets close to 8 in a row.
@pytest.mark.parametrize(("enter", "confirm"), [(1, 1), (2, 4), (3, 5), (4, 8)])
def test_confirmation_needs_exactly_enter_plus_confirm_consecutive_frames(
    enter: int, confirm: int
) -> None:
    stab = PoseStabilizer(StabilityConfig(enter_frames=enter, confirm_frames=confirm))
    states = [stab.update(True).state for _ in range(enter + confirm)]
    assert all(s is not PoseState.ACTIVE for s in states[:-1])
    assert states[-1] is PoseState.ACTIVE


def test_default_confirmation_is_8_frames() -> None:
    cfg = StabilityConfig()
    assert cfg.enter_frames + cfg.confirm_frames == 8


def test_alternating_signal_never_activates_over_200_frames() -> None:
    # Both phases of the alternation, so neither "on-first" nor "off-first" slips through.
    for offset in (0, 1):
        stab = PoseStabilizer()
        for i in range(200):
            assert stab.update((i + offset) % 2 == 0).state is not PoseState.ACTIVE


def test_alternating_hand_present_absent_never_confirms_a_gesture() -> None:
    rec = GestureRecognizer()
    for i in range(200):
        frame = [palm()] if i % 2 == 0 else []
        assert all(s.gesture is Gesture.NONE for s in rec.update(frame))


def test_alternating_palm_fist_never_confirms_any_gesture() -> None:
    # The classifier flips between two gestures every frame (e.g. jittery landmarks).
    rec = GestureRecognizer()
    for i in range(200):
        frame = [palm() if i % 2 == 0 else fist()]
        assert all(s.gesture is Gesture.NONE for s in rec.update(frame))


def test_seven_good_frames_then_a_miss_never_confirms_through_the_recognizer() -> None:
    rec = GestureRecognizer()
    for _ in range(30):
        for _ in range(7):
            assert rec.update([palm()])[0].gesture is Gesture.NONE
        rec.update([fist()])  # one bad frame resets the 7-frame streak


def test_alternating_shield_never_activates_over_200_frames() -> None:
    det = SpellDetector(is_shield)
    shield = far_pair(palm(LEFT), palm(RIGHT))
    for i in range(200):
        assert det.update(shield if i % 2 == 0 else []).state is not PoseState.ACTIVE


def test_alternating_ikkon_never_activates_over_200_frames() -> None:
    det = SpellDetector(is_ikkon)
    pose = far_pair(pinch_hand(LEFT), pinch_hand(RIGHT))
    for i in range(200):
        assert det.update(pose if i % 2 == 0 else []).state is not PoseState.ACTIVE


def test_confirmation_still_works_after_a_flicker_burst() -> None:
    # Flicker must not "poison" the detector: a steady pose afterwards activates in 8 frames.
    det = SpellDetector(is_shield)
    shield = far_pair(palm(LEFT), palm(RIGHT))
    for i in range(200):
        det.update(shield if i % 2 == 0 else [])
    assert run_spell(det, shield, 8) is PoseState.ACTIVE