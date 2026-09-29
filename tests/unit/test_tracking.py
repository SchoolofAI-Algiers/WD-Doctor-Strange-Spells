"""Unit and property tests for HandTracker.

Everything here is fast and needs no camera and no model file: MediaPipe's
landmarker is replaced by a mock, and its results by small fake objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import mediapipe as mp
import numpy as np
import pytest
from mediapipe.tasks.python import vision
from src.tracking.hand_tracker import HandLandmarks, HandTracker

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# Fakes and helpers
# --------------------------------------------------------------------------
@dataclass
class FakeLandmark:
    x: float
    y: float
    z: float


@dataclass
class FakeCategory:
    category_name: str
    score: float


@dataclass
class FakeResult:
    hand_landmarks: list[list[FakeLandmark]]
    handedness: list[list[FakeCategory]]


def make_hand(
    x: float = 0.5,
    y: float = 0.5,
    z: float = 0.0,
) -> list[FakeLandmark]:
    """A fake hand with 21 identical landmarks."""
    return [FakeLandmark(x, y, z) for _ in range(21)]


def make_result(hands, labels=None) -> FakeResult:
    """Build a fake MediaPipe result.

    ``labels`` is a list of ``(name, score)`` pairs, one per hand.
    """
    if labels is None:
        labels = [("Right", 0.9)] * len(hands)
    return FakeResult(
        hand_landmarks=hands,
        handedness=[[FakeCategory(name, score)] for name, score in labels],
    )


def make_tracker() -> HandTracker:
    """HandTracker with a mocked landmarker: no model file needed."""
    tracker = HandTracker.__new__(HandTracker)
    tracker._landmarker = MagicMock()
    tracker._timestamp_ms = 0
    return tracker


# --------------------------------------------------------------------------
# _convert: coordinate conversion
# --------------------------------------------------------------------------
def test_convert_center_point_to_pixels():
    hands = make_tracker()._convert(
        make_result([make_hand(0.5, 0.5)]),
        640,
        480,
    )

    assert hands[0].landmarks_px[0][0] == pytest.approx(320.0)
    assert hands[0].landmarks_px[0][1] == pytest.approx(240.0)


def test_convert_uses_width_for_x_and_height_for_y():
    """Non-square frame: catches swapped width/height bugs."""
    hands = make_tracker()._convert(
        make_result([make_hand(0.25, 0.5)]),
        640,
        480,
    )

    assert hands[0].landmarks_px[0][0] == pytest.approx(160.0)
    assert hands[0].landmarks_px[0][1] == pytest.approx(240.0)


def test_convert_z_is_passed_through_unchanged():
    hands = make_tracker()._convert(
        make_result([make_hand(0.5, 0.5, z=-0.07)]),
        640,
        480,
    )

    assert hands[0].landmarks_px[0][2] == pytest.approx(-0.07)
    assert hands[0].landmarks_norm[0][2] == pytest.approx(-0.07)


def test_convert_landmarks_norm_stays_normalized():
    hands = make_tracker()._convert(
        make_result([make_hand(0.25, 0.75)]),
        640,
        480,
    )

    assert hands[0].landmarks_norm[0][0] == pytest.approx(0.25)
    assert hands[0].landmarks_norm[0][1] == pytest.approx(0.75)


def test_convert_preserves_landmark_order():
    """Index i must be MediaPipe's landmark i (0=wrist ... 20=pinky tip)."""
    hand = [FakeLandmark(x=i / 100, y=0.0, z=0.0) for i in range(21)]
    hands = make_tracker()._convert(make_result([hand]), 200, 100)

    for i in range(21):
        assert hands[0].landmarks_px[i][0] == pytest.approx(i / 100 * 200)


def test_convert_output_shapes_and_dtypes():
    hands = make_tracker()._convert(make_result([make_hand()]), 640, 480)

    assert isinstance(hands[0], HandLandmarks)
    assert hands[0].landmarks_px.shape == (21, 3)
    assert hands[0].landmarks_norm.shape == (21, 3)
    assert hands[0].landmarks_px.dtype == np.float32
    assert hands[0].landmarks_norm.dtype == np.float32


def test_convert_out_of_frame_values_are_not_clipped():
    """Documents current behavior: a hand partly off-screen gives values
    outside 0..1 and they are passed through as-is (not clipped)."""
    hands = make_tracker()._convert(
        make_result([make_hand(-0.1, 1.2)]), 100, 100
    )

    assert hands[0].landmarks_px[0][0] == pytest.approx(-10.0)
    assert hands[0].landmarks_px[0][1] == pytest.approx(120.0)


# --------------------------------------------------------------------------
# _convert: number of hands
# --------------------------------------------------------------------------
def test_convert_no_hands_returns_empty_list_not_none():
    hands = make_tracker()._convert(make_result([]), 640, 480)

    assert hands == []
    assert isinstance(hands, list)


def test_convert_two_hands_keeps_order_and_labels():
    result = make_result(
        [make_hand(0.25, 0.25), make_hand(0.75, 0.75)],
        labels=[("Left", 0.9), ("Right", 0.88)],
    )
    hands = make_tracker()._convert(result, 640, 480)

    assert [h.handedness for h in hands] == ["Left", "Right"]
    assert hands[0].landmarks_px[0][0] == pytest.approx(160.0)
    assert hands[1].landmarks_px[0][0] == pytest.approx(480.0)


@pytest.mark.parametrize("n_hands", [0, 1, 2])
def test_convert_returns_one_entry_per_detected_hand(n_hands):
    result = make_result([make_hand() for _ in range(n_hands)])

    assert len(make_tracker()._convert(result, 640, 480)) == n_hands


# --------------------------------------------------------------------------
# _convert: handedness parsing
# --------------------------------------------------------------------------
def test_convert_handedness_label_and_score_are_passed_through():
    result = make_result([make_hand()], labels=[("Left", 0.8123)])
    hand = make_tracker()._convert(result, 640, 480)[0]

    assert hand.handedness == "Left"
    assert hand.score == pytest.approx(0.8123)


def test_convert_missing_handedness_falls_back_to_unknown():
    result = FakeResult(hand_landmarks=[make_hand()], handedness=[])
    hand = make_tracker()._convert(result, 640, 480)[0]

    assert hand.handedness == "Unknown"
    assert hand.score == 0.0


def test_convert_partial_handedness_only_second_hand_is_unknown():
    result = FakeResult(
        hand_landmarks=[make_hand(), make_hand()],
        handedness=[[FakeCategory("Right", 0.9)]],  # one label for two hands
    )
    hands = make_tracker()._convert(result, 640, 480)

    assert hands[0].handedness == "Right"
    assert hands[1].handedness == "Unknown"


# --------------------------------------------------------------------------
# Property-style tests (seeded random data, no extra dependency)
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "size",
    [(320, 240), (640, 480), (1280, 720), (1920, 1080), (100, 100)],
)
@pytest.mark.parametrize("seed", range(5))
def test_property_in_frame_landmarks_stay_within_pixel_bounds(size, seed):
    width, height = size
    rng = np.random.default_rng(seed)
    hand = [FakeLandmark(*rng.uniform(0, 1, 3)) for _ in range(21)]

    px = make_tracker()._convert(
        make_result([hand]),
        width,
        height,
    )[0].landmarks_px

    assert px.shape == (21, 3)
    assert np.all(px[:, 0] >= 0) and np.all(px[:, 0] <= width)
    assert np.all(px[:, 1] >= 0) and np.all(px[:, 1] <= height)


@pytest.mark.parametrize("seed", range(5))
def test_property_pixels_are_consistent_with_normalized(seed):
    """px_x / width == norm_x and px_y / height == norm_y."""
    width, height = 640, 480
    rng = np.random.default_rng(seed)
    hand = [FakeLandmark(*rng.uniform(0, 1, 3)) for _ in range(21)]

    out = make_tracker()._convert(make_result([hand]), width, height)[0]

    np.testing.assert_allclose(
        out.landmarks_px[:, 0] / width,
        out.landmarks_norm[:, 0],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        out.landmarks_px[:, 1] / height,
        out.landmarks_norm[:, 1],
        rtol=1e-5,
    )
    np.testing.assert_array_equal(
        out.landmarks_px[:, 2], out.landmarks_norm[:, 2]
    )


@pytest.mark.parametrize("n_hands", [1, 2])
def test_property_every_hand_has_exactly_21_landmarks(n_hands):
    result = make_result([make_hand() for _ in range(n_hands)])

    for hand in make_tracker()._convert(result, 640, 480):
        assert hand.landmarks_px.shape[0] == 21
        assert hand.landmarks_norm.shape[0] == 21


# --------------------------------------------------------------------------
# process(): the per-frame pipeline, with a mocked landmarker
# --------------------------------------------------------------------------
def test_process_returns_empty_list_when_nothing_detected():
    tracker = make_tracker()
    tracker._landmarker.detect_for_video.return_value = make_result([])

    hands = tracker.process(np.zeros((480, 640, 3), dtype=np.uint8))

    assert hands == []


def test_process_converts_bgr_to_rgb_before_detection():
    tracker = make_tracker()
    tracker._landmarker.detect_for_video.return_value = make_result([])
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[0, 0] = (10, 20, 30)  # OpenCV order: B=10, G=20, R=30

    tracker.process(frame)

    mp_image = tracker._landmarker.detect_for_video.call_args[0][0]
    assert isinstance(mp_image, mp.Image)
    assert list(mp_image.numpy_view()[0, 0]) == [30, 20, 10]  # RGB order


def test_process_uses_frame_dimensions_for_pixel_conversion():
    tracker = make_tracker()
    tracker._landmarker.detect_for_video.return_value = make_result(
        [make_hand(0.5, 0.5)]
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)  # height=480, width=640

    hands = tracker.process(frame)

    assert hands[0].landmarks_px[0][0] == pytest.approx(320.0)
    assert hands[0].landmarks_px[0][1] == pytest.approx(240.0)


def test_process_timestamps_strictly_increase():
    """VIDEO mode requires increasing timestamps.

    Otherwise, MediaPipe raises an error.
    """
    tracker = make_tracker()
    tracker._landmarker.detect_for_video.return_value = make_result([])
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    for _ in range(5):
        tracker.process(frame)

    timestamps = [
        c[0][1]
        for c in tracker._landmarker.detect_for_video.call_args_list
    ]
    assert timestamps == sorted(set(timestamps))
    assert len(timestamps) == 5


def test_process_does_not_modify_the_input_frame():
    tracker = make_tracker()
    tracker._landmarker.detect_for_video.return_value = make_result([])
    frame = np.random.default_rng(0).integers(
        0, 255, (20, 20, 3), dtype=np.uint8
    )
    original = frame.copy()

    tracker.process(frame)

    np.testing.assert_array_equal(frame, original)


# --------------------------------------------------------------------------
# __init__: configuration
# --------------------------------------------------------------------------
def test_init_passes_configuration_to_mediapipe():
    with patch.object(vision.HandLandmarker, "create_from_options") as create:
        HandTracker(
            model_path="some_model.task",
            max_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.4,
        )

    options = create.call_args[0][0]
    assert options.num_hands == 1
    assert options.min_hand_detection_confidence == pytest.approx(0.6)
    assert options.min_tracking_confidence == pytest.approx(0.4)
    assert options.running_mode == vision.RunningMode.VIDEO
    assert options.base_options.model_asset_path == "some_model.task"


def test_init_defaults_match_the_spec():
    with patch.object(vision.HandLandmarker, "create_from_options") as create:
        HandTracker()

    options = create.call_args[0][0]
    assert options.num_hands == 2
    assert options.min_hand_detection_confidence == pytest.approx(0.7)
    assert options.min_tracking_confidence == pytest.approx(0.5)


def test_init_with_missing_model_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        HandTracker(model_path="this_model_does_not_exist.task")


# --------------------------------------------------------------------------
# Lifecycle: close() and the `with` statement
# --------------------------------------------------------------------------
def test_close_releases_the_landmarker():
    tracker = make_tracker()

    tracker.close()

    tracker._landmarker.close.assert_called_once()


def test_context_manager_returns_self_and_closes_on_exit():
    tracker = make_tracker()

    with tracker as entered:
        assert entered is tracker
        tracker._landmarker.close.assert_not_called()

    tracker._landmarker.close.assert_called_once()


def test_context_manager_closes_even_if_an_error_happens_inside():
    tracker = make_tracker()

    with pytest.raises(RuntimeError), tracker:
        raise RuntimeError("boom")

    tracker._landmarker.close.assert_called_once()
