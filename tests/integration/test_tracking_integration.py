"""Integration tests for HandTracker: real MediaPipe model, real frames.

Requirements (tests skip automatically if these are missing):
  - hand_landmarker.task in the project root
  - for the detection tests: tests/fixtures/hand.jpg, a clear photo of an
    open hand (well lit, plain background) that you took yourself
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from src.tracking.hand_tracker import HandTracker

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "hand_landmarker.task"
HAND_IMAGE = PROJECT_ROOT / "tests" / "fixtures" / "hand.jpg"

requires_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=(
        "hand_landmarker.task not found in project root "
        "(see DEVELOPMENT.md)"
    ),
)
requires_hand_image = pytest.mark.skipif(
    not HAND_IMAGE.exists(),
    reason="tests/fixtures/hand.jpg not found (add a photo of an open hand)",
)


@pytest.fixture
def tracker():
    with HandTracker(model_path=str(MODEL_PATH)) as t:
        yield t


@requires_model
def test_blank_frame_returns_empty_list(tracker):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    assert tracker.process(frame) == []


@requires_model
def test_random_noise_frame_does_not_crash(tracker):
    frame = np.random.default_rng(0).integers(
        0, 255, (480, 640, 3), dtype=np.uint8
    )

    hands = tracker.process(frame)

    assert isinstance(hands, list)
    for hand in hands:
        assert hand.landmarks_px.shape == (21, 3)


@requires_model
@pytest.mark.parametrize("size", [(320, 240), (640, 480), (1280, 720)])
def test_works_at_different_resolutions(tracker, size):
    width, height = size
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    assert isinstance(tracker.process(frame), list)


@requires_model
def test_many_consecutive_frames_stay_stable(tracker):
    """VIDEO mode must accept a long stream of frames without errors."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    for _ in range(60):
        assert tracker.process(frame) == []


@requires_model
def test_can_create_and_close_repeatedly():
    for _ in range(3):
        with HandTracker(model_path=str(MODEL_PATH)) as t:
            t.process(np.zeros((120, 160, 3), dtype=np.uint8))


@requires_model
@requires_hand_image
def test_detects_a_hand_in_the_sample_image(tracker):
    frame = cv2.imread(str(HAND_IMAGE))
    assert frame is not None, "hand.jpg exists but OpenCV could not read it"

    hands = tracker.process(frame)

    assert len(hands) >= 1


@requires_model
@requires_hand_image
def test_sample_image_landmarks_have_the_agreed_format(tracker):
    frame = cv2.imread(str(HAND_IMAGE))
    height, width = frame.shape[:2]

    hand = tracker.process(frame)[0]

    assert hand.landmarks_px.shape == (21, 3)
    assert hand.landmarks_norm.shape == (21, 3)
    assert hand.handedness in {"Left", "Right"}
    assert 0.5 <= hand.score <= 1.0
    # a clearly visible, fully in-frame hand: every landmark inside the image
    assert np.all(hand.landmarks_px[:, 0] >= 0) and np.all(
        hand.landmarks_px[:, 0] <= width
    )
    assert np.all(hand.landmarks_px[:, 1] >= 0) and np.all(
        hand.landmarks_px[:, 1] <= height
    )


@requires_model
@requires_hand_image
def test_sample_image_geometry_is_plausible(tracker):
    """Anatomy sanity check: fingertips are farther from the wrist than the
    knuckle at the base of the middle finger, for an open hand."""
    frame = cv2.imread(str(HAND_IMAGE))
    px = tracker.process(frame)[0].landmarks_px

    wrist = px[0, :2]
    middle_base = px[9, :2]
    middle_tip = px[12, :2]

    assert np.linalg.norm(middle_tip - wrist) > np.linalg.norm(
        middle_base - wrist
    )


@requires_model
@requires_hand_image
def test_pixel_and_normalized_outputs_agree_on_real_data(tracker):
    frame = cv2.imread(str(HAND_IMAGE))
    height, width = frame.shape[:2]

    hand = tracker.process(frame)[0]

    np.testing.assert_allclose(
        hand.landmarks_px[:, 0], hand.landmarks_norm[:, 0] * width, rtol=1e-4
    )
    np.testing.assert_allclose(
        hand.landmarks_px[:, 1], hand.landmarks_norm[:, 1] * height, rtol=1e-4
    )
