"""Unit tests for HandTracker's coordinate conversion — mocked, no camera/model needed."""

from dataclasses import dataclass
from typing import List

import numpy as np
import pytest

from src.tracking.hand_tracker import HandTracker


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
    hand_landmarks: List[List[FakeLandmark]]
    handedness: List[List[FakeCategory]]


def make_tracker_without_model() -> HandTracker:
    """Build a HandTracker instance without running __init__'s model load,
    since _convert() doesn't touch the model at all."""
    return HandTracker.__new__(HandTracker)


def test_convert_single_hand_pixel_coordinates():
    tracker = make_tracker_without_model()

    fake_landmarks = [FakeLandmark(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    fake_result = FakeResult(
        hand_landmarks=[fake_landmarks],
        handedness=[[FakeCategory(category_name="Right", score=0.95)]],
    )

    hands = tracker._convert(fake_result, frame_width=640, frame_height=480)

    assert len(hands) == 1
    hand = hands[0]

    assert hand.landmarks_px.shape == (21, 3)
    assert hand.landmarks_norm.shape == (21, 3)

    assert hand.landmarks_px[0][0] == pytest.approx(320.0)
    assert hand.landmarks_px[0][1] == pytest.approx(240.0)
    assert hand.landmarks_px[0][2] == pytest.approx(0.0)

    assert hand.handedness == "Right"
    assert hand.score == pytest.approx(0.95)


def test_convert_no_hands_returns_empty_list():
    tracker = make_tracker_without_model()
    fake_result = FakeResult(hand_landmarks=[], handedness=[])

    hands = tracker._convert(fake_result, frame_width=640, frame_height=480)

    assert hands == []
    assert hands is not None


def test_convert_two_hands():
    tracker = make_tracker_without_model()

    hand1 = [FakeLandmark(x=0.25, y=0.25, z=0.0) for _ in range(21)]
    hand2 = [FakeLandmark(x=0.75, y=0.75, z=0.0) for _ in range(21)]
    fake_result = FakeResult(
        hand_landmarks=[hand1, hand2],
        handedness=[
            [FakeCategory(category_name="Left", score=0.9)],
            [FakeCategory(category_name="Right", score=0.88)],
        ],
    )

    hands = tracker._convert(fake_result, frame_width=640, frame_height=480)

    assert len(hands) == 2
    assert hands[0].handedness == "Left"
    assert hands[1].handedness == "Right"
    assert hands[0].landmarks_px[0][0] == pytest.approx(160.0)
    assert hands[1].landmarks_px[0][0] == pytest.approx(480.0)


def test_landmark_count_is_always_21():
    tracker = make_tracker_without_model()
    fake_landmarks = [FakeLandmark(x=0.1, y=0.1, z=0.0) for _ in range(21)]
    fake_result = FakeResult(
        hand_landmarks=[fake_landmarks],
        handedness=[[FakeCategory(category_name="Left", score=0.8)]],
    )

    hands = tracker._convert(fake_result, frame_width=100, frame_height=100)

    assert hands[0].landmarks_px.shape[0] == 21
    assert hands[0].landmarks_norm.shape[0] == 21