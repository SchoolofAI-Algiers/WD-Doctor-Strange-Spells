"""Unit tests for the Capture subsystem. They use a fake camera, so no webcam is needed."""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from src.capture.capture import Capture, list_available_cameras

pytestmark = pytest.mark.unit


@pytest.fixture
def fake_cap() -> Iterator[MagicMock]:
    """Replace cv2.VideoCapture with a fake camera that returns a black 640x480 frame."""
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    with patch("src.capture.capture.cv2.VideoCapture", return_value=cap):
        yield cap


def test_reconnect_failure_is_swallowed(fake_cap: MagicMock) -> None:
    cam = Capture()
    fake_cap.isOpened.return_value = False  # the camera is now gone

    cam._reconnect()  # must not raise

    cam.release()


def test_get_fps_is_zero_when_timestamps_are_identical(fake_cap: MagicMock) -> None:
    with Capture() as cam:
        cam._times.extend([5.0, 5.0, 5.0])  # elapsed = 0, so no division by zero

        assert cam.get_fps() == 0.0


def test_read_converts_bgra_to_bgr(fake_cap: MagicMock) -> None:
    fake_cap.read.return_value = (True, np.zeros((480, 640, 4), dtype=np.uint8))

    with Capture() as cam:
        frame = cam.read()

    assert frame.shape == (480, 640, 3)


def test_read_converts_float_frame_to_uint8(fake_cap: MagicMock) -> None:
    fake_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.float32))

    with Capture() as cam:
        frame = cam.read()

    assert frame.dtype == np.uint8


def test_read_returns_bgr_uint8_frame(fake_cap: MagicMock) -> None:
    with Capture() as cam:
        frame = cam.read()

    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8


def test_read_converts_grayscale_to_bgr(fake_cap: MagicMock) -> None:
    fake_cap.read.return_value = (True, np.zeros((480, 640), dtype=np.uint8))

    with Capture() as cam:
        frame = cam.read()

    assert frame.shape == (480, 640, 3)


def test_open_failure_raises(fake_cap: MagicMock) -> None:
    fake_cap.isOpened.return_value = False

    with pytest.raises(RuntimeError):
        Capture()


def test_read_after_release_is_refused(fake_cap: MagicMock) -> None:
    cam = Capture()
    cam.release()

    with pytest.raises(RuntimeError, match="released"):
        cam.read()


def test_context_manager_releases_camera(fake_cap: MagicMock) -> None:
    with Capture() as cam:
        pass

    fake_cap.release.assert_called()
    assert cam.cap is None


def test_read_times_out_when_camera_never_returns_a_frame(fake_cap: MagicMock) -> None:
    fake_cap.read.return_value = (False, None)

    with Capture() as cam, pytest.raises(TimeoutError):
        cam.read(timeout=0.3)


def test_read_reconnects_after_a_failed_read(fake_cap: MagicMock) -> None:
    good_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    fake_cap.read.side_effect = [(False, None), (True, good_frame)]  # fail once, then work

    with Capture() as cam:
        frame = cam.read()

    assert frame.shape == (480, 640, 3)


def test_get_fps_is_zero_before_two_frames(fake_cap: MagicMock) -> None:
    with Capture() as cam:
        assert cam.get_fps() == 0.0
        cam.read()
        assert cam.get_fps() == 0.0


def test_get_fps_uses_the_rolling_window(fake_cap: MagicMock) -> None:
    with Capture() as cam:
        cam._times.extend([10.0, 10.1, 10.2])  # 3 timestamps = 2 gaps over 0.2 s

        assert cam.get_fps() == pytest.approx(10.0)


def test_list_available_cameras_returns_working_indices() -> None:
    def make_cap(index: int) -> MagicMock:
        cap = MagicMock()
        cap.isOpened.return_value = index in (0, 2)  # only cameras 0 and 2 "exist"
        return cap

    with patch("src.capture.capture.cv2.VideoCapture", side_effect=make_cap):
        assert list_available_cameras(max_index=4) == [0, 2]
