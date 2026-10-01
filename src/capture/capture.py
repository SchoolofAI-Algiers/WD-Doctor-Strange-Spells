import contextlib
import time
from collections import deque
from types import TracebackType
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


def list_available_cameras(
    max_index: int = 5,
) -> list[int]:  # 5 is random chosen as max of cameras to ckeck
    """Return the camera indices that actually open."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(
            i
        )  # opencv identifies cameras by their index, starting from 0. The first camera is usually index 0, the second camera is index 1, and so on. The maximum index to check is specified by max_index.
        if cap.isOpened():
            found.append(i)
        cap.release()
    return found


class Capture:
    """Webcam acquisition: BGR uint8 frames, FPS measurement, timeout and reconnect."""

    def __init__(
        self,
        device_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        buffer_size: int = 1,
    ) -> None:
        """Store the camera settings and open the camera. Raises RuntimeError if it can't open."""
        # Initialize the video capture object
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size  # buffer size is the number of frames that can be stored in the camera's internal buffer before they are processed. A larger buffer size can help prevent dropped frames, but it can also introduce latency. A smaller buffer size can reduce latency, but it may also increase the risk of dropped frames if the processing cannot keep up with the frame rate.

        self._times: deque[float] = deque(maxlen=30)  # max 30 latest timestamps
        self._closed = False  # becomes True after release()
        self.cap: cv2.VideoCapture | None = None  # VideoCapture object
        self._open()  # opens the camera and applies the settings, raises RuntimeError if it can't open

    def _open(
        self,
    ) -> None:  # this needs to be a separate function because we need to call it from __init__ and _reconnect so its cant be in __init__ directly.
        """Open the camera and request width, height, fps and buffer size."""
        # open is called when we create an object+ when reconnecting cam as it has default settings of the cam
        cap = cv2.VideoCapture(
            self.device_id
        )  # captures video from device_id (0 = default camera) .
        self.cap = cap

        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"Could not open your device's camera {self.device_id}")

        # request the camera to use specified  width, height,fps and buffer size.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

    def _reconnect(
        self,
    ) -> None:  # drops the old handle (we affected none) and tries to open the camera again with the same settings
        """Release the old handle and try to reopen the camera. Never raises."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None  # we will call open that creates it again

            with contextlib.suppress(
                RuntimeError
            ):  # if error of this type happens, we pass and keep executing code
                self._open()  # camera still missing, read() will keep trying until its timeout runs out

    def read(
        self, timeout: float = 2.0
    ) -> NDArray[
        np.uint8
    ]:  # reads a frame from the camera, raises TimeoutError if no frame arrives within timeout seconds
        """Return one BGR uint8 frame. Raises TimeoutError if none arrives within timeout seconds."""
        if self._closed:  # when its realised we set it to true , we cant rfead if not camera
            raise RuntimeError("Capture has been released")

        # timout is the max limit of waiting for a frame to arive
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.cap is not None:
                ok, frame = (
                    self.cap.read()
                )  # read returns a boolean (ok), True if it was successful, False if it failed, and the frame itself (frame)

                if ok and frame is not None:
                    self._times.append(
                        time.monotonic()
                    )  # append time frame was captured to the deque
                    return self._normalize(frame)

            self._reconnect()  # read failed or camera is missing, try to recover
            time.sleep(0.1)  # it gives time to usb to be back and camera to be reconnected.

        raise TimeoutError(f"No frame from camera {self.device_id} within {timeout}s")

    def get_fps(
        self,
    ) -> float:  # result is a float representing the frames per second (FPS) of the video capture
        """Return the rolling-average FPS over the last 30 frames (0.0 if it can't be measured yet)."""

        if len(self._times) < 2:  # if we don't have at least 2 timestamps, we can't calculate FPS
            return 0.0
        elapsed = (
            self._times[-1] - self._times[0]
        )  # calculate the elapsed time between the first and last timestamps in the deque [0] is first and [-1] is last, so we subtract the first timestamp from the last timestamp to get the total time elapsed during the capture of frames
        if elapsed <= 0.0:
            return 0.0  # to avoid division by zero or negative elabsed time

        return (
            (len(self._times) - 1) / elapsed
        )  # number of frames / elapsed time = frames per second (FPS), -1 as 5 frames for ex gives us 4 gaps between them

    def release(self) -> None:
        """Release the camera for good. Later read() calls are refused."""
        self._closed = True  # setting closed to true (camera closed)
        if self.cap is not None:
            self.cap.release()  # if we dont release the camera, it will remain locked and unavailable for other applications or future runs of the program. This can lead to errors or unexpected behavior when trying to access the camera again.
            self.cap = None

    def _normalize(
        self, frame: NDArray[Any]
    ) -> NDArray[
        np.uint8
    ]:  # we guarantees BGR, 3 channels, uint8 , there are 3 possible  cases for the frame: grayscale, BGRA, and non-uint8.
        """Convert a frame to BGR, 3 channels, uint8."""

        if frame.ndim == 2:  # grayscale -> BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:  # 4 channels                          # BGRA -> BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if frame.dtype != np.uint8:  # force uint8
            frame = frame.astype(np.uint8)
        return frame

    def __enter__(self) -> "Capture":  # runs when the `with` block starts always
        """Return the Capture object for use in a `with` block."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:  # runs when the `with` block ends, even after an error
        """Release the camera when the `with` block ends, even after an error."""
        self.release()
