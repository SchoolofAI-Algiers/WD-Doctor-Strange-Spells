from __future__ import annotations

from collections import deque
from enum import Enum

# PART 1 by Djo - Trajectory, Stationary, Depth 

Point = tuple[float, float]

class TrajectoryBuffer:  # remembers the last N positions of the hand.
    def __init__(self, maxlen: int = 60) -> None:
        if maxlen <1:
            raise ValueError("maxlen must be a >=1")
        # deque with maxlen: when full, adding a new item drops the oldest one
        self._pts: deque[tuple[float, float, float]] = deque(maxlen=maxlen)

    def append(self, x: float, y: float, t: float) -> None:
        # Store one position with its timestamp (seconds), position of hand in t
        self._pts.append((x, y, t))

    def points(self) -> list[tuple[float, float]]:
        # Return only the (x, y) pairs, oldest first, x,y of deque in _pts
        return [(x, y) for x, y, _ in self._pts]

    def last(self)-> Point | None:
        if not self._pts:
            return None
        x,y,_ = self._pts[-1]
        return (x,y)
    
    def clear(self) -> None:
        self._pts.clear()  # clears the deque of all points stored in _pts

    def __len__(self) -> int:
        return len(self._pts)  # Return the number of points stored in the deque _pts


class StationaryDetector:  # detects when a hand is not moving.
    # Detects when a hand is not moving , based on its speed and size.

    def __init__(
        self, enter_thresh: float = 0.10, exit_thresh: float = 0.25, min_frames: int = 5
    ) -> (
        None
    ):  
        if not 0 < enter_thresh < exit_thresh:
            raise ValueError("enter_thresh must be >0 and < exit_thresh")
        if min_frames < 1:
            raise ValueError("min_frames must be >=1")
        
        # threshold are by hand size perr sec, 0.1=20px per s and 0.25 is 50 px per sec
        # A Hand is considered stationary if its speed is below enter_thresh for at least min_frames consecutive frames. It is considered moving again if its speed exceeds exit_thresh.
        # when the hand is between enter and exit threshold, we don't change the state, we wait for the next frame to decide if it is still or not

        self.enter_thresh = enter_thresh
        self.exit_thresh = exit_thresh
        self.min_frames = min_frames
        self.stationary = False
        self._count = 0  # consecutive slow frames seen so far

    def update(
        self, speed_px_s: float, hand_size_px: float
    ) -> bool:  # speed_px_s: how fast the hand is moving rn, is it still or not , hand_size_px: how big the hand looks in the image

        if hand_size_px <= 0:  # bad measurement: keep the previous answer
            return self.stationary
        s = (
            speed_px_s / hand_size_px
        )  # hand-sizes per second, this represents how fast the hand moves per sec

        if self.stationary:
            if (
                s > self.exit_thresh
            ):  # If the hand is moving fast enough, it's no longer stationary
                self.stationary = False
                self._count = 0
        elif (
            s < self.enter_thresh
        ):  # If the hand is moving slowly enough, increment the count of consecutive slow frames
            self._count += 1  # it has to stay below the enter threshold for 5 frames to be considered still
            if self._count >= self.min_frames:
                self.stationary = True
        else:
            self._count = 0  # we reset the counting, a still hand have to be still for 5 frames under the enter treshold
        return self.stationary

    def reset(self) -> None:
        self.stationary = False
        self._count = 0


class DepthTrend(Enum):
    NONE = 0
    TOWARD = 1
    AWAY = 2


class DepthTrendDetector:
    """Is the hand coming closer to the camera or going away?

    A hand that gets bigger in the image is coming closer. We smooth the size
    (EMA) and compare it with the smoothed size `window` frames ago. A change of
    a few percent in ONE frame is just noise; 8 percent over 5 frames is real.
    """

    def __init__(
        self, alpha: float = 0.4, window: int = 5, ratio_thresh: float = 1.02
    ) -> None:  # window: how many frames to compare with the current one
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        if window < 1:
            raise ValueError("window must be >=1")
        if ratio_thresh <=1.0:
            raise ValueError("ratio_thresh must be >1")
        
        self.alpha = alpha  # alpha: smoothing factor
        self.ratio_thresh = ratio_thresh  # ratio threshol: how big change is , it has to be above 8% to be considered moving toward or away
        self._need = window + 1  # values needed before we can compare
        self._hist: deque[float] = deque(
            maxlen=window + 1
        )  # we compare the current one with previous 5 (we need 6 window + 1)
        self._smooth: float | None = None

    def update(
        self, size: float
    ) -> DepthTrend:  # we give it obj and measure the size of the hand with px
        if size <= 0:
            return (
                DepthTrend.NONE
            )  # we don't want to consider bad measurements, we return NONE
        if self._smooth is None:
            self._smooth = size  # first frame, we take size as it is
        else:
            self._smooth = (
                self.alpha * size + (1 - self.alpha) * self._smooth
            )  # EMA Smoothing
        self._hist.append(
            self._smooth
        )  # we append in deque the smnoothest size of hand

        if len(self._hist) < self._need:  # not enough history yet
            return DepthTrend.NONE

        ratio = (
            self._hist[-1] / self._hist[0]
        )  # ratio is the current smothest size / frame 5th ago smoothed size
        if ratio > self.ratio_thresh:
            return DepthTrend.TOWARD
        if ratio < 1.0 / self.ratio_thresh:
            return DepthTrend.AWAY
        return DepthTrend.NONE

    def reset(self) -> None:
        self._hist.clear()
        self._smooth = None


class FingertipTrail:
    """records the path your index fingertip draws while you're pointing,
    and it decides when that path should be wiped and started again.

    we are going to save the position of tip (landmark number 8) in each frame
    """

    def __init__(
        self, maxlen: int = 60, grace_frames: int = 3
    ) -> None:  # the storage: last 60 (x, y, t) points
        self.buffer = TrajectoryBuffer(maxlen)
        self.grace_frames = grace_frames  # so if we point less than 3 frames, its npt considered new drawing, we don't clear the buffer, we wait for 3 frames to see if the hand is still pointing or not
        self._idle = 0  # frames since the last pointing frame, which means it strarts counting when the hand is not pointing, if it is pointing we reset the counter to 0

    def update(
        self, tip: tuple[float, float] | None, t: float, is_pointing: bool
    ) -> list[
        tuple[float, float]
    ]:  # tip is the position of index finger tip, t is the time in seconds, is_pointing is a boolean that tells us if the hand is pointing or not
        if is_pointing and tip is not None:
            if self._idle > self.grace_frames:
                self.buffer.clear()  # a new drawing starts
            self._idle = 0
            self.buffer.append(
                tip[0], tip[1], t
            )  # we append x and y position of tip and the time t to the buffer, which is a deque of last 60 points
        else:
            self._idle += 1  # i increment the idle if the finger is not pointing
        return self.buffer.points()

    def reset(self) -> None:
        self.buffer.clear()
        self._idle = 0
