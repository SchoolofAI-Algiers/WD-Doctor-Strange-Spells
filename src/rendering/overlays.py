"""Overlays for spells that are not made of rotating PNG rings: a looping video (Shield)
and a photo shown through a soft oval window (Portal). Pure OpenCV/NumPy, no pipeline logic."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

Frame = NDArray[np.uint8]
Slices = tuple[slice, slice]


class Fader:
    """0 -> 1 while `on`, 1 -> 0 when off, a little each frame."""

    def __init__(self, frames_in: int = 6, frames_out: int = 10) -> None:
        self.level = 0.0
        self._up = 1.0 / max(1, frames_in)
        self._down = 1.0 / max(1, frames_out)

    def step(self, on: bool) -> float:
        """Move one frame toward 1 (on) or 0 (off) and return the new level."""
        self.level = min(1.0, self.level + self._up) if on else max(0.0, self.level - self._down)
        return self.level


def _clip(frame: Frame, x0: int, y0: int, w: int, h: int) -> tuple[Slices, Slices] | None:
    """Part of a (w, h) sprite placed at (x0, y0) that lands inside the frame.

    Returns (frame slices, sprite slices), or None if it is fully off-screen.
    """
    fx0, fy0 = max(x0, 0), max(y0, 0)
    fx1, fy1 = min(x0 + w, frame.shape[1]), min(y0 + h, frame.shape[0])
    if fx1 <= fx0 or fy1 <= fy0:
        return None
    return (
        (slice(fy0, fy1), slice(fx0, fx1)),
        (slice(fy0 - y0, fy1 - y0), slice(fx0 - x0, fx1 - x0)),
    )


class VideoSpell:
    """A looping video drawn with ADDITIVE blending: black pixels add nothing, so a video
    on a black background looks transparent. Frames are loaded once, at startup.

    Raises FileNotFoundError if the video can't be opened, ValueError if it has no frames.
    """

    def __init__(self, path: str | Path, size: int = 256) -> None:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        self.fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.frames: list[Frame] = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            h, w = f.shape[:2]
            s = min(h, w)  # crop the centre square so the circle is not stretched
            f = f[(h - s) // 2 : (h + s) // 2, (w - s) // 2 : (w + s) // 2]
            self.frames.append(cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA))
        cap.release()
        if not self.frames:
            raise ValueError(f"No frames in video: {path}")

    def draw(
        self,
        frame: Frame,
        center: tuple[float, float],
        radius: float,
        angle_rad: float,
        t: float,
        opacity: float,
    ) -> None:
        """Add the video frame for time `t` (seconds) around `center`, in place."""
        d = int(radius * 2)
        if d < 4 or opacity <= 0.0:
            return
        sprite = cv2.resize(self.frames[int(t * self.fps) % len(self.frames)], (d, d))
        m = cv2.getRotationMatrix2D((d / 2, d / 2), -math.degrees(angle_rad), 1.0)
        sprite = cv2.warpAffine(sprite, m, (d, d))
        sprite = cv2.convertScaleAbs(sprite, alpha=min(opacity, 1.0))
        clip = _clip(frame, int(center[0] - d / 2), int(center[1] - d / 2), d, d)
        if clip is None:
            return
        frame_sl, sprite_sl = clip
        roi = frame[frame_sl]
        cv2.add(roi, sprite[sprite_sl], dst=roi)  # saturating add: never wraps past 255


class ImagePortal:
    """A photo shown inside a soft-edged oval, like looking through a window.

    Raises FileNotFoundError if the image can't be read.
    """

    def __init__(self, path: str | Path) -> None:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        self.img = img
        self.aspect = img.shape[1] / img.shape[0]  # width / height

    def draw(
        self, frame: Frame, center: tuple[float, float], height_px: float, opacity: float
    ) -> None:
        """Blend the photo (oval mask, `height_px` tall) around `center`, in place."""
        h = int(height_px)
        w = int(h * self.aspect)
        if h < 8 or w < 8 or opacity <= 0.0:
            return
        sprite = cv2.resize(self.img, (w, h)).astype(np.float32)
        mask = np.zeros((h, w), np.float32)
        cv2.ellipse(mask, (w // 2, h // 2), (w // 2 - 3, h // 2 - 3), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, min(w, h) * 0.03))
        alpha = (mask * min(opacity, 1.0))[:, :, None]
        clip = _clip(frame, int(center[0] - w / 2), int(center[1] - h / 2), w, h)
        if clip is None:
            return
        frame_sl, sprite_sl = clip
        roi = frame[frame_sl]
        roi[...] = (roi * (1 - alpha[sprite_sl]) + sprite[sprite_sl] * alpha[sprite_sl]).astype(
            np.uint8
        )
