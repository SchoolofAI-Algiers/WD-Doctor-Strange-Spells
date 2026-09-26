"""Hand tracking module: wraps MediaPipe's HandLandmarker (Tasks API).

Converts raw MediaPipe output into the a list of hands, each hand a
(21, 3) array of [x_px, y_px, z_raw].
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


@dataclass
class HandLandmarks:
    """One detected hand's landmarks and metadata.

    Attributes:
        landmarks_norm: (21, 3) array of [x, y, z], MediaPipe's raw
            normalized output (x, y in 0..1, z relative depth).
        landmarks_px: (21, 3) array of [x_px, y_px, z]. x, y converted
            to pixel coordinates; z passed through unchanged (this is
            the array the geometry module consumes).
        handedness: "Left" or "Right".
        score: Detection confidence for the handedness label, in [0, 1].
    """

    landmarks_norm: np.ndarray
    landmarks_px: np.ndarray
    handedness: str
    score: float


class HandTracker:
    """Detects hands in a frame and extracts their 21 landmarks.

    Wraps MediaPipe's Tasks API (`HandLandmarker`) — the legacy
    `mp.solutions.hands` API is not available in mediapipe>=0.10.31+.
    """

    def __init__(
        self,
        model_path: str = "hand_landmarker.task",
        max_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        base_options = BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=vision.RunningMode.VIDEO,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def process(self, frame_bgr: np.ndarray) -> list[HandLandmarks]:
        """Run detection on one BGR frame.

        Args:
            frame_bgr: Frame from Capture, shape (H, W, 3), BGR, uint8.

        Returns:
            List of HandLandmarks, one per detected hand. Empty list
            if no hand is detected (never None).
        """
        frame_height, frame_width = frame_bgr.shape[:2]

        # Stage 1: BGR -> RGB (MediaPipe expects RGB, OpenCV gives BGR)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Stage 2: run MediaPipe (VIDEO mode needs a monotonically
        # increasing timestamp so it can use tracking, not just
        # per-frame detection)
        self._timestamp_ms += 33
        result = self._landmarker.detect_for_video(
            mp_image, self._timestamp_ms
        )

        # Stages 3 & 4: convert coordinates, package the result
        return self._convert(result, frame_width, frame_height)

    def _convert(
        self, result, frame_width: int, frame_height: int
    ) -> list[HandLandmarks]:
        if not result.hand_landmarks:
            return []

        hands: list[HandLandmarks] = []
        for i, hand_lms in enumerate(result.hand_landmarks):
            landmarks_norm = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32
            )
            landmarks_px = np.array(
                [
                    [lm.x * frame_width, lm.y * frame_height, lm.z]
                    for lm in hand_lms
                ],
                dtype=np.float32,
            )

            if result.handedness and len(result.handedness) > i:
                label = result.handedness[i][0].category_name
                score = result.handedness[i][0].score
            else:
                label, score = "Unknown", 0.0

            hands.append(
                HandLandmarks(
                    landmarks_norm=landmarks_norm,
                    landmarks_px=landmarks_px,
                    handedness=label,
                    score=score,
                )
            )
        return hands

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()

    def __enter__(self) -> HandTracker:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
