"""Hand geometry computations from MediaPipe landmarks."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]


def distance(p1: FloatArray, p2: FloatArray) -> float:
    """Compute the Euclidean distance between two points.

    Args:
        p1: First point, array of shape (2,).
        p2: Second point, array of shape (2,).

    Returns:
        The Euclidean distance between p1 and p2.
    """
    d = float(np.sqrt(np.sum((p2 - p1) ** 2)))
    return d


def norm2px(landmarks_norm: FloatArray, image_width: int, image_height: int) -> FloatArray:
    """Convert normalized MediaPipe landmarks to pixel coordinates.

    Args:
        landmarks_norm: Array of shape (N, 2) or (N, 3) with normalized x, y(, z).
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        Array of shape (N, 2) with x, y in pixels. The z column is dropped.
    """
    resolution = np.array([image_width, image_height])
    landmarks_px: FloatArray = landmarks_norm[:, :2] * resolution
    return landmarks_px


def palm_center(landmarks_px: FloatArray, palm_center_strategy: str = "five_point") -> FloatArray:
    """Compute the palm center in pixel coordinates.

    Args:
        landmarks_px: Array of shape (21, 2) with landmarks in pixels.
        palm_center_strategy: One of 'five_point', 'wrist_only', 'mcp_only',
            'weighted_mcp'.

    Returns:
        Array of shape (2,) with the palm center x, y in pixels.

    Raises:
        ValueError: If the strategy is unknown.
    """
    center: FloatArray
    if palm_center_strategy == "five_point":
        palm_indices = [0, 5, 9, 13, 17]
        center = np.mean(landmarks_px[palm_indices], axis=0)

    elif palm_center_strategy == "wrist_only":
        center = landmarks_px[0]

    elif palm_center_strategy == "mcp_only":
        palm_indices = [5, 9, 13, 17]
        center = np.mean(landmarks_px[palm_indices], axis=0)

    elif palm_center_strategy == "weighted_mcp":
        palm_indices = [0, 5, 9, 13, 17]
        center = np.average(landmarks_px[palm_indices], weights=[1, 1, 2, 1, 1], axis=0)
    else:
        raise ValueError(f"Unknown palm_center_strategy: {palm_center_strategy!r}")

    return center


def hand_size(landmarks_px: FloatArray, hand_size_method: str = "wrist_middle_tip") -> float:
    """Compute the hand size in pixels.

    Args:
        landmarks_px: Array of shape (21, 2) with landmarks in pixels.
        hand_size_method: One of 'wrist_middle_tip', 'wrist_index_tip', 'avg_fingers'.

    Returns:
        The hand size in pixels.

    Raises:
        ValueError: If the method is unknown.
    """
    if hand_size_method == "wrist_middle_tip":
        hand_size_px = distance(landmarks_px[0], landmarks_px[12])

    elif hand_size_method == "wrist_index_tip":
        hand_size_px = distance(landmarks_px[0], landmarks_px[8])

    elif hand_size_method == "avg_fingers":
        distances = [
            distance(landmarks_px[0], landmarks_px[12]),
            distance(landmarks_px[0], landmarks_px[8]),
            distance(landmarks_px[0], landmarks_px[16]),
            distance(landmarks_px[0], landmarks_px[20]),
        ]
        hand_size_px = float(np.mean(distances))

    else:
        raise ValueError(f"Unknown hand_size_method: {hand_size_method!r}")

    return hand_size_px


def palm_orientation(landmarks_px: FloatArray) -> float:
    """Compute the palm orientation angle in the image plane.

    Uses the vector from the middle-finger MCP (landmark 9) to the index-finger
    MCP (landmark 5).

    Args:
        landmarks_px: Array of shape (21, 2) with landmarks in pixels.

    Returns:
        The angle in radians, in [-pi, pi]. Image y axis points down.
    """
    v = landmarks_px[5] - landmarks_px[9]
    palm_angle_rad = float(np.arctan2(v[1], v[0]))
    return palm_angle_rad


def landmark_normalization(landmarks_px: FloatArray, hand_size_px: float) -> FloatArray:
    """Center landmarks on the wrist and scale them by the hand size.

    Args:
        landmarks_px: Array of shape (N, 2) with landmarks in pixels.
        hand_size_px: Hand size in pixels, used as the scale.

    Returns:
        Array of shape (N, 2) with the wrist at the origin. All zeros if the
        hand size is not greater than 1e-6.
    """
    if hand_size_px > 1e-6:
        centered = landmarks_px - landmarks_px[0]
        normalized: FloatArray = centered / hand_size_px
        return normalized
    else:
        return np.zeros_like(landmarks_px)


@dataclass(frozen=True)
class GeometryConfig:
    """Configuration of the geometry computations.

    Attributes:
        palm_center_strategy: Strategy passed to palm_center.
        hand_size_method: Method passed to hand_size.
    """

    palm_center_strategy: str = "five_point"
    hand_size_method: str = "wrist_middle_tip"


@dataclass(frozen=True)
class HandGeometry:
    """Geometry of one hand.

    Attributes:
        palm_center_px: Palm center (x, y) in pixels.
        palm_center_norm: Palm center (x, y) as a fraction of the image size.
        hand_size_px: Hand size in pixels.
        hand_size_norm: Hand size as a fraction of the longer image side.
        palm_angle_rad: Palm orientation angle in radians.
        landmarks_norm: Array of shape (N, 2), wrist-centered and scaled by hand size.
    """

    palm_center_px: tuple[float, float]
    palm_center_norm: tuple[float, float]
    hand_size_px: float
    hand_size_norm: float
    palm_angle_rad: float
    landmarks_norm: FloatArray


class GeometryCalculator:
    """Computes the geometry of a hand from its landmarks."""

    def __init__(
        self, image_width: int, image_height: int, config: GeometryConfig | None = None
    ) -> None:
        """Create a calculator for a given image size.

        Args:
            image_width: Image width in pixels.
            image_height: Image height in pixels.
            config: Geometry configuration. Defaults to GeometryConfig().
        """
        self.image_width = image_width
        self.image_height = image_height
        self.config = config or GeometryConfig()

    def compute(self, hand: Any) -> HandGeometry:
        """Compute hand geometry.

        Args:
            hand: HandLandmarks (from the tracking subsystem). Must expose
                `landmarks_norm`, an (N, 2) or (N, 3) array of MediaPipe
                normalized coordinates.

        Returns:
            The computed HandGeometry.
        """
        landmarks_norm = hand.landmarks_norm
        landmarks_px = norm2px(landmarks_norm, self.image_width, self.image_height)
        palm_center_px = palm_center(landmarks_px, self.config.palm_center_strategy)
        palm_center_norm = palm_center_px / np.array([self.image_width, self.image_height])
        hand_size_px = hand_size(landmarks_px, self.config.hand_size_method)
        hand_size_norm = hand_size_px / max(
            self.image_width, self.image_height
        )  # fraction of the longer image side
        palm_angle_rad = palm_orientation(landmarks_px)
        landmarks_centered = landmark_normalization(landmarks_px, hand_size_px)

        return HandGeometry(
            palm_center_px=(float(palm_center_px[0]), float(palm_center_px[1])),
            palm_center_norm=(float(palm_center_norm[0]), float(palm_center_norm[1])),
            hand_size_px=hand_size_px,
            hand_size_norm=hand_size_norm,
            palm_angle_rad=palm_angle_rad,
            landmarks_norm=landmarks_centered,
        )
