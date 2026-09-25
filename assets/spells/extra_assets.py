import math

import math
import os
import time
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ORANGE = (0, 0, 255)
GOLD = (35, 35, 255)
HOT = (80, 80, 255)
GLOW = 0.8


def polar(center, r, deg):
    a = math.radians(deg)
    return int(center[0] + r * math.cos(a)), int(center[1] + r * math.sin(a))


def dashed_ring(layer, center, r, start_deg, n, fill, color, thickness):
    step = 360 / n
    for i in range(n):
        a0 = start_deg + i * step
        cv2.ellipse(
            layer,
            center,
            (int(r), int(r)),
            0,
            a0,
            a0 + step * fill,
            color,
            thickness,
            cv2.LINE_AA,
        )


def polygon(layer, center, r, start_deg, sides, color, thickness):
    pts = [polar(center, r, start_deg + i * 360 / sides) for i in range(sides)]
    cv2.polylines(
        layer,
        [np.array(pts, np.int32).reshape(-1, 1, 2)],
        True,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_shield(
    layer, center, R, t, colors, alpha=1.0, angle_deg=0, sides=4
):
    c_orange, c_gold, c_hot = colors["ORANGE"], colors["GOLD"], colors["HOT"]

    def tint(c):
        return tuple(int(v * alpha) for v in c)

    th = max(1, int(R * 0.018))
    AA = cv2.LINE_AA

    cv2.circle(layer, center, int(R), tint(c_orange), th * 2, AA)
    cv2.circle(layer, center, int(R * 0.94), tint(c_gold), th, AA)

    for i in range(60):
        deg = i * 6 + t * 10 + angle_deg
        inner = 0.84 if i % 5 == 0 else 0.88
        cv2.line(
            layer,
            polar(center, R * inner, deg),
            polar(center, R * 0.92, deg),
            tint(c_gold),
            1,
            AA,
        )

    dashed_ring(
        layer,
        center,
        R * 0.78,
        t * 35 + angle_deg,
        16,
        0.65,
        tint(c_orange),
        th * 2,
    )
    cv2.circle(layer, center, int(R * 0.70), tint(c_gold), th, AA)

    for off in (0, 45):
        polygon(
            layer,
            center,
            R * 0.70,
            -t * 25 + off + angle_deg,
            sides,
            tint(c_orange),
            th,
        )

    dashed_ring(
        layer,
        center,
        R * 0.50,
        -t * 50 + angle_deg,
        10,
        0.5,
        tint(c_gold),
        th * 2,
    )
    # 6. Wild Outward Spark Lines (Shooting from shield edge)
    num_sparks = 40  # More lines = denser spark portal effect
    for i in range(num_sparks):
        deg = i * (360 / num_sparks) + t * 120 + angle_deg

        # Animate progress from 0.0 (ring edge) to 1.0 (outer boundary)
        progress = ((t * 4 + i * 0.17) % 1.0)

        # Spark inner point and flying outer tip
        start_dist = R + (progress * 0.1 * R)
        spark_length = (0.05 + 0.25 * math.sin(progress * math.pi)) * R
        end_dist = start_dist + spark_length

        p_start = polar(center, start_dist, deg)
        p_end = polar(center, end_dist, deg)

        # 1. Glowing spark line vector pointing outward
        cv2.line(layer, p_start, p_end, tint(c_orange), max(1, th // 2), AA)

        # 2. Bright hot white/gold head at the leading tip
        cv2.line(
            layer,
            p_end,
            polar(center, end_dist + 2, deg),
            tint(c_hot),
            max(1, th),
            AA,
        )

        # 3. Tiny flying spark head dot
        cv2.circle(
            layer, p_end, max(1, int(R * 0.012)), tint(c_gold), -1, AA
        )
    for i in range(8):
        deg = i * 45 + t * 20 + angle_deg
        cv2.circle(
            layer,
            polar(center, R * 0.60, deg),
            max(2, int(R * 0.04)),
            tint(c_gold),
            th,
            AA,
        )
        cv2.line(
            layer,
            polar(center, R * 0.38, deg),
            polar(center, R * 0.55, deg),
            tint(c_orange),
            1,
            AA,
        )

    cv2.circle(layer, center, max(2, int(R * 0.06)), tint(c_hot), -1, AA)


def composite(img, layer):
    layer = layer.astype(np.float32)
    glow = cv2.GaussianBlur(layer, (0, 0), 10)
    total = np.clip(layer + glow * GLOW, 0, 255)
    m = total[:, :, 2:3] / 255.0
    out = img.astype(np.float32) * (1.0 - m) + total
    return np.clip(out, 0, 255).astype(np.uint8)


def count_raised_fingers(pts):
    fingers = ((8, 6), (12, 10), (16, 14), (20, 18))
    count = 0
    for tip, pip in fingers:
        d_tip = math.hypot(pts[tip][0] - pts[0][0], pts[tip][1] - pts[0][1])
        d_pip = math.hypot(pts[pip][0] - pts[0][0], pts[pip][1] - pts[0][1])
        if d_tip > d_pip:
            count += 1
    return count