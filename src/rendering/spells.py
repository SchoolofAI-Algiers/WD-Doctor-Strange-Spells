"""Minimal spell presets: just enough to see a spinning shield on camera.

Run from the repo root:  python -m src.rendering.spells
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ASSETS = Path(__file__).resolve().parents[2] / "assets" / "spells"


@dataclass(frozen=True)
class LayerSpec:
    path: Path
    speed_deg_per_s: float          # sign = spin direction
    scale: float = 1.0              # diameter relative to the spell's diameter
    color_bgr: tuple[int, int, int] = (255, 255, 255)


@dataclass(frozen=True)
class SpellPreset:
    name: str
    layers: tuple[LayerSpec, ...]   # drawn in order, first = bottom


# Outer and inner spin in opposite directions.
SHIELD = SpellPreset(
    name="shield",
    layers=(
        LayerSpec(ASSETS / "outer" / "shield of seraphim.jpg", +40.0, 1.0, (0, 180, 255)),
        LayerSpec(ASSETS / "inner" / "shield green.jpg", -60.0, 0.7, (90, 255, 90)),
    ),
)


@lru_cache(maxsize=None)
def _load_alpha(path: Path) -> np.ndarray:
    """JPG line art on white -> float alpha in [0, 1] (lines opaque, white clear)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read spell asset: {path}")
    # min over channels: black AND green lines both come out dark -> opaque
    return 1.0 - img.min(axis=2).astype(np.float32) / 255.0


def draw_spell(frame: np.ndarray, preset: SpellPreset, center: tuple[int, int],
               radius: float, t: float | None = None) -> np.ndarray:
    """Draw `preset` onto `frame` (in place) centered at `center`."""
    t = time.perf_counter() if t is None else t
    h, w = frame.shape[:2]
    cx, cy = center

    for layer in preset.layers:
        alpha = _load_alpha(layer.path)
        ah, aw = alpha.shape
        diameter = 2.0 * radius * layer.scale
        m = cv2.getRotationMatrix2D((aw / 2, ah / 2), t * layer.speed_deg_per_s, diameter / aw)
        m[0, 2] += cx - aw / 2
        m[1, 2] += cy - ah / 2
        a = cv2.warpAffine(alpha, m, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)[..., None]
        color = np.array(layer.color_bgr, dtype=np.float32)
        frame[:] = (frame * (1.0 - a) + color * a).astype(np.uint8)
    return frame


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    start = time.perf_counter()
    while cap.isOpened():
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.flip(img, 1)
        ih, iw = img.shape[:2]
        draw_spell(img, SHIELD, (iw // 2, ih // 2), radius=min(iw, ih) * 0.35,
                   t=time.perf_counter() - start)
        cv2.imshow("shield", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()