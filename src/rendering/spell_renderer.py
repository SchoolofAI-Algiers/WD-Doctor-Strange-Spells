from dataclasses import dataclass
import cv2
import numpy as np
import math

ASSET_SIZE = 512
ASSET_RADIUS = 256
_ASSET_CACHE: dict[str, np.ndarray] = {}


@dataclass
class SpellConfig:
    inner_asset: str = "assets/spells/inner/orange.png"
    outer_asset: str = "assets/spells/outer/dark_red.png"
    base_scale: float = 1.5
    inner_rotation_speed: float = 2.0     # rad/s
    outer_rotation_speed: float = -1.0    # rad/s
    opacity: float = 0.9
    fade_in_frames: int = 5
    fade_out_frames: int = 8
    min_radius_px: float = 20.0
    size_smoothing: float = 0.2
    glow_enabled: bool = False


@dataclass
class SpellTransform:
    position_px: tuple[float, float]
    scale: float
    rotation_rad: float
    opacity: float


class SpellRenderer:

    def __init__(self, config: SpellConfig, frame_shape: tuple[int, int]):
        self.cfg = config
        self.H, self.W = frame_shape  # (height, width)
        self.inner = load_asset(config.inner_asset)
        self.outer = load_asset(config.outer_asset)
        self._smooth_size: float | None = None

        # Allocated ONCE. Flat so any (h, w) sub-window can be a contiguous view.
        cap = self.H * self.W
        self._warp_buf = np.zeros(cap * 4, np.float32)
        self._tmp_buf = np.zeros(cap * 3, np.float32)
        self._inv_buf = np.zeros(cap, np.float32)
        self._M = np.zeros((2, 3), np.float64)

    def reset_smoothing(self) -> None:
        self._smooth_size = None  # Hand tracking gives you a slightly different hand_size_px every frame, even if you hold your hand perfectly still. The landmark detector is noisy, so you might get 100, 103, 98, 104, 99... Since the spell's size is computed directly from that number, the spell would constantly flicker and pulse by a few pixels. It looks jittery and cheap.

    def compute_transform(self, geometry, gesture, motion, animation_time: float) -> SpellTransform:
        raw = float(geometry.hand_size_px)
        if not math.isfinite(raw) or raw <= 0:
            raw = self._smooth_size or 1.0  # bad measurement: keep last good value

        # exponential moving average
        if self._smooth_size is None:
            self._smooth_size = raw
        else:
            a = self.cfg.size_smoothing
            self._smooth_size = (1 - a) * self._smooth_size + a * raw

        spell_radius = self._smooth_size * self.cfg.base_scale
        spell_radius = min(max(spell_radius, self.cfg.min_radius_px), max(self.H, self.W))
        scale = spell_radius / ASSET_RADIUS

        return SpellTransform(
            position_px=geometry.palm_center_px,
            scale=scale,
            rotation_rad=geometry.palm_angle_rad,
            opacity=self.cfg.opacity * gesture.confidence,
        )

    def _draw_layer(self, roi, warp, tmp, inv, asset, scale, dst_c, angle, opacity) -> None:
        build_affine(angle, scale, (ASSET_RADIUS, ASSET_RADIUS), dst_c, out=self._M)
        cv2.warpAffine(asset, self._M, (warp.shape[1], warp.shape[0]), dst=warp,
                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                       borderValue=(0, 0, 0, 0))
        if opacity < 1.0:
            np.multiply(warp, opacity, out=warp)  # in place, no allocation
        composite_premultiplied(roi, warp, tmp, inv)

    def render(self, frame_bgr: np.ndarray, transform: SpellTransform,
               gesture, animation_time: float) -> np.ndarray:

        opacity = min(max(transform.opacity, 0.0), 1.0)  # NEW
        if opacity <= 0.0:  # NEW
            return frame_bgr  # NEW

        r = transform.scale * ASSET_RADIUS
        px, py = transform.position_px

        x0 = max(int(math.floor(px - r)), 0)
        y0 = max(int(math.floor(py - r)), 0)
        x1 = min(int(math.ceil(px + r)), self.W)
        y1 = min(int(math.ceil(py + r)), self.H)
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            return frame_bgr  # fully off-screen

        roi = frame_bgr[y0:y1, x0:x1]
        warp = self._warp_buf[:h * w * 4].reshape(h, w, 4)
        tmp = self._tmp_buf[:h * w * 3].reshape(h, w, 3)
        inv = self._inv_buf[:h * w].reshape(h, w, 1)
        dst_c = (px - x0, py - y0)

        cfg = self.cfg
        self._draw_layer(roi, warp, tmp, inv, self.outer, transform.scale, dst_c,
                         layer_angle(transform.rotation_rad, cfg.outer_rotation_speed, animation_time),
                         opacity)  # NEW argument
        self._draw_layer(roi, warp, tmp, inv, self.inner, transform.scale, dst_c,
                         layer_angle(transform.rotation_rad, cfg.inner_rotation_speed, animation_time),
                         opacity)  # NEW argument
        return frame_bgr






def load_asset(path: str) -> np.ndarray:
    if path in _ASSET_CACHE:
        return _ASSET_CACHE[path]

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Spell asset not found or unreadable: {path}")

    # make sure we have 4 channels (B, G, R, A)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    img = img.astype(np.float32)
    img[:, :, 3] /= 255.0                       # alpha: 0..255 -> 0..1
    img[:, :, :3] *= img[:, :, 3:4]             # premultiply: color * alpha
    img = cv2.resize(img, (ASSET_SIZE, ASSET_SIZE), interpolation=cv2.INTER_AREA)

    _ASSET_CACHE[path] = np.ascontiguousarray(img)
    return _ASSET_CACHE[path]

def layer_angle(hand_rotation_rad: float, speed_rad_s: float, animation_time: float) -> float:
    return hand_rotation_rad + speed_rad_s * animation_time

def build_affine(angle, scale, src_center, dst_center, out=None):
    if out is None:
        out = np.zeros((2, 3), np.float64)
    c = np.cos(angle) * scale
    s = np.sin(angle) * scale
    sx, sy = src_center
    dx, dy = dst_center
    out[0, 0], out[0, 1] = c, -s
    out[1, 0], out[1, 1] = s, c
    out[0, 2] = dx - (c * sx - s * sy)
    out[1, 2] = dy - (s * sx + c * sy)
    return out

def composite_premultiplied(roi, layer, tmp, inv):
    np.subtract(1.0, layer[:, :, 3:4], out=inv)   # 1 - alpha
    np.multiply(roi, inv, out=tmp)                # background * (1 - alpha)
    np.add(tmp, layer[:, :, :3], out=tmp)         # + premultiplied spell color
    np.clip(tmp, 0, 255, out=tmp)
    roi[...] = tmp


    # write back into the frame