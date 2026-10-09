import math
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from src.gestures.recognizer import Gesture, PoseState
from src.gestures.recognizer import GestureState as RecognizedGesture

ASSET_SIZE = 512
ASSET_RADIUS = 256
_ASSET_CACHE: dict[tuple[str, bool, tuple[int, int, int] | None], NDArray[np.float32]] = {}

SPELL_FOR: dict[Gesture, str] = {  # recognized gesture -> spell id in make_spells()
    Gesture.OPEN_PALM: "shield",
}


@dataclass
class ActiveGesture:
    state: PoseState
    confidence: float


def to_active(g: RecognizedGesture) -> ActiveGesture:
    on = g.gesture in SPELL_FOR and g.frames_held > 0
    return ActiveGesture(
        state=PoseState.ACTIVE if on else PoseState.IDLE,
        confidence=g.confidence if on else 0.0,
    )


# Backward-compat alias: old tests use sr.GestureState, canonical enum is PoseState.
GestureState = PoseState


@dataclass
class LayerConfig:
    asset: str
    rotation_speed: float = 0.0  # rad/s, 0 = does not spin
    scale_mult: float = 1.0  # size relative to the spell radius
    offset: tuple[float, float] = (0.0, 0.0)  # in spell radii, rotated with the hand
    follow_hand: bool = True  # False = stays upright instead of tilting with the hand
    opacity_mult: float = 1.0
    key_white: bool = False  # treat white as transparent (JPG line art)
    tint: tuple[int, int, int] | None = None  # recolour the artwork, BGR order


@dataclass
class SpellConfig:
    layers: list[LayerConfig] = field(default_factory=list)  # drawn in order, first = bottom
    base_scale: float = 1.5
    opacity: float = 0.9
    fade_in_frames: int = 5
    fade_out_frames: int = 8
    min_radius_px: float = 20.0
    glow_enabled: bool = False


@dataclass
class SpellTransform:
    position_px: tuple[float, float]
    scale: float
    rotation_rad: float
    opacity: float
    spell_id: str = "default"


@dataclass
class _HandState:
    fade: float = 0.0  # 0 = invisible, 1 = fully visible
    confidence: float = 1.0  # last confidence seen
    last_transform: SpellTransform | None = None  # used to draw the fade-out when lost
    spell_id: str | None = None  # which spell this hand is currently showing


class SpellRenderer:
    def __init__(self, spells: dict[str, SpellConfig], frame_shape: tuple[int, int]) -> None:
        self._spells = spells
        self.H, self.W = frame_shape  # (height, width)

        for name, cfg in spells.items():
            if not cfg.layers:
                raise ValueError(f"Spell '{name}' has no layers")

        # every layer of every spell is loaded once, at startup
        self._assets: dict[str, list[NDArray[np.float32]]] = {
            name: [load_asset(layer.asset, layer.key_white, layer.tint) for layer in cfg.layers]
            for name, cfg in spells.items()
        }

        # how far each spell reaches from the palm, in spell radii.
        # Used to size the drawing window so no layer is cut off.
        self._extent: dict[str, float] = {
            name: max(layer.scale_mult + math.hypot(*layer.offset) for layer in cfg.layers)
            for name, cfg in spells.items()
        }

        self._hands: dict[int, _HandState] = {}

        # Allocated ONCE. Flat so any (h, w) sub-window can be a contiguous view.
        cap = self.H * self.W
        self._warp_buf = np.zeros(cap * 4, np.float32)
        self._tmp_buf = np.zeros(cap * 3, np.float32)
        self._inv_buf = np.zeros(cap, np.float32)
        self._M = np.zeros((2, 3), np.float64)

    def _is_active(self, gesture: Any) -> bool:
        return bool(gesture.state == PoseState.ACTIVE)

    def compute_transform(
        self,
        geometry: Any,
        gesture: Any,
        motion: Any,
        animation_time: float,
        hand_id: int = 0,
        spell_id: str = "default",
    ) -> SpellTransform:
        cfg = self._spells[spell_id]  # an unknown name raises KeyError immediately

        st = self._hands.get(hand_id)
        if st is None:
            st = self._hands[hand_id] = _HandState()

        if st.spell_id != spell_id:  # the hand switched to a different spell
            st.spell_id = spell_id
            st.fade = 0.0  # the new spell fades in from invisible

        size = float(geometry.hand_size_px)
        if not math.isfinite(size) or size <= 0:
            size = cfg.min_radius_px  # bad measurement: draw at minimum size

        spell_radius = size * cfg.base_scale
        spell_radius = min(max(spell_radius, cfg.min_radius_px), max(self.H, self.W))
        scale = spell_radius / ASSET_RADIUS

        # step the fade level toward 1 (active) or 0 (not active)
        if self._is_active(gesture):
            st.fade = min(1.0, st.fade + 1.0 / max(1, cfg.fade_in_frames))
        else:
            st.fade = max(0.0, st.fade - 1.0 / max(1, cfg.fade_out_frames))

        st.confidence = float(gesture.confidence)
        transform = SpellTransform(
            position_px=geometry.palm_center_px,
            scale=scale,
            rotation_rad=geometry.palm_angle_rad,
            opacity=cfg.opacity * st.confidence * st.fade,
            spell_id=spell_id,
        )
        st.last_transform = transform
        return transform

    def ghost_spells(self, seen_ids: set[int]) -> list[tuple[SpellTransform, None]]:
        """Fade out hands that were tracked before but are missing this frame.

        Returns a list of (transform, None) pairs to append to the spells list.
        """
        ghosts: list[tuple[SpellTransform, None]] = []
        for hid in [h for h in self._hands if h not in seen_ids]:
            st = self._hands[hid]
            if st.last_transform is None or st.spell_id is None:
                del self._hands[hid]
                continue
            cfg = self._spells[st.spell_id]
            st.fade -= 1.0 / max(1, cfg.fade_out_frames)
            if st.fade <= 0.0:
                del self._hands[hid]  # fully faded: forget this hand
                continue
            t = replace(
                st.last_transform,
                opacity=cfg.opacity * st.confidence * st.fade,
            )
            ghosts.append((t, None))  # render() doesn't use the gesture
        return ghosts

    def _draw_layer(
        self,
        roi: NDArray[np.uint8],
        warp: NDArray[np.float32],
        tmp: NDArray[np.float32],
        inv: NDArray[np.float32],
        asset: NDArray[np.float32],
        scale: float,
        dst_c: tuple[float, float],
        angle: float,
        opacity: float,
    ) -> None:
        build_affine(angle, scale, (ASSET_RADIUS, ASSET_RADIUS), dst_c, out=self._M)
        cv2.warpAffine(
            asset,
            self._M,
            (warp.shape[1], warp.shape[0]),
            dst=warp,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        if opacity < 1.0:
            np.multiply(warp, opacity, out=warp)  # in place, no allocation
        composite_premultiplied(roi, warp, tmp, inv)

    def render(
        self,
        frame_bgr: NDArray[np.uint8],
        transform: SpellTransform,
        gesture: Any,
        animation_time: float,
    ) -> NDArray[np.uint8]:
        opacity = min(max(transform.opacity, 0.0), 1.0)
        if opacity <= 0.0:
            return frame_bgr

        cfg = self._spells[transform.spell_id]
        assets = self._assets[transform.spell_id]

        base_r = transform.scale * ASSET_RADIUS  # spell radius in px
        r = base_r * self._extent[transform.spell_id]  # window covers every layer
        px, py = transform.position_px

        x0 = max(int(math.floor(px - r)), 0)
        y0 = max(int(math.floor(py - r)), 0)
        x1 = min(int(math.ceil(px + r)), self.W)
        y1 = min(int(math.ceil(py + r)), self.H)
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            return frame_bgr  # fully off-screen

        roi = frame_bgr[y0:y1, x0:x1]
        warp = self._warp_buf[: h * w * 4].reshape(h, w, 4)
        tmp = self._tmp_buf[: h * w * 3].reshape(h, w, 3)
        inv = self._inv_buf[: h * w].reshape(h, w, 1)

        hand_rot = transform.rotation_rad
        cos_h, sin_h = math.cos(hand_rot), math.sin(hand_rot)

        for layer, asset in zip(cfg.layers, assets, strict=True):
            # the offset is rotated with the hand, so "in front of the palm" stays in front
            ox = (layer.offset[0] * cos_h - layer.offset[1] * sin_h) * base_r
            oy = (layer.offset[0] * sin_h + layer.offset[1] * cos_h) * base_r
            dst_c = (px + ox - x0, py + oy - y0)

            angle = layer_angle(
                hand_rot if layer.follow_hand else 0.0, layer.rotation_speed, animation_time
            )

            self._draw_layer(
                roi,
                warp,
                tmp,
                inv,
                asset,
                transform.scale * layer.scale_mult,
                dst_c,
                angle,
                opacity * layer.opacity_mult,
            )
        return frame_bgr

    # draws one spell per hand, one after another (they share the scratch buffers)
    def render_all(
        self,
        frame_bgr: NDArray[np.uint8],
        spells: Iterable[tuple[SpellTransform, Any]],
        animation_time: float,
    ) -> NDArray[np.uint8]:
        """spells: iterable of (SpellTransform, gesture), one per hand."""
        for transform, gesture in spells:
            self.render(frame_bgr, transform, gesture, animation_time)
        return frame_bgr


def load_asset(
    path: str, key_white: bool = False, tint: tuple[int, int, int] | None = None
) -> NDArray[np.float32]:
    key = (path, key_white, tint)
    if key in _ASSET_CACHE:
        return _ASSET_CACHE[key]

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Spell asset not found or unreadable: {path}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if key_white and img.shape[2] == 3:
        # line art on white: white becomes transparent, ink keeps its colour
        bgr = img.astype(np.float32)
        m = bgr.min(axis=2, keepdims=True)  # 255 = paper, 0 = ink
        color = bgr - m  # compute BEFORE wiping the haze
        m[m > 240] = 255.0  # wipe JPEG haze
        alpha = 1.0 - m / 255.0
        color *= alpha > 0  # no colour where fully transparent
        img = np.concatenate([color, alpha], axis=2)
    else:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        fimg: NDArray[np.float32] = img.astype(np.float32)
        fimg[:, :, 3] = fimg[:, :, 3] / 255.0  # alpha: 0..255 -> 0..1
        fimg[:, :, :3] = fimg[:, :, :3] * fimg[:, :, 3:4]  # premultiply: color * alpha
        img = fimg

    if tint is not None:  # optional recolour, keeps the alpha shape
        img[:, :, :3] = np.array(tint, np.float32) * img[:, :, 3:4]

    # keep the aspect ratio, pad to 512x512 with transparent pixels
    h, w = img.shape[:2]
    k = ASSET_SIZE / max(h, w)
    new_w, new_h = max(1, round(w * k)), max(1, round(h * k))
    interp = cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR
    img = cv2.resize(img, (new_w, new_h), interpolation=interp)

    canvas = np.zeros((ASSET_SIZE, ASSET_SIZE, 4), np.float32)
    y, x = (ASSET_SIZE - new_h) // 2, (ASSET_SIZE - new_w) // 2
    canvas[y : y + new_h, x : x + new_w] = img

    _ASSET_CACHE[key] = np.ascontiguousarray(canvas)
    return _ASSET_CACHE[key]


def layer_angle(hand_rotation_rad: float, speed_rad_s: float, animation_time: float) -> float:
    return hand_rotation_rad + speed_rad_s * animation_time


def build_affine(
    angle: float,
    scale: float,
    src_center: tuple[float, float],
    dst_center: tuple[float, float],
    out: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
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


def composite_premultiplied(
    roi: NDArray[np.uint8],
    layer: NDArray[np.float32],
    tmp: NDArray[np.float32],
    inv: NDArray[np.float32],
) -> None:
    np.subtract(1.0, layer[:, :, 3:4], out=inv)  # 1 - alpha
    np.multiply(roi, inv, out=tmp)  # background * (1 - alpha)
    np.add(tmp, layer[:, :, :3], out=tmp)  # + premultiplied spell color
    np.clip(tmp, 0, 255, out=tmp)
    roi[...] = tmp  # write back into the frame
