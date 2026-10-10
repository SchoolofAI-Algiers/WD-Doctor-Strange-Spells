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

class ChromaVideo:
    """A video shot on a green screen, with the green keyed out (made transparent).

    Every frame is decoded and keyed ONCE at startup (downscaled so memory stays small).
    `lo`/`hi` control how green a pixel must be to start/finish disappearing.

    Raises FileNotFoundError if the video can't be opened, ValueError if it has no frames.
    """

    def __init__(
        self,
        path: str | Path,
        max_side: int = 480,
        lo: float = 20.0,
        hi: float = 70.0,
        despill: bool = True,
        loop: bool = True,
    ) -> None:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        self.fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.loop = loop
        self.frames: list[Frame] = []
        self.alphas: list[NDArray[np.uint8]] = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            h, w = f.shape[:2]
            k = min(1.0, max_side / max(h, w))
            if k < 1.0:
                f = cv2.resize(f, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
            bgr, alpha = self._key(f, lo, hi, despill)
            self.frames.append(bgr)
            self.alphas.append(alpha)
        cap.release()
        if not self.frames:
            raise ValueError(f"No frames in video: {path}")
        h, w = self.frames[0].shape[:2]
        self.aspect = w / h  # width / height

    @staticmethod
    def _key(f: Frame, lo: float, hi: float, despill: bool) -> tuple[Frame, NDArray[np.uint8]]:
        """Alpha from 'how much greener than red/blue'; optional green-spill removal."""
        b, g, r = (c.astype(np.float32) for c in cv2.split(f))
        rb = np.maximum(r, b)
        greenness = g - rb
        alpha = 1.0 - np.clip((greenness - lo) / (hi - lo), 0.0, 1.0)
        if despill:
            g = np.minimum(g, rb)  # removes the green tint left on the edges
            f = cv2.merge([b, g, r]).astype(np.uint8)
        return f, (alpha * 255).astype(np.uint8)

    def draw(
        self,
        frame: Frame,
        center: tuple[float, float],
        height_px: float,
        t: float,
        opacity: float,
    ) -> None:
        """Blend the keyed video frame for time `t` (seconds since it started), in place."""
        h = int(height_px)
        w = int(h * self.aspect)
        if h < 8 or w < 8 or opacity <= 0.0:
            return
        n = len(self.frames)
        i = int(t * self.fps)
        i = i % n if self.loop else min(max(i, 0), n - 1)
        sprite = cv2.resize(self.frames[i], (w, h)).astype(np.float32)
        a = cv2.resize(self.alphas[i], (w, h)).astype(np.float32) / 255.0
        alpha = (a * min(opacity, 1.0))[:, :, None]
        clip = _clip(frame, int(center[0] - w / 2), int(center[1] - h / 2), w, h)
        if clip is None:
            return
        frame_sl, sprite_sl = clip
        roi = frame[frame_sl]
        roi[...] = (roi * (1 - alpha[sprite_sl]) + sprite[sprite_sl] * alpha[sprite_sl]).astype(
            np.uint8
        )

class MirrorDimension:
    """Kaleidoscope that 'breaks reality': the camera image is folded into mirrored
    wedges around a point, spins, and expands outward with a glowing ring at its edge.
    Pure OpenCV/NumPy, no assets needed."""

    def __init__(
        self,
        segments: int = 6,
        spin: float = 0.5,  # rad/s
        zoom: float = 0.6,  # <1 magnifies the source image inside the wedges
        grow_seconds: float = 0.8,  # time for the effect to reach full size
        ring_color: tuple[int, int, int] = (0, 140, 255),  # BGR orange
    ) -> None:
        self.segments = max(2, segments)
        self.spin = spin
        self.zoom = zoom
        self.grow_seconds = max(1e-3, grow_seconds)
        self.ring_color = ring_color
        self._grid: tuple[NDArray[np.float32], NDArray[np.float32]] | None = None

    def draw(
        self,
        frame: Frame,
        center: tuple[float, float],
        t: float,  # seconds since the effect started
        level: float,  # 0..1 fade level (from a Fader)
    ) -> None:
        """Warp `frame` in place. Call it BEFORE drawing the other spells on the frame."""
        if level <= 0.0:
            return
        h, w = frame.shape[:2]
        if self._grid is None or self._grid[0].shape != (h, w):
            ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
            self._grid = (xs, ys)
        xs, ys = self._grid
        cx, cy = center

        dx, dy = xs - cx, ys - cy
        r = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx) + self.spin * t

        # fold every angle into one wedge: 0..seg, mirrored back and forth
        seg = np.pi / self.segments
        folded = np.abs((theta % (2 * seg)) - seg)
        src_x = (cx + r * self.zoom * np.cos(folded)).astype(np.float32)
        src_y = (cy + r * self.zoom * np.sin(folded)).astype(np.float32)
        warped = cv2.remap(frame, src_x, src_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # the kaleidoscope grows from the centre (ease-out), with a soft edge
        grow = min(1.0, t / self.grow_seconds)
        grow = 1.0 - (1.0 - grow) ** 3
        radius = grow * math.hypot(w, h)
        feather = max(12.0, 0.06 * radius)
        mask = np.clip((radius - r) / feather, 0.0, 1.0) * min(level, 1.0)
        mask = mask[:, :, None]
        frame[...] = (frame * (1 - mask) + warped * mask).astype(np.uint8)

        # glowing ring at the expanding edge
        if grow < 1.0 and radius > 4:
            ring = np.zeros_like(frame)
            cv2.circle(ring, (int(cx), int(cy)), int(radius), self.ring_color, 4, cv2.LINE_AA)
            ring = cv2.GaussianBlur(ring, (0, 0), 4)
            cv2.add(frame, (ring * min(level, 1.0)).astype(np.uint8), dst=frame)


class GlassShatter:
    """Reality breaking like glass AROUND THE HANDS. Many small shards, densest around each
    hand. A break zone (a circle around each hand) grows from the palm and follows the hand
    live; inside it every shard is rotated, shifted away from the hand, shaded differently
    and outlined with cracks. The cracks fade out into normal reality at the zone's edge.

    Call start() when the spell begins (builds new random shards), then draw() every frame.
    """

    def __init__(
        self,
        shards: int = 170,
        region_mult: float = 2.2,  # zone radius, in hand sizes
        max_shift: float = 0.03,  # shard drift, as a fraction of the frame diagonal
        max_rot: float = 0.09,  # shard rotation, radians
        grow_seconds: float = 0.45,  # time for the zone to reach full size
    ) -> None:
        self.shards = max(20, shards)
        self.region_mult = region_mult
        self.max_shift = max_shift
        self.max_rot = max_rot
        self.grow_seconds = max(1e-3, grow_seconds)
        self._ready = False

    def start(
        self,
        centers: list[tuple[float, float]],
        hand_size: float,
        frame_shape: tuple[int, ...],
    ) -> None:
        """Build new random shards, dense around each hand in `centers` (pixels)."""
        h, w = frame_shape[:2]
        rng = np.random.default_rng()
        self._r_max = max(60.0, hand_size * self.region_mult)
        n = self.shards

        # seeds: 70% packed around the hands (smallest shards near the palm), 30% anywhere
        n_near = int(0.7 * n / max(1, len(centers)))
        sx_list, sy_list = [], []
        for cx, cy in centers:
            a = rng.uniform(0, 2 * np.pi, n_near)
            r = self._r_max * 1.15 * rng.random(n_near) ** 1.7
            sx_list.append(cx + r * np.cos(a))
            sy_list.append(cy + r * np.sin(a))
        n_far = n - n_near * len(centers)
        sx_list.append(rng.uniform(0, w, n_far))
        sy_list.append(rng.uniform(0, h, n_far))
        sx = np.concatenate(sx_list).astype(np.float32)
        sy = np.concatenate(sy_list).astype(np.float32)
        k = len(sx)

        # which shard owns each pixel: computed at half resolution, then doubled (fast)
        hh, hw = (h + 1) // 2, (w + 1) // 2
        gy, gx = np.mgrid[0:hh, 0:hw].astype(np.float32) * 2.0
        best = np.full((hh, hw), np.inf, np.float32)
        lab_h = np.zeros((hh, hw), np.int32)
        for i in range(k):
            d = (gx - sx[i]) ** 2 + (gy - sy[i]) ** 2
            closer = d < best
            best[closer] = d[closer]
            lab_h[closer] = i
        labels = np.repeat(np.repeat(lab_h, 2, axis=0), 2, axis=1)[:h, :w]

        # crack lines = borders between shards (thin core + soft dark halo)
        edge = np.zeros((h, w), bool)
        edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
        edge[:-1, :] |= labels[:-1, :] != labels[1:, :]
        core = cv2.dilate(edge.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(np.float32)
        halo = cv2.GaussianBlur(core, (0, 0), 3)

        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        self._h, self._w, self._diag = h, w, math.hypot(w, h)
        self._xs, self._ys, self._labels = xs, ys, labels
        self._sx, self._sy = sx, sy
        self._mag = (0.4 + 0.6 * rng.random(k)).astype(np.float32)
        self._rot = rng.uniform(-self.max_rot, self.max_rot, k).astype(np.float32)
        self._facet = (1.0 + rng.uniform(-0.18, 0.18, k).astype(np.float32))[labels][:, :, None]
        self._core, self._halo = core[:, :, None], halo[:, :, None]
        self._ready = True

    def draw(
        self,
        frame: Frame,
        t: float,  # seconds since start()
        level: float,  # 0..1 fade
        centers: list[tuple[float, float]],  # current hand positions (live)
    ) -> None:
        """Shatter `frame` in place, around the hands in `centers`."""
        if not self._ready or level <= 0.0 or not centers or frame.shape[:2] != (self._h, self._w):
            return
        grow = min(1.0, t / self.grow_seconds)
        radius = (1.0 - (1.0 - grow) ** 3) * self._r_max  # current size of the break zone
        cs = np.array(centers, np.float32)  # (m, 2)

        # per pixel: distance to the nearest hand
        r_pix = np.hypot(self._xs - cs[0, 0], self._ys - cs[0, 1])
        for cx, cy in cs[1:]:
            np.minimum(r_pix, np.hypot(self._xs - cx, self._ys - cy), out=r_pix)

        # per shard: nearest hand, distance to it, and the direction away from it
        dists = np.hypot(self._sx[None, :] - cs[:, 0:1], self._sy[None, :] - cs[:, 1:2])
        near = dists.argmin(axis=0)
        cols = np.arange(len(self._sx))
        d_k = dists[near, cols]
        vx, vy = self._sx - cs[near, 0], self._sy - cs[near, 1]
        norm = np.maximum(np.hypot(vx, vy), 1e-6)

        # shards inside the zone move; the closer to the hand, the more they move
        inside = np.clip((radius - d_k) / (0.25 * radius + 1.0), 0.0, 1.0)
        closeness = 1.0 - 0.5 * np.clip(d_k / (radius + 1.0), 0.0, 1.0)
        amt = inside * closeness * (0.4 + 0.6 * min(1.0, t / (self.grow_seconds + 0.6)))
        push = self.max_shift * self._diag * self._mag * amt
        tx, ty = (vx / norm) * push, (vy / norm) * push
        ang = self._rot * amt
        c, s = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)

        # inverse map: each pixel samples its own shard's rotated + shifted content
        lab = self._labels
        dx = self._xs - self._sx[lab] - tx[lab]
        dy = self._ys - self._sy[lab] - ty[lab]
        src_x = self._sx[lab] + c[lab] * dx + s[lab] * dy
        src_y = self._sy[lab] - s[lab] * dx + c[lab] * dy
        warped = cv2.remap(frame, src_x, src_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        out = warped.astype(np.float32) * self._facet

        # the zone: full effect near the hand, fading out into normal reality at its edge
        reached = np.clip((radius - r_pix) / (0.3 * radius + 1.0), 0.0, 1.0)[:, :, None]
        out = out * (1 - self._halo * 0.45 * reached)  # dark seams
        out = out + self._core * reached * 230.0  # bright crack lines
        if t < 0.15:  # impact flash on the hands
            out = out + 255.0 * 0.5 * (1.0 - t / 0.15) * reached

        mask = reached * min(level, 1.0)
        frame[...] = np.clip(frame * (1 - mask) + out * mask, 0, 255).astype(np.uint8)