"""Pipeline orchestration (single-loop MVP).

Per frame:
    Capture -> HandTracker -> GeometryCalculator -> GestureRecognizer + MotionAnalyzer
            -> two-hand spells (Mirror / Ruby / Portal) -> SpellRenderer -> overlay

The pipeline holds almost no logic of its own: it calls each subsystem in order,
passes data between them, measures time and cleans up.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from src.capture.capture import Capture
from src.geometry.calculator import GeometryCalculator, GeometryConfig
from src.gestures.recognizer import (
    Gesture,
    GestureRecognizer,
    GestureState,
    PoseState,
    SpellDetector,
    TwoHandContext,
    hand_ids_from_handedness,
    is_mirror,
    is_portal,
    is_ruby,
    mirror_metrics,
)
from src.motion.analyzer import INDEX_TIP, MotionAnalyzer, MotionState
from src.pipeline.config import PipelineConfig
from src.pipeline.state import PipelineState
from src.rendering.overlays import Fader, ImagePortal, VideoSpell
from src.rendering.spell_renderer import (
    ASSET_RADIUS,
    SPELL_FOR,
    SpellRenderer,
    SpellTransform,
    to_active,
)
from src.tracking.hand_tracker import HandLandmarks, HandTracker
from src.utils.drawing import draw_debug_overlay

Frame = NDArray[np.uint8]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Priority when several two-hand detectors fire on the same frame
# (Ruby's pull-apart motion is a subset of Mirror's). Only one flag
# is ever reported ACTIVE so the overlay is unambiguous.
SPELL_PRIORITY = ("PORTAL", "RUBY", "MIRROR")


def arbitrate_spells(raw_active: dict[str, bool]) -> dict[str, bool]:
    """Keep only the highest-priority ACTIVE spell, turn the rest off."""
    for name in SPELL_PRIORITY:
        if raw_active.get(name, False):
            return {k: (k == name) for k in raw_active}
    return dict(raw_active)


class Pipeline:
    """Runs the whole hand-tracking -> spell-casting chain, one frame at a time."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        capture: Capture | None = None,
        tracker: HandTracker | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        """
        capture, tracker and clock can be injected (fakes in tests); by default they
        are created from the config. Raises RuntimeError if the camera can't open and
        FileNotFoundError if the model file is missing.
        """
        self.config = cfg = config or PipelineConfig()
        self._clock = clock

        self.capture = capture or Capture(
            cfg.camera_id, cfg.frame_width, cfg.frame_height, cfg.target_fps
        )
        try:
            self.tracker = tracker or HandTracker(
                cfg.model_path,
                cfg.max_hands,
                cfg.detection_confidence,
                cfg.tracking_confidence,
            )
        except Exception:
            self.capture.release()  # don't leave the camera locked if the tracker fails
            raise

        # Spell media, loaded once at startup (the video is decoded into memory here).
        # A missing file raises FileNotFoundError, like a missing model does.
        try:
            self.shield_video = VideoSpell(PROJECT_ROOT / cfg.shield_video)
            self.portal_image = ImagePortal(PROJECT_ROOT / cfg.portal_image)
        except Exception:
            self.capture.release()  # same cleanup as when the tracker fails
            self.tracker.close()
            raise
        self._portal_fade = Fader()
        self._portal_anchor = (0.0, 0.0, 0.0)  # x, y, height: kept so the fade-out has a place

        self.recognizer = GestureRecognizer(cfg.gesture_config)
        self.motion = MotionAnalyzer(cfg.motion_config)

        # Two-hand spells. Mirror, Ruby and Portal have no art yet: they are only detected.
        ctx_config = replace(cfg.gesture_config, together_max_ratio=cfg.together_max_ratio)
        self.context = ctx = TwoHandContext(ctx_config)
        spell_cfg = replace(cfg.gesture_config, stability=cfg.spell_stability)
        mirror_cfg = replace(cfg.gesture_config, stability=cfg.mirror_stability)
        self.detectors: dict[str, SpellDetector] = {
            "MIRROR": SpellDetector(lambda g, c: is_mirror(ctx), mirror_cfg),
            "RUBY": SpellDetector(lambda g, c: is_ruby(ctx), spell_cfg),
            "PORTAL": SpellDetector(lambda g, c: is_portal(ctx), spell_cfg),
        }

        # These need the real frame size, so they are built on the first frame.
        self._frame_size: tuple[int, int] | None = None
        self._geometry: GeometryCalculator | None = None
        self._renderer: SpellRenderer | None = None

        self.state = PipelineState()
        self.show_debug = cfg.show_debug
        self.running = False

        self._t0: float | None = None
        self._prev: float = 0.0
        self._frames = 0
        self._dts: deque[float] = deque(maxlen=30)
        self._tracker_failures = 0
        self._is_shut_down = False

    def step(self) -> Frame:
        """Process exactly one frame and return the output image (BGR).

        Raises TimeoutError if the camera gives no frame (it is really gone).
        """
        cfg = self.config

        frame = self.capture.read()
        if cfg.mirror_view:
            frame = cv2.flip(frame, 1)  # flip BEFORE tracking: the Left/Right labels follow it
        geometry_calc, renderer = self._ensure_sized(frame)

        now = self._clock()
        if self._t0 is None:
            self._t0 = self._prev = now
        dt = now - self._prev
        self._prev = now
        anim_time = now - self._t0
        if self._frames > 0:  # the first frame has no previous frame to measure against
            self._dts.append(dt)
        self._frames += 1
        fps = self._fps()

        # detection -> geometry -> stable hand ids
        hands = self._detect(frame)
        geometries = [geometry_calc.compute(h) for h in hands]
        handedness = [h.handedness for h in hands]
        ids = hand_ids_from_handedness(len(hands), handedness)  # 0 = Left, 1 = Right
        by_id = dict(zip(ids, geometries, strict=True))

        # gestures (stabilized) and motion, using the same ids
        states = self.recognizer.update(geometries, handedness)
        raw = dict(self.recognizer.raw)
        # Two-hand spells read the STABILIZED pose (not raw) so PEACE/OPEN_PALM
        # cannot flicker for one frame and reset the 5-frame confirmation.
        stable = {s.hand_id: s.gesture for s in states}
        motions: dict[int, MotionState] = {}
        for hid, hand, geo in zip(ids, hands, geometries, strict=True):
            tip = (float(hand.landmarks_px[INDEX_TIP][0]), float(hand.landmarks_px[INDEX_TIP][1]))
            motions[hid] = self.motion.update(
                hid, geo, dt, tip_px=tip, is_pointing=raw.get(hid) is Gesture.POINTING
            )
        self.motion.retain_only(by_id)  # forget hands that left the frame

        # two-hand spells: call every frame, even with no hands, so timers keep running
        self.context.observe(anim_time, by_id, stable, motions)
        raw_active = {
            name: det.update(geometries).state is PoseState.ACTIVE
            for name, det in self.detectors.items()
        }
        # Only one two-hand flag is ever ACTIVE (PORTAL > RUBY > MIRROR).
        active = arbitrate_spells(raw_active)

        # rendering: one shield per confirmed open palm, plus fading ghosts of lost hands.
        # A two-hand spell suppresses new shields so Ruby (both palms open)
        # does not draw 2 shields + a label at the same time. Clapping
        # (together_now, before any detector confirms) also suppresses shields.
        two_hand_on = any(active.values()) or self.context.together_now
        spells: list[tuple[SpellTransform, Any]] = []
        seen: set[int] = set()
        for g, geo in zip(states, geometries, strict=True):
            spell_id = SPELL_FOR.get(g.gesture)
            if spell_id is None:
                continue
            if two_hand_on and g.hand_id in by_id:
                # Suppress new shields while a two-hand spell owns the screen;
                # still step the shield fade down so it resumes cleanly afterwards.
                # Mark as seen so no ghost is spawned for it this frame.
                renderer.compute_transform(
                    geo,
                    to_active(GestureState(Gesture.NONE, 0.0, 0, g.hand_id)),
                    motions.get(g.hand_id),
                    anim_time,
                    hand_id=g.hand_id,
                    spell_id=spell_id,
                )
                seen.add(g.hand_id)
                continue
            transform = renderer.compute_transform(
                geo,
                to_active(g),
                motions.get(g.hand_id),
                anim_time,
                hand_id=g.hand_id,
                spell_id=spell_id,
            )
            seen.add(g.hand_id)
            spells.append((transform, g))

        spells += renderer.ghost_spells(seen)

        # shield: the video replaces the PNG rings (ghosts have the same (transform, _) shape)
        for transform, _ in spells:
            self.shield_video.draw(
                frame,
                transform.position_px,
                transform.scale * ASSET_RADIUS,
                0.0,  # use transform.rotation_rad if you want it to tilt with the hand
                anim_time,
                transform.opacity,
            )

        # portal: the door photo opens between the two hands and fades out when it ends.
        # Track the midpoint continuously (not only while ACTIVE) so the door is
        # already in place on the first ACTIVE frame instead of popping in.
        if len(geometries) == 2:
            a, b = geometries
            self._portal_anchor = (
                (a.palm_center_px[0] + b.palm_center_px[0]) / 2,
                (a.palm_center_px[1] + b.palm_center_px[1]) / 2,
                4.0 * (a.hand_size_px + b.hand_size_px) / 2,
            )
        portal_level = self._portal_fade.step(active["PORTAL"])
        px, py, portal_height = self._portal_anchor
        self.portal_image.draw(frame, (px, py), portal_height, portal_level)

        self.state = PipelineState(
            frame=frame,
            timestamp=now,
            dt=dt,
            animation_time=anim_time,
            hands=hands,
            hand_ids=ids,
            geometries=geometries,
            gestures=states,
            raw_gestures=raw,
            motions=motions,
            spells_active=active,
            together_now=self.context.together_now,
            mirror_metrics=mirror_metrics(self.context),
            fps_detection=fps,  # single loop: detection and render run at the same rate
            fps_render=fps,
        )

        if self.show_debug:
            draw_debug_overlay(frame, self.state)
        return frame

    def run(self, max_frames: int | None = None, display: bool = True) -> None:
        """Loop until quit (q / ESC / closed window), camera loss or max_frames.

        Always shuts down at the end. display=False runs without a window.
        """
        self.running = True
        window = self.config.window_name
        n = 0
        try:
            while self.running:
                try:
                    output = self.step()
                except TimeoutError as exc:
                    print(f"Camera lost: {exc}")
                    break
                n += 1

                if display:
                    cv2.imshow(window, output)
                    self.handle_key(cv2.waitKey(1) & 0xFF)
                    if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        break  # the user closed the window
                if max_frames is not None and n >= max_frames:
                    break
        finally:
            self.shutdown()

    def handle_key(self, key: int) -> None:
        """q / ESC = quit, s = screenshot, d = toggle the debug overlay."""
        if key in (ord("q"), 27):
            self.running = False
        elif key == ord("d"):
            self.show_debug = not self.show_debug
        elif key == ord("s") and self.state.frame is not None:
            print(f"Screenshot saved: {self.save_screenshot(self.state.frame)}")

    def save_screenshot(self, frame: Frame) -> Path:
        folder = Path(self.config.screenshot_dir)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"spell_{datetime.now():%Y%m%d_%H%M%S}.png"
        cv2.imwrite(str(path), frame)
        return path

    def shutdown(self) -> None:
        """Release the camera, close the tracker and the windows. Safe to call twice."""
        if self._is_shut_down:
            return
        self._is_shut_down = True
        self.running = False
        try:
            self.capture.release()
        finally:
            try:
                self.tracker.close()
            finally:
                cv2.destroyAllWindows()

    def __enter__(self) -> Pipeline:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.shutdown()

    def _ensure_sized(self, frame: Frame) -> tuple[GeometryCalculator, SpellRenderer]:
        """Build the size-dependent parts on the first frame (and again if the size changes)."""
        h, w = frame.shape[:2]
        if self._frame_size != (h, w) or self._geometry is None or self._renderer is None:
            cfg = self.config
            self._geometry = GeometryCalculator(
                w, h, GeometryConfig(cfg.palm_center_strategy, cfg.hand_size_method)
            )
            self._renderer = SpellRenderer(cfg.spells, (h, w))
            self._frame_size = (h, w)
        return self._geometry, self._renderer

    def _detect(self, frame: Frame) -> list[HandLandmarks]:
        """Run the tracker. A failing frame counts as 'no hands'; too many in a row is fatal."""
        try:
            hands = self.tracker.process(frame)
        except Exception as exc:
            self._tracker_failures += 1
            if self._tracker_failures >= self.config.max_tracker_failures:
                raise RuntimeError(
                    f"Hand tracker failed {self._tracker_failures} frames in a row"
                ) from exc
            return []
        self._tracker_failures = 0
        return hands

    def _fps(self) -> float:
        """Rolling-average FPS over the last 30 frames (0.0 until it can be measured)."""
        total = sum(self._dts)
        return len(self._dts) / total if total > 0 else 0.0