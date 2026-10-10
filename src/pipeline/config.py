"""Configuration of the pipeline: every tunable value lives here, none in the loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.gestures.recognizer import GestureConfig, StabilityConfig
from src.motion.analyzer import MotionConfig
from src.rendering.spell_renderer import SpellConfig
from src.rendering.spells import make_spells


@dataclass
class PipelineConfig:
    # camera
    camera_id: int = 0 #default camera
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30
    mirror_view: bool = True  # selfie view: your right hand = right side of the screen

    # hand tracking
    model_path: str | Path | None = None 
    max_hands: int = 2 #max of hands that can be detected 
    detection_confidence: float = 0.7
    tracking_confidence: float = 0.5
    max_tracker_failures: int = 30  # consecutive tracker errors tolerated before giving up

    # geometry
    palm_center_strategy: str = "five_point"
    hand_size_method: str = "wrist_middle_tip"

    # gestures / motion / rendering
    gesture_config: GestureConfig = field(default_factory=GestureConfig) #this is the configuration for the gesture recognizer
    motion_config: MotionConfig = field(default_factory=MotionConfig) #this is the configuration for the motion analyzer
    spells: dict[str, SpellConfig] = field(default_factory=make_spells)

    # spell media (paths relative to the project root, or absolute)
    shield_video: str = "assets/spells/DR.STRANGE Shield.mp4"  # drawn on every confirmed open palm
    portal_image: str = "assets/spells/school_of_ai_door.jpg"  # shown between the hands on PORTAL

    # two-hand spells (Mirror, Ruby, Portal)
    together_max_ratio: float = 2.5  # "together" = palm centers within this many hand sizes
    spell_stability: StabilityConfig = field(
        default_factory=lambda: StabilityConfig(enter_frames=2, confirm_frames=3, exit_frames=3)
    )
    # Mirror is a short burst: turns on fast, stays on ~8 frames after
    mirror_stability: StabilityConfig = field(
        default_factory=lambda: StabilityConfig(enter_frames=1, confirm_frames=1, exit_frames=8)
    )

    # display
    show_debug: bool = True
    window_name: str = "Dr. Strange Spells"
    screenshot_dir: str = "screenshots"

    # future: render at 60 FPS in its own thread 
    dual_loop: bool = False