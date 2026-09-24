# Subsystem: Pipeline Orchestration

## Responsibility
Coordinate all subsystems, manage state, handle timing, separate detection from rendering loops for performance.

## Architecture

### Dual-Loop Design (Recommended)
```
MAIN THREAD
  Capture (30 FPS) → Tracking (30 FPS) → Geometry (30 FPS)
                           ↓
                    STATE BUFFER (Thread-Safe)
                           ↓
  RENDER LOOP (60 FPS)
    Read state → Interpolate → Render → Display
```

### Single-Loop Alternative (Simpler, for MVP)
```python
while running:
    frame = capture.read()
    hands = tracker.process(frame)
    geometries = [geometry.compute(h) for h in hands]
    gestures = recognizer.update(geometries)
    motions = [motion.update(id, g, dt) for id, g in enumerate(geometries)]
    transforms = [renderer.compute_transform(g, gs, m, t) for g, gs, m in zip(...)]
    output = renderer.render(frame, transforms, gestures, t)
    cv2.imshow(output)
```

## State Management

### Pipeline State
```python
@dataclass
class PipelineState:
    frame: np.ndarray
    timestamp: float
    dt: float
    hands: list[HandLandmarks]
    geometries: list[HandGeometry]
    gestures: list[GestureState]
    motions: list[MotionState]
    animation_time: float
    fps_detection: float
    fps_render: float
```

### Hand Identity Tracking
- MediaPipe provides tracking IDs
- Map tracker ID → internal hand_id (0, 1, ...)
- Handle hand loss/gain gracefully

## Interface
```python
class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.capture = Capture(...)
        self.tracker = HandTracker(...)
        self.geometry = GeometryCalculator(...)
        self.gestures = GestureRecognizer(...)
        self.motion = MotionAnalyzer(...)
        self.renderer = SpellRenderer(...)
        self.state = PipelineState()
    
    def run(self):
        ...
    
    def step(self) -> np.ndarray:
        ...
    
    def shutdown(self):
        ...
```

## Configuration
```python
@dataclass
class PipelineConfig:
    camera_id: int = 0
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30
    max_hands: int = 2
    detection_confidence: float = 0.7
    tracking_confidence: float = 0.5
    palm_center_strategy: str = "five_point"
    gesture_config: GestureConfig = field(default_factory=GestureConfig)
    motion_config: MotionConfig = field(default_factory=MotionConfig)
    spell_config: SpellConfig = field(default_factory=SpellConfig)
    dual_loop: bool = False
```

## Requirements
- [ ] Initialize all subsystems with config
- [ ] Single-loop MVP implementation
- [ ] Frame timing (measure dt, track FPS)
- [ ] Graceful shutdown (release camera, close windows)
- [ ] Keyboard controls: `q`=quit, `s`=screenshot, `d`=toggle debug draw
- [ ] Debug overlay: landmarks, palm center, gesture labels, FPS
- [ ] Error handling: camera disconnect, tracker failure

## Performance Budget (640×480 @ 30fps = 33ms/frame)

| Stage | Target |
|-------|--------|
| Capture | <5ms |
| MediaPipe | <15ms |
| Geometry | <0.5ms |
| Gestures | <0.2ms |
| Motion | <0.1ms |
| Render | <2ms |
| **Total** | **<23ms** |

## Future Extensions
- Dual-thread with lock-free ring buffer
- Async capture + processing + render threads
- Dynamic resolution scaling based on FPS
- Headless mode for CI/testing
- Plugin system for custom gesture→spell mappings