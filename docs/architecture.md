# System Architecture

## Overview

Real-time hand-tracking → spell-casting pipeline with clear separation between **Computer Vision** (detection, geometry, gestures) and **Visual Effects** (rendering, compositing).

## Data Flow

```
Webcam (640×480 @ 30fps, BGR)
    │
    ▼
Capture → frame: np.ndarray (H, W, 3)
    │
    ▼
HandTracker (MediaPipe) → list[HandLandmarks] (21 pts each, norm + px)
    │
    ▼
GeometryCalc → list[HandGeometry]
    • palm_center_px, hand_size_px
    • palm_angle_rad
    • landmarks_norm (wrist-relative, scale-invariant)
    │
    ├──▶ GestureRecognizer → list[GestureState]
    │     (Rule-based + State Machine + Temporal Smoothing)
    │
    └──▶ MotionAnalyzer → list[MotionState]
          (Velocity, Trajectory, Spell Drawing Paths)
    │
    ▼
PIPELINE STATE BUFFER (Shared between Detection & Render)
    │
    ▼
SpellRenderer
    • compute_transform() → SpellTransform (pos, scale, rot, opacity)
    • render() → Composited Frame (BGR)
    │
    ▼
Display (cv2.imshow)
```

## Subsystem Boundaries

| Layer                  | Subsystems                        | Responsibility                                | Rate      |
| ---------------------- | --------------------------------- | --------------------------------------------- | --------- |
| **Input**        | Capture                           | Frame acquisition                             | 30 Hz     |
| **CV Core**      | HandTracker                       | Landmark detection                            | 30 Hz     |
| **CV Features**  | GeometryCalc                      | Palm center, size, orientation, normalization | 30 Hz     |
| **CV Semantics** | GestureRecognizer, MotionAnalyzer | Discrete states + continuous motion           | 30 Hz     |
| **State**        | Pipeline Buffer                   | Thread-safe shared state                      | —        |
| **VFX**          | SpellRenderer                     | Transform calc + Alpha compositing            | 30–60 Hz |

## Key Design Decisions

### 1. Detection ⟂ Rendering Separation

- **Why**: Detection (MediaPipe) is heavy; rendering is light but needs high, consistent FPS for smooth animation
- **How**: Shared state buffer with latest detection results; render loop interpolates

### 2. Geometry as Pure Functions

- **Why**: Testable, deterministic, no hidden state
- **How**: `HandGeometry = f(HandLandmarks, image_dims)` — no side effects

### 3. Gesture = State Machine, Not Frame Classifier

- **Why**: Prevents flicker from landmark noise
- **How**: CANDIDATE → ACTIVE → NONE with frame counters + EMA confidence

### 4. Spell as Transform + Asset, Not Hardcoded Draw Calls

- **Why**: Enables multiple spell types, animation, composition
- **How**: `SpellTransform` computed from geometry + gesture + motion + time

### 5. Normalized Landmarks for ML-Readiness

- **Why**: Future swap to ML classifier
- **How**: Wrist-relative, hand-size-normalized coordinates in `HandGeometry`

## Coordinate Systems

| Space                   | Origin   | Axes                          | Range               | Used By                            |
| ----------------------- | -------- | ----------------------------- | ------------------- | ---------------------------------- |
| **Image**         | Top-left | X→right, Y→down             | [0, W) × [0, H)    | Capture, Renderer                  |
| **Normalized**    | Top-left | X→right, Y→down             | [0, 1] × [0, 1]    | MediaPipe output                   |
| **Hand-Relative** | Wrist    | X→right, Y→down             | ~[-1, 1] × [-1, 1] | Geometry, Gestures                 |
| **World (3D)**    | Camera   | X→right, Y→down, Z→forward | Meters              | Future (MediaPipe world landmarks) |

**Conversion**: Always explicit. `GeometryCalculator` owns norm↔px conversion.

## Threading Model (Future)

```
Thread 1 (Capture + CV):     30 Hz fixed
  └─ Capture → Tracker → Geometry → Gestures → Motion → State Buffer

Thread 2 (Render):           60 Hz (vsync)
  └─ Read State Buffer → Interpolate → Render → Display
```

**Synchronization**: Lock-free ring buffer or `queue.Queue(maxsize=1)` for latest state.

## Performance Budget (640×480 @ 30fps = 33ms/frame)

| Stage           | Target          | Notes                       |
| --------------- | --------------- | --------------------------- |
| Capture         | <5ms            | `cv2.VideoCapture.read()` |
| MediaPipe       | <15ms           | Model complexity 1          |
| Geometry        | <0.5ms          | Pure NumPy                  |
| Gestures        | <0.2ms          | Rule eval + state machine   |
| Motion          | <0.1ms          | Simple math                 |
| Render          | <2ms            | Alpha blend 2 spells        |
| **Total** | **<23ms** | **Headroom: 10ms**    |

## Extension Points

| Extension             | Integration Point                                    |
| --------------------- | ---------------------------------------------------- |
| ML Gesture Classifier | Replace`GestureRecognizer` (same interface)        |
| 3D Hand Pose          | Use MediaPipe world landmarks in`GeometryCalc`     |
| Particle Effects      | Add`ParticleSystem` to `SpellRenderer.render()`  |
| Glow/Bloom            | Post-process pass in`SpellRenderer`                |
| Multi-Spell Types     | `SpellConfig` per gesture in `GestureRecognizer` |
| GPU Rendering         | Swap`SpellRenderer` backend (OpenGL/Vulkan)        |
| Headless/CI           | `Pipeline.run(headless=True)` returns frames       |

## Error Handling

| Failure Mode          | Handling                                   |
| --------------------- | ------------------------------------------ |
| Camera disconnect     | Retry with backoff, show error overlay     |
| MediaPipe init fail   | Clear error message, exit gracefully       |
| Asset load fail       | Fallback to procedural circle, log warning |
| Hand lost mid-gesture | State machine handles via`exit_frames`   |
| FPS drop              | Log warning, optionally reduce resolution  |

## Configuration Management

- All tunable params in `PipelineConfig` (dataclass)
- Load from YAML/JSON for experiments (future)
- CLI args override config (future)

## Testing Strategy

| Level       | Scope                                                   | Tools                             |
| ----------- | ------------------------------------------------------- | --------------------------------- |
| Unit        | Pure functions (geometry, gestures, motion, transforms) | `pytest`, synthetic data        |
| Integration | Subsystem pairs (tracker→geometry, geometry→gestures) | `pytest`, recorded landmarks    |
| System      | Full pipeline on test video                             | `pytest`, reference frames      |
| Visual      | Manual verification                                     | `cv2.imshow`, screen capture    |
| Perf        | FPS, latency, memory                                    | `cProfile`, `memory_profiler` |

## Directory Structure (src/)

```
src/
├── capture/
│   ├── __init__.py
│   └── capture.py
├── tracking/
│   ├── __init__.py
│   └── hand_tracker.py
├── geometry/
│   ├── __init__.py
│   ├── calculator.py
│   └── normalization.py
├── gestures/
│   ├── __init__.py
│   ├── classifier.py
│   └── state_machine.py
├── motion/
│   ├── __init__.py
│   └── analyzer.py
├── rendering/
│   ├── __init__.py
│   ├── spell_renderer.py
│   ├── compositing.py
│   └── assets.py
├── pipeline/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── config.py
│   └── state.py
├── utils/
│   ├── __init__.py
│   ├── math.py
│   └── drawing.py
└── main.py
```
