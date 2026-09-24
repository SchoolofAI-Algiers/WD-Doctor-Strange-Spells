# WD-Doctor-Strange-Spells

**Dr. Strange Spells: Real-Time Hand Tracking and Augmented-Reality Spell Casting**
School of AI Algiers | Technical Department | Welcome Day Technical Project
Project Mentor: Rayan Derradji

---

## Project Overview

A real-time computer vision pipeline that tracks hand landmarks via webcam and renders animated "magic circle" spell effects that follow hand geometry, orientation, and gesturesm creating the illusion of casting Doctor Strange-style spells.

**Core Philosophy**: The visual effects are the output layer. The real project is the pipeline:

```
Pixels → Landmarks → Geometry → Motion → Semantic State → Visual Effect
```

We intentionally **do not** copy the reference implementation's code. We rebuild the CV pipeline ourselves, using only the visual assets (MIT licensed) from [YerraRahul23/Dr.Strange-Magic-Effect](https://github.com/YerraRahul23/Dr.Strange-Magic-Effect).

---

## System Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│  Capture    │──▶│  Hand Tracker│───▶│  Geometry   │──▶│  Gesture     │
│  (Webcam)   │    │  (MediaPipe) │    │  Calculator │    │  Recognizer  │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                                                                    │
                                                                    ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│  Composite  │◀───│  Spell      │◀───│  Transform │◀───│  Motion      │
│  & Render   │    │  Renderer    │    │  Calculator │    │  Analyzer    │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
```

**Key Separation**: Detection pipeline (CV) → State/Geometry → Rendering pipeline (VFX) — run at independent rates if needed.

---

## Subsystems

| Subsystem                        | Description                                                             | Status     |
| -------------------------------- | ----------------------------------------------------------------------- | ---------- |
| **Capture**                | Webcam acquisition, frame handling, FPS measurement                     | 📋 Planned |
| **Hand Tracking**          | MediaPipe integration, 21-landmark extraction, multi-hand support       | 📋 Planned |
| **Geometry**               | Palm center, hand size, palm orientation (2D), coordinate normalization | 📋 Planned |
| **Gesture Recognition**    | Rule-based classifiers, state machine, temporal smoothing               | 📋 Planned |
| **Motion Analysis**        | Velocity, trajectory, acceleration, rotation velocity, spell drawing    | 📋 Planned |
| **Spell Rendering**        | Alpha compositing, dual rotation (hand + animation), scaling, animation | 📋 Planned |
| **Pipeline Orchestration** | State management, detection/render separation, performance optimization | 📋 Planned |

Detailed breakdowns: [`docs/subsystems/`](docs/subsystems/)

---

## Assets

Located in `assets/spells/`:

| Type                    | Files                                              | Description         |
| ----------------------- | -------------------------------------------------- | ------------------- |
| **Inner Circles** | `blue.png`, `light_orange.png`, `orange.png` | Inner rotating ring |
| **Outer Circles** | `dark_red.png`, `orange.png`, `red.png`      | Outer rotating ring |

**License**: MIT — Copyright (c) 2025 Yerra Rahul
**Attribution**: See [`ASSET_ATTRIBUTION.md`](ASSET_ATTRIBUTION.md)

---

## Learning Objectives

### Computer Vision

- Webcam acquisition, image coordinate systems
- Hand detection, landmark tracking, bounding boxes
- Geometric measurements, perspective, rotation, scaling
- Image compositing, alpha blending

### Machine Learning / AI

- Landmark detector internals
- Detection vs tracking vs classification
- Feature engineering, rule-based gesture recognition
- Temporal classification, path to ML classifiers

### Software Engineering

- Real-time pipeline design
- Separating detection from rendering
- State management, reusable components
- Performance optimization, Git collaboration, testing

### Graphics

- Transparent overlays, rotation, scaling
- Coordinate transformations, animation
- Particle effects, glow/bloom, layer compositing

---

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run (once implemented)
python -m src.main
```

---

## Development Workflow

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for:

- Branch naming conventions
- Commit message format (Conventional Commits)
- Pull request process
- Code review guidelines
- Testing requirements

---

## Project Structure

```
WD-Doctor-Strange-Spells/
├── assets/
│   └── spells/
│       ├── inner/          # Inner circle assets (PNG with alpha)
│       └── outer/          # Outer circle assets (PNG with alpha)
├── docs/
│   ├── subsystems/         # Detailed subsystem specifications
│   └── architecture.md     # System architecture document
├── src/                    # Source code (to be created)
│   ├── capture/            # Webcam handling
│   ├── tracking/           # MediaPipe hand tracking
│   ├── geometry/           # Palm center, size, orientation
│   ├── gestures/           # Recognition + state machine
│   ├── motion/             # Velocity, trajectory, drawing
│   ├── rendering/          # Spell rendering + compositing
│   ├── pipeline/           # Orchestration + state
│   └── main.py             # Entry point
├── tests/                  # Unit + integration tests
├── requirements.txt
├── CONTRIBUTING.md
├── ASSET_ATTRIBUTION.md
└── README.md
```

---

## Milestones

1. **M1 — Raw Hand Tracking**: Webcam + MediaPipe + landmark drawing + FPS (1 hand → 2 hands)
2. **M2 — Geometry**: Palm center, hand size, orientation → spell follows hand
3. **M3 — Rendering**: Alpha-blended rotating circles, scale with hand distance
4. **M4 — Gestures**: Rule-based state machine with temporal stability
5. **M5 — Motion**: Velocity, trajectories, spell drawing in air
6. **M6 — Polish**: Particles, glow, bloom, multi-spell types, performance tuning

---

## References

- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)
- [OpenCV Alpha Blending](https://docs.opencv.org/4.x/df/dfb/group__core__array.html#ga4e36d0eda0828c5f0a825d8d0b8e8c4c)
- Reference Implementation: [YerraRahul23/Dr.Strange-Magic-Effect](https://github.com/YerraRahul23/Dr.Strange-Magic-Effect) (assets only)
- Project Documentation: [`Docs/Dr.Strange Spells Project Documentation.pdf`](<Docs/Dr.Strange%20Spells%20Project%20Documentation.pdf>)

---

## License

This project's code: **TBD**
Assets: **MIT License** — Copyright (c) 2025 Yerra Rahul
