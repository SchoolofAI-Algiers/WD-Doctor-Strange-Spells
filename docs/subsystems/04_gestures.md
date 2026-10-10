# Subsystem: Gesture Recognition

## Responsibility
Classify hand pose into discrete gesture states using rule-based logic on normalized geometry, with temporal stability via state machine.

## Interface
```python
from enum import Enum

class Gesture(Enum):
    NONE = "none"
    OPEN_PALM = "open_palm"
    CLOSED_FIST = "closed_fist"
    POINTING = "pointing"
    PEACE = "peace"
    TWO_HANDS_TOGETHER = "two_hands_together"   
    THUMB_MIDDLE_PINCH = "thumb_middle_pinch"

@dataclass
class GestureState:
    gesture: Gesture
    confidence: float
    frames_held: int
    hand_id: int            # Left = 0, Right = 1

class GestureRecognizer:
    def __init__(self, config: GestureConfig | None = None):
        ...

    def update(self, geometries: list[HandGeometry],
               handedness: list[str] | None = None) -> list[GestureState]:
        ...

    def reset(self):
        ...

    raw: dict[int, Gesture]   # unstabilised pose of each hand for the current frame
```

`handedness` is optional: one label (`"Left"` / `"Right"`) per geometry, in the same order, as given by the tracker.
It makes `hand_id` follow the physical hand instead of the list order. If it is missing, has the wrong length,
or contains an unknown or duplicated label, ids fall back to list positions (`0, 1`).

## How It Works

Every frame, for each detected hand:

1. `classify_hand` turns the geometry into a raw `Gesture` (rules below).
2. One `PoseStabilizer` per (hand, gesture) confirms it over several frames, so a flickering detection
   never becomes a gesture.
3. `GestureRecognizer.update` returns the confirmed gesture of each hand as a `GestureState`.

Two-hand spells reuse the same state machine: a predicate says "the pose is there this frame", and a
`SpellDetector` confirms it over time. Spells that depend on movement or on what happened a moment ago
(Mirror, Ruby, Portal) read a small history kept by `TwoHandContext`.

The module only reads `HandGeometry` (never modifies it) and is pure logic: `numpy` only, no OpenCV, no MediaPipe.

## Architecture

```
HandGeometry ──► classify_hand ──► raw pose ──► PoseStabilizer ──► GestureState (confirmed)
                                       │
                                       ▼
          TwoHandContext (+ MotionState) ──► is_mirror / is_ruby / is_portal ──┐
          both hands' geometries ──────────► is_shield / is_ikkon ─────────────┤
                                                                               ▼
                                                      SpellDetector ──► IDLE / CANDIDATE / ACTIVE
```

Everything lives in `src/gestures/recognizer.py`.

| Class / function | Role |
|------------------|------|
| `GestureConfig`, `StabilityConfig` | thresholds and frame counts (see Parameters) |
| `GestureState`, `PoseStatus` | outputs: confirmed gesture of a hand / state of a stabilised pose |
| `extended_fingers(landmarks_norm, config)` | set of extended finger names |
| `is_thumb_middle_pinch(geometry, config)` | thumb tip touching middle tip |
| `classify_hand(geometry, config)` | one hand to one raw `Gesture` |
| `PoseStabilizer.update(pose_seen)` | IDLE / CANDIDATE / ACTIVE state machine, returns `PoseStatus` |
| `hand_ids_from_handedness(count, handedness)` | Left to 0, Right to 1, positional fallback |
| `GestureRecognizer` | classify + stabilise every hand, keeps `raw` |
| `hands_together`, `is_shield`, `is_ikkon` | two-hand predicates on geometries |
| `SpellDetector(predicate)` | stabilises any predicate `(geometries, config) -> bool` |
| `TwoHandContext.observe(t, geom, pose, motion)` | per-frame history (clap, Mirror) for context spells |
| `is_mirror`, `is_ruby`, `is_portal`, `mirror_metrics` | context spells; `mirror_metrics` gives the numbers shown for debugging |

## Rule-Based Classification

### Finger Extension
```
extended = distance(tip, mcp) > threshold * distance(pip, mcp)
```
| Finger | Tip | PIP | MCP | Threshold |
|--------|-----|-----|-----|-----------|
| Thumb | 4 | 3 | 2 | 1.3 |
| Index | 8 | 6 | 5 | 1.5 |
| Middle | 12 | 10 | 9 | 1.5 |
| Ring | 16 | 14 | 13 | 1.5 |
| Pinky | 20 | 18 | 17 | 1.5 |

The thumb is still computed by `extended_fingers`, but the pose rules below ignore it because its test is unreliable.

### Gesture Rules (Priority Order)
0. **Size guard**: `hand_size_norm < min_hand_size_norm` gives `NONE`. A lost or invented hand collapses to
   almost one point, every finger then reads "not extended", and it would be classified as a fist.
1. **CLOSED_FIST**: none of index, middle, ring, pinky extended
2. **THUMB_MIDDLE_PINCH**: index, ring, pinky extended + thumb tip touching middle tip
3. **OPEN_PALM**: index, middle, ring, pinky extended (palm facing the camera is not checked yet)
4. **POINTING**: only index extended
5. **PEACE**: index + middle extended, others not
6. **NONE**: No rule matches

Pinch test: `distance(lm[4], lm[12]) / distance(lm[0], lm[9]) < pinch_max_ratio`. The scale is the palm length
(wrist to middle MCP), not `hand_size_px`, because the hand size shrinks when the middle finger curls.

### Two-Hand Gestures
- **TWO_HANDS_TOGETHER**: `hands_together`: exactly 2 hands and palm distance / mean hand size < `together_max_ratio`.
  It is a predicate used by the spells, not a per-hand `Gesture` (the pose of a hand is never overwritten by it).

| Spell | Trigger | Function |
|-------|---------|----------|
| Shield of the Seraphim | both open palms, or both fists | `is_shield` |
| Images of Ikkon | both hands in thumb-middle pinch | `is_ikkon` |
| Mirror Dimension | TOGETHER confirmed, then hands pull apart horizontally | `is_mirror` |
| Ruby Rings | clap, then both open palms within 1 s | `is_ruby` |
| Dr Strange Portal | one hand `PEACE` and still, the other moving fast | `is_portal` |

Rules of the three context spells (speeds are in hand-sizes per second, so they do not depend on the distance to the camera):
- **Mirror**: within 0.8 s of a confirmed TOGETHER, palms separate faster than 0.8, the two x velocities have opposite signs, and `abs(vy) < 0.6 * abs(vx)` for both hands.
- **Ruby**: hands no longer together, clap less than 1.0 s ago, no Mirror in the last 1.5 s (both end with open hands), both raw poses `OPEN_PALM`.
- **Portal**: one hand `PEACE` and stationary, the other with `speed / hand_size_px > 1.0` (rotation is not checked).

`TwoHandContext.observe` must be called every frame, even with no hand, with `geom = {id: HandGeometry}`,
`pose = recognizer.raw` and `motion = {id: MotionState}` (fields read: `velocity_px_s`, `speed_px_s`, `is_stationary`).

## Temporal Stability (State Machine)

### State Transitions
```
IDLE →(N frames)→ CANDIDATE →(M frames)→ ACTIVE →(K frames lost)→ IDLE
```
- Before ACTIVE, any missed frame resets the count: a pose seen one frame out of two never activates.
- Once ACTIVE, short dropouts are tolerated (less than `exit_frames` in a row).
- Confidence is a moving average of the detections. Total latency: about 8 frames (0.27 s at 30 FPS).

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `enter_frames` | 3 | Frames to enter CANDIDATE |
| `confirm_frames` | 5 | Frames to confirm ACTIVE |
| `exit_frames` | 3 | Frames lost to exit ACTIVE |
| `ema_alpha` | 0.3 | Confidence smoothing |

Other thresholds (`GestureConfig` and module constants):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_hand_size_norm` | 0.05 | Smaller hands are `NONE` |
| `thumb_threshold` / `finger_threshold` | 1.3 / 1.5 | Extension ratios |
| `together_max_ratio` | 1.0 | Max palm distance / hand size to be together |
| `pinch_max_ratio` | 0.3 | Max thumb-middle distance / palm length for a pinch |
| `MIRROR_WINDOW` / `MIRROR_MIN_APART` | 0.8 s / 0.8 | Mirror timing and speed |
| `RUBY_CLAP_WINDOW` / `RUBY_MIRROR_COOLDOWN` | 1.0 s / 1.5 s | Ruby timing |
| `PORTAL_MIN_SPEED` | 1.0 | Portal speed of the moving hand |

## Requirements
- [x] Rule-based classifier for single-hand gestures
- [x] Two-hand gesture detection (distance-based)
- [x] State machine per hand (track by hand_id)
- [x] Configurable thresholds and frame counts
- [x] Reset state when hand lost
- [x] Return confidence + frames_held for rendering decisions
- [x] Pure logic — no OpenCV, no MediaPipe

## Reproducibility

Requirements: Python 3.10+, `numpy`. Dev tools: `pytest`, `ruff`, `mypy`.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

ruff format . && ruff check .      # style and lint (make lint)
mypy src/                          # types (make typecheck)
pytest tests/unit/test_gestures.py -v   # gesture tests, no camera needed (make test)
```

Run only some tests with `-k`, for example `pytest tests/unit/test_gestures.py -k flicker -v`.

The live demo (`python DEMO.py`) is documented with the rendering subsystem.

## Testing
Tests use synthetic landmarks, so they need no camera and no MediaPipe. Motion is faked with a small
`FakeMotion` stand-in for `MotionState`, so the context spells need no motion analyzer either.
Everything lives in `tests/unit/test_gestures.py` (233 tests).
- Unit: Synthetic geometries → expected gestures (every gesture, thumb ignored, scale and rotation invariance)
- Unit: Ghost hand: a tiny hand is `NONE` (strict size boundary) and would be a fist without the size guard;
  it never reaches a confirmed gesture through the recognizer or a `SpellDetector`
- Unit: State machine transitions, dropout tolerance, confidence, reset
- Unit: Hand ids from handedness and every fallback; ids follow the hand when the list order swaps;
  a lost hand resets only that hand
- Unit: `hands_together`, Shield and Ikkon (directly and through `SpellDetector`)
- Unit: `TwoHandContext`: initial state, `both()`, `together_now`, `together_seen_t` (one close frame),
  `together_active_t` (needs `enter_frames + confirm_frames` frames), `mirror_t`
- Unit: `mirror_metrics`: apart speed, opposite directions, horizontal check, scaling with hand size,
  degenerate cases (missing hand, zero size, coincident palms)
- Unit: Mirror, Ruby, Portal: each condition on its own, the boundaries (`MIRROR_WINDOW` inclusive; speed
  and clap-window thresholds strict; Ruby cooldown after a Mirror), and how they interact (Ruby does not
  fire on the pull-apart of a Mirror, a quick clap is a Ruby only)
- Property: Flicker resistance (an alternating pose over 200 frames never activates), checked on the
  stabilizer, the recognizer and the `SpellDetector`
- Not tested yet: `GestureRecognizer.raw` (only used indirectly by the demo); Portal rotation is not
  checked by the code either