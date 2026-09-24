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
    THUMBS_UP = "thumbs_up"
    TWO_HANDS_TOGETHER = "two_hands_together"

@dataclass
class GestureState:
    gesture: Gesture
    confidence: float
    frames_held: int
    hand_id: int

class GestureRecognizer:
    def __init__(self, config: GestureConfig):
        ...
    
    def update(self, geometries: list[HandGeometry]) -> list[GestureState]:
        ...
    
    def reset(self):
        ...
```

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

### Gesture Rules (Priority Order)
1. **CLOSED_FIST**: All 5 fingers NOT extended
2. **OPEN_PALM**: All 5 fingers extended + palm facing camera
3. **POINTING**: Only index extended
4. **PEACE**: Index + middle extended, others not
5. **THUMBS_UP**: Only thumb extended
6. **NONE**: No rule matches

### Two-Hand Gestures
- **TWO_HANDS_TOGETHER**: Both hands detected, palm centers < threshold distance

## Temporal Stability (State Machine)

### State Transitions
```
NONE →(N frames)→ CANDIDATE →(M frames)→ ACTIVE →(K frames lost)→ NONE
```

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `enter_frames` | 3 | Frames to enter CANDIDATE |
| `confirm_frames` | 5 | Frames to confirm ACTIVE |
| `exit_frames` | 3 | Frames lost to exit ACTIVE |
| `ema_alpha` | 0.3 | Confidence smoothing |

## Requirements
- [ ] Rule-based classifier for single-hand gestures
- [ ] Two-hand gesture detection (distance-based)
- [ ] State machine per hand (track by hand_id)
- [ ] Configurable thresholds and frame counts
- [ ] Reset state when hand lost
- [ ] Return confidence + frames_held for rendering decisions
- [ ] Pure logic — no OpenCV, no MediaPipe

## Testing
- Unit: Synthetic geometries → expected gestures
- Unit: State machine transitions
- Property: Flicker resistance