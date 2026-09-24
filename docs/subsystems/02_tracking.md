# Subsystem: Hand Tracking (MediaPipe Integration)

## Responsibility
Detect hands, extract 21 landmarks per hand, provide normalized + pixel coordinates, handle multi-hand.

## Interface
```python
@dataclass
class HandLandmarks:
    landmarks_norm: np.ndarray  # shape (21, 3) — x, y, z
    landmarks_px: np.ndarray    # shape (21, 2) — x, y
    handedness: str             # "Left" or "Right"
    score: float                # Detection confidence [0,1]

class HandTracker:
    def __init__(
        self,
        max_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1
    ):
        ...
    
    def process(self, frame_bgr: np.ndarray) -> list[HandLandmarks]:
        ...
    
    def close(self):
        ...
```

## MediaPipe Landmark Indices (21 points)
```
0:  WRIST
1:  THUMB_CMC        5:  INDEX_FINGER_MCP      9:  MIDDLE_FINGER_MCP
2:  THUMB_MCP        6:  INDEX_FINGER_PIP     10: MIDDLE_FINGER_PIP
3:  THUMB_IP         7:  INDEX_FINGER_DIP     11: MIDDLE_FINGER_DIP
4:  THUMB_TIP        8:  INDEX_FINGER_TIP     12: MIDDLE_FINGER_TIP
                                    13: RING_FINGER_MCP
                                    14: RING_FINGER_PIP
                                    15: RING_FINGER_DIP
                                    16: RING_FINGER_TIP
                                    17: PINKY_MCP
                                    18: PINKY_PIP
                                    19: PINKY_DIP
                                    20: PINKY_TIP
```

## Requirements
- [ ] Initialize MediaPipe Hands with configurable params
- [ ] Process BGR frame → RGB → MediaPipe → results
- [ ] Convert normalized landmarks to pixel coordinates
- [ ] Return empty list if no hands detected (not None)
- [ ] Maintain hand identity across frames (MediaPipe tracking)
- [ ] Handle 1 and 2 hands simultaneously
- [ ] Expose handedness (Left/Right) for each detection
- [ ] Resource cleanup on close

## Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_hands` | 2 | Max simultaneous hands |
| `min_detection_confidence` | 0.7 | Detection threshold |
| `min_tracking_confidence` | 0.5 | Tracking threshold |
| `model_complexity` | 1 | 0=fast/lite, 1=accurate |

## Performance Targets
- Process time: <15ms/frame (model_complexity=1)
- Memory: <50MB
- Stable 30 FPS on 640x480

## Testing
- Unit: Mock MediaPipe results → test coordinate conversion, handedness parsing
- Integration: Test on sample images with known landmarks
- Property: Landmark count = 21 per hand, coordinates in bounds

## Future Extensions
- Hand bounding box extraction
- 3D world landmarks (MediaPipe provides)
- Gesture classification model (replace rule-based)