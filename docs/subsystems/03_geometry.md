# Subsystem: Geometry Calculator

## Responsibility
Compute palm center, hand size, palm orientation, and normalized landmarks from raw hand landmarks.

## Interface
```python
@dataclass
class HandGeometry:
    palm_center_px: tuple[float, float]
    palm_center_norm: tuple[float, float]
    hand_size_px: float
    hand_size_norm: float
    palm_angle_rad: float
    landmarks_norm: np.ndarray

class GeometryCalculator:
    def __init__(self, image_width: int, image_height: int):
        ...
    
    def compute(self, hand: HandLandmarks) -> HandGeometry:
        ...
```

## Calculations

### Palm Center (Configurable Strategy)
**Default (Average of 5 MCP + Wrist)**:
```python
palm_indices = [0, 5, 9, 13, 17]
palm_center = np.mean(landmarks_px[palm_indices], axis=0)
```

**Alternative Strategies**:
- `wrist_only`: `landmarks_px[0]`
- `mcp_only`: Average of indices [5, 9, 13, 17]
- `weighted_mcp`: Weight middle MCP higher

### Hand Size
```python
hand_size_px = distance(landmarks_px[0], landmarks_px[12])
```

### Palm Orientation (2D)
```python
v = landmarks_px[5] - landmarks_px[9]
palm_angle_rad = atan2(v[1], v[0])
```

### Landmark Normalization
```python
centered = landmarks_px - landmarks_px[0]
normalized = centered / hand_size_px
```

## Requirements
- [ ] Compute all geometry fields for each detected hand
- [ ] Configurable palm center strategy (default = 5-point average)
- [ ] Handle degenerate cases (hand_size ≈ 0)
- [ ] Coordinate conversion: pixel ↔ normalized
- [ ] Pure functions — no state, no I/O

## Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `palm_center_strategy` | "five_point" | "wrist_only" \| "mcp_only" \| "five_point" \| "weighted_mcp" |
| `hand_size_method` | "wrist_middle_tip" | "wrist_middle_tip" \| "wrist_index_tip" \| "avg_fingers" |

## Performance Targets
- Compute time: <0.5ms per hand
- No allocations in hot path

## Testing
- Unit: Known landmark configurations → expected geometry values
- Property: Normalized landmarks invariant to translation/scale
- Edge: Single point, collinear points, zero hand size