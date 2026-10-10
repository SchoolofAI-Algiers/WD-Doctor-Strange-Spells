# Subsystem: Geometry Calculator

## Responsibility
Compute palm center, hand size, palm orientation, and normalized landmarks from raw hand landmarks.

Module: `geometry_calculatorV2.py`. It contains a set of pure module-level functions plus a thin `GeometryCalculator` class that wires them together for one image size.

## Interface
```python
FloatArray = NDArray[np.floating[Any]]

@dataclass(frozen=True)
class GeometryConfig:
    palm_center_strategy: str = "five_point"
    hand_size_method: str = "wrist_middle_tip"

@dataclass(frozen=True)
class HandGeometry:
    palm_center_px: tuple[float, float]
    palm_center_norm: tuple[float, float]
    hand_size_px: float
    hand_size_norm: float
    palm_angle_rad: float
    landmarks_norm: FloatArray          # shape (N, 2), see "Two meanings of norm" below

class GeometryCalculator:
    def __init__(self, image_width: int, image_height: int,
                 config: GeometryConfig | None = None) -> None:
        ...

    def compute(self, hand: Any) -> HandGeometry:
        ...
```

- `hand` is a `HandLandmarks` object from the tracking subsystem. The only thing the calculator requires is the attribute `hand.landmarks_norm`: an `(N, 2)` or `(N, 3)` array of MediaPipe normalized coordinates (0 to 1 over the image). It is typed `Any` so this module does not import the tracking subsystem.
- `config=None` means `GeometryConfig()` (all defaults).
- Both dataclasses are `frozen=True`: fields cannot be reassigned after creation.

### Module-level functions
| Function | Signature | Returns |
|----------|-----------|---------|
| `distance` | `(p1, p2)` | `float`, Euclidean distance |
| `norm2px` | `(landmarks_norm, image_width, image_height)` | `(N, 2)` array in pixels (z dropped) |
| `palm_center` | `(landmarks_px, palm_center_strategy="five_point")` | `(2,)` array in pixels |
| `hand_size` | `(landmarks_px, hand_size_method="wrist_middle_tip")` | `float`, pixels |
| `palm_orientation` | `(landmarks_px)` | `float`, radians |
| `landmark_normalization` | `(landmarks_px, hand_size_px)` | `(N, 2)` array, wrist-centered and scaled |

`palm_center` was called `palmCenter` in earlier drafts.

## Calculations

### Coordinate conversion
```python
resolution = np.array([image_width, image_height])
landmarks_px = landmarks_norm[:, :2] * resolution
```
Only x and y are kept. If the input has a z column it is dropped, so the output is always `(N, 2)`.

### Palm Center (Configurable Strategy)
**Default `five_point` (average of wrist + 4 MCP)**:
```python
palm_indices = [0, 5, 9, 13, 17]
center = np.mean(landmarks_px[palm_indices], axis=0)
```

**Alternative strategies**:
- `wrist_only`: `landmarks_px[0]`
- `mcp_only`: average of indices `[5, 9, 13, 17]`
- `weighted_mcp`: average of `[0, 5, 9, 13, 17]` with weights `[1, 1, 2, 1, 1]`, so the middle-finger MCP (landmark 9) counts twice

An unknown strategy name raises `ValueError`.

### Hand Size
**Default `wrist_middle_tip`**:
```python
hand_size_px = distance(landmarks_px[0], landmarks_px[12])
```

**Alternative methods**:
- `wrist_index_tip`: `distance(landmarks_px[0], landmarks_px[8])`
- `avg_fingers`: mean of the wrist distances to the index (8), middle (12), ring (16) and pinky (20) fingertips

An unknown method name raises `ValueError`.

### Palm Orientation (2D)
```python
v = landmarks_px[5] - landmarks_px[9]          # middle MCP -> index MCP
palm_angle_rad = float(np.arctan2(v[1], v[0]))
```
The result is in `[-pi, pi]`. The image y axis points down, so an angle of `0` means "pointing right" and `+pi/2` means "pointing down" (clockwise on screen).

### Landmark Normalization
```python
if hand_size_px > 1e-6:
    centered = landmarks_px - landmarks_px[0]
    normalized = centered / hand_size_px
else:
    normalized = np.zeros_like(landmarks_px)
```
The wrist ends up at `(0, 0)` and the hand size becomes the unit length.

### Normalized fields of `HandGeometry`
```python
palm_center_norm = palm_center_px / [image_width, image_height]
hand_size_norm   = hand_size_px / max(image_width, image_height)   # fraction of the longer side
```

### Two meanings of "norm" (naming warning)
| Name | Where | Meaning |
|------|-------|---------|
| `hand.landmarks_norm` (input) | tracking subsystem | MediaPipe coordinates, 0 to 1 over the **image** |
| `HandGeometry.landmarks_norm` (output) | this subsystem | pixels centered on the wrist and divided by the **hand size** |

They are different spaces. Do not feed the output back into `norm2px`.

## Requirements
- [x] Compute all geometry fields for each detected hand (`GeometryCalculator.compute`)
- [x] Configurable palm center strategy (default = 5-point average)
- [x] Configurable hand size method (default = wrist to middle fingertip)
- [x] Unknown strategy / method names fail with `ValueError` instead of a confusing error
- [x] Handle degenerate hand size in normalization (`landmarks_norm` is all zeros when `hand_size_px <= 1e-6`)
- [x] Coordinate conversion normalized -> pixel (`norm2px`)
- [ ] Coordinate conversion pixel -> normalized as a reusable function (today it is done inline for `palm_center_norm` and `hand_size_norm`)
- [x] Pure functions with no I/O; the calculator only holds its image size and an immutable config

### Degenerate cases (what actually happens)
| Case | Result |
|------|--------|
| `hand_size_px <= 1e-6` | `landmarks_norm` is all zeros |
| `hand_size_px` itself | **not** clamped: it can be `0.0`, and `hand_size_norm` is then `0.0` too. The consumer must handle it (the spell renderer substitutes a minimum radius) |
| landmarks 5 and 9 identical | `palm_angle_rad = 0.0` (`arctan2(0, 0)`) |
| all landmarks identical | size `0`, angle `0`, palm center equals that point |

## Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `palm_center_strategy` | "five_point" | "wrist_only" \| "mcp_only" \| "five_point" \| "weighted_mcp" |
| `hand_size_method` | "wrist_middle_tip" | "wrist_middle_tip" \| "wrist_index_tip" \| "avg_fingers" |

Passed as a `GeometryConfig` to `GeometryCalculator`.

## Performance Targets
- Compute time: <0.5ms per hand
