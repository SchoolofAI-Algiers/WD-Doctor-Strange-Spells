# Subsystem: Motion Analysis

## Responsibility
Compute velocity, acceleration, trajectory from palm center over time. Record fingertip paths for spell drawing.

## Interface
```python
@dataclass
class MotionState:
    velocity_px_s: tuple[float, float]
    speed_px_s: float
    direction_rad: float
    acceleration_px_s2: tuple[float, float]
    trajectory: list[tuple[float, float]]
    fingertip_trajectory: list[tuple[float, float]]
    is_stationary: bool
    is_moving_toward_camera: bool

class MotionAnalyzer:
    def __init__(self, max_trajectory_len: int = 60, dt_smoothing: float = 0.1):
        ...
    
    def update(self, hand_id: int, geometry: HandGeometry, dt: float) -> MotionState:
        ...
    
    def get_trajectory(self, hand_id: int) -> list[tuple[float, float]]:
        ...
    
    def clear(self, hand_id: int):
        ...
```

## Calculations

### Velocity
```python
vx = (x_t - x_prev) / dt
vy = (y_t - y_prev) / dt
vx_smooth = alpha * vx + (1 - alpha) * vx_prev
speed = sqrt(vx_smooth^2 + vy_smooth^2)
direction = atan2(vy_smooth, vx_smooth)
```

### Acceleration
```python
ax = (vx_smooth - vx_prev_smooth) / dt
```

### Trajectory Buffer
- Ring buffer of last N palm centers (default 60 = ~2s at 30fps)

### Fingertip Trajectory (Spell Drawing)
- Record index fingertip (landmark 8) when gesture = POINTING

### Depth Movement Heuristic
```python
size_ratio = hand_size_t / hand_size_prev
if size_ratio > 1.02: moving_toward = True
elif size_ratio < 0.98: moving_away = True
```

## Requirements
- [ ] Per-hand motion state (tracked by hand_id)
- [ ] Velocity, speed, direction, acceleration
- [ ] Ring-buffer trajectory (palm + fingertip)
- [ ] Stationary detection
- [ ] Depth movement heuristic
- [ ] Automatic cleanup when hand lost

## Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_trajectory_len` | 60 | Max points in ring buffer |
| `velocity_ema_alpha` | 0.3 | Smoothing for velocity |
| `stationary_speed_threshold` | 20 | px/s below = stationary |
| `depth_change_threshold` | 0.02 | Hand size ratio for toward/away |

## Testing
- Unit: Known positions + dt → expected velocity/acceleration
- Property: Stationary hand → speed ≈ 0
- Property: Constant velocity → acceleration ≈ 0