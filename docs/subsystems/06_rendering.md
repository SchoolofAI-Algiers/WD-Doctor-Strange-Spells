# Subsystem: Spell Rendering & Compositing

## Responsibility
Render spell effects (magic circles) at correct position, rotation, scale, with alpha compositing over camera frame.

## Interface
```python
@dataclass
class SpellConfig:
    inner_asset: str
    outer_asset: str
    base_scale: float
    inner_rotation_speed: float
    outer_rotation_speed: float
    opacity: float
    glow_enabled: bool = False

@dataclass
class SpellTransform:
    position_px: tuple[float, float]
    scale: float
    rotation_rad: float
    opacity: float

class SpellRenderer:
    def __init__(self, config: SpellConfig, frame_shape: tuple[int, int]):
        ...
    
    def render(self, frame_bgr: np.ndarray, transform: SpellTransform, gesture: Gesture, animation_time: float) -> np.ndarray:
        ...
    
    def compute_transform(self, geometry: HandGeometry, gesture: Gesture, motion: MotionState, animation_time: float) -> SpellTransform:
        ...
```

## Asset Handling
- **Load once at init**: `cv2.imread(path, cv2.IMREAD_UNCHANGED)` → BGRA
- **Cache**: Store as `np.ndarray` (H, W, 4)

## Transform Calculation

### Position
```python
position = geometry.palm_center_px
```

### Scale
```python
spell_radius = geometry.hand_size_px * config.base_scale
asset_radius = min(asset.shape[0], asset.shape[1]) / 2
scale = spell_radius / asset_radius
```

### Rotation (Dual)
```python
hand_rotation = geometry.palm_angle_rad
anim_rotation = config.rotation_speed * animation_time
final_rotation = hand_rotation + anim_rotation
```

### Opacity
```python
final_opacity = config.opacity * gesture_confidence
```

## Alpha Compositing
```python
alpha = spell[:, :, 3:4] / 255.0
blended = alpha * spell_bgr + (1 - alpha) * frame_region
frame[y:y+h, x:x+w] = blended.astype(np.uint8)
```

### Optimized (Pre-multiplied)
```python
# At load time:
spell_bgra[:, :, :3] *= spell_bgra[:, :, 3:4] / 255.0

# At render:
frame_region = frame_region * (1 - spell_alpha) + spell_bgr
```

## Requirements
- [ ] Load and cache spell assets at startup
- [ ] Compute transform from geometry + gesture + motion + time
- [ ] Dual rotation (hand-relative + animation)
- [ ] Scale with hand size
- [ ] Alpha composite with configurable opacity
- [ ] Support multiple simultaneous spells (multi-hand)
- [ ] Gesture-gated rendering (only render on ACTIVE gestures)
- [ ] Fade in/out on gesture transitions
- [ ] No per-frame allocations

## Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `inner_asset` | "assets/spells/inner/orange.png" | |
| `outer_asset` | "assets/spells/outer/dark_red.png" | |
| `base_scale` | 1.5 | Spell radius = hand_size × this |
| `inner_rotation_speed` | 2.0 | rad/s |
| `outer_rotation_speed` | -1.0 | rad/s (opposite direction) |
| `opacity` | 0.9 | Base opacity |
| `fade_in_frames` | 5 | Frames to fade in |
| `fade_out_frames` | 8 | Frames to fade out |

## Testing
- Unit: Transform calculation with known inputs
- Unit: Alpha compositing correctness
- Visual: Manual verification at various hand sizes/distances