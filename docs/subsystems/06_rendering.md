# Subsystem: Spell Rendering & Compositing

## Responsibility
Render spell effects (layered magic circles) at the correct position, rotation and scale, with alpha compositing over the camera frame. The renderer also keeps a small per-hand state so spells fade in and out.

Module: `src/rendering/spell_renderer.py`. Spell presets live in a separate module (`make_spells()`).

## Interface
```python
@dataclass
class LayerConfig:
    asset: str
    rotation_speed: float = 0.0             # rad/s, 0 = does not spin
    scale_mult: float = 1.0                 # size relative to the spell radius
    offset: tuple[float, float] = (0.0, 0.0)  # in spell radii, rotated with the hand
    follow_hand: bool = True                # False = stays upright instead of tilting with the hand
    opacity_mult: float = 1.0
    key_white: bool = False                 # treat white as transparent (JPG line art)
    tint: tuple[int, int, int] | None = None  # recolour the artwork, BGR order

@dataclass
class SpellConfig:
    layers: list[LayerConfig]               # drawn in order, first = bottom
    base_scale: float = 1.5
    opacity: float = 0.9
    fade_in_frames: int = 5
    fade_out_frames: int = 8
    min_radius_px: float = 20.0
    glow_enabled: bool = False              # reserved, not implemented yet

@dataclass
class SpellTransform:
    position_px: tuple[float, float]
    scale: float
    rotation_rad: float                     # hand tilt only; spin is added per layer in render()
    opacity: float
    spell_id: str = "default"

class SpellRenderer:
    def __init__(self, spells: dict[str, SpellConfig], frame_shape: tuple[int, int]) -> None: ...

    def compute_transform(self, geometry, gesture, motion, animation_time: float,
                          hand_id: int = 0, spell_id: str = "default") -> SpellTransform: ...

    def ghost_spells(self, seen_ids: set[int]) -> list[tuple[SpellTransform, None]]: ...

    def render(self, frame_bgr, transform: SpellTransform, gesture, animation_time: float): ...

    def render_all(self, frame_bgr, spells, animation_time: float): ...
```

- `frame_shape` is `(height, width)`. Frames passed to `render` must be `(H, W, 3)` `uint8` (BGR) and match it.
- `spells` is a dict of spell id to `SpellConfig` (a "spell book"). One renderer serves all spells and all hands.
- `geometry` is a `HandGeometry`; it must provide `hand_size_px`, `palm_center_px`, `palm_angle_rad`.
- `gesture` must provide `.state` (a `PoseState`) and `.confidence` (0 to 1).
- `motion` is accepted for interface compatibility and is **not used yet**. `render` ignores `gesture` too (the fade already happened in `compute_transform`).

### Module-level helpers
| Function | Purpose |
|----------|---------|
| `load_asset(path, key_white=False, tint=None)` | load, premultiply, pad to 512x512, cache |
| `layer_angle(hand_rotation_rad, speed_rad_s, animation_time)` | `hand + speed * time` |
| `build_affine(angle, scale, src_center, dst_center, out=None)` | 2x3 matrix: rotate and scale around the source center, then place it at `dst_center` |
| `composite_premultiplied(roi, layer, tmp, inv)` | in-place alpha blend of a premultiplied layer |
| `to_active(recognized_gesture)` | adapter from the gesture recognizer's output to `ActiveGesture` |

### Gesture adapter
```python
@dataclass
class ActiveGesture:
    state: PoseState        # ACTIVE or IDLE
    confidence: float

SPELL_FOR = {Gesture.OPEN_PALM: "shield"}   # recognized gesture -> spell id
```
`to_active` returns `ACTIVE` with the real confidence when the gesture is in `SPELL_FOR` and `frames_held > 0`, otherwise `IDLE` with confidence `0.0`. `GestureState` is kept as an alias of `PoseState` for old tests.

## Asset Handling
- **Load once at init:** every layer of every spell is loaded in `SpellRenderer.__init__`, so a missing file fails at startup with `FileNotFoundError`. A spell with no layers raises `ValueError`.
- **Cache:** module-level dict keyed by `(path, key_white, tint)`. The same file with different options is a different entry.
- **Format:** `float32`, shape `(512, 512, 4)`, BGRA, **premultiplied** (color already multiplied by alpha; color 0 to 255, alpha 0 to 1).
- **Normal images:** read with `IMREAD_UNCHANGED`. Grayscale and BGR inputs get an opaque alpha.
- **Aspect ratio:** the image is resized so its longer side is 512, then centered on a transparent 512x512 canvas. The circle should touch the edges of the image so that its radius is 256.
- **`key_white`:** for line art on a white page (JPG). The lightest channel becomes the paper level: white turns transparent, ink keeps its color, and values above 240 are wiped to remove JPEG haze. Ignored if the image already has alpha.
- **`tint`:** replaces the color (BGR) and keeps the alpha shape.

## Transform Calculation (`compute_transform`)
This method is **not pure**: it also updates the fade state of the hand `hand_id`.

### Position
```python
position = geometry.palm_center_px
```

### Scale
```python
size = geometry.hand_size_px          # NaN, inf or <= 0 -> cfg.min_radius_px
spell_radius = size * cfg.base_scale
spell_radius = clamp(spell_radius, cfg.min_radius_px, max(H, W))
scale = spell_radius / ASSET_RADIUS   # ASSET_RADIUS = 256 (all assets are 512x512)
```

### Rotation
`transform.rotation_rad = geometry.palm_angle_rad` (hand tilt only). The animation spin is added **per layer** in `render()`:
```python
angle = layer_angle(hand_rot if layer.follow_hand else 0.0, layer.rotation_speed, animation_time)
```
Positive angles turn clockwise on screen (image y axis points down), matching `palm_orientation`.

### Opacity and fade
```python
fade: step toward 1 if state == ACTIVE (+1/fade_in_frames), else toward 0 (-1/fade_out_frames)
opacity = cfg.opacity * confidence * fade
```
- Fade is per hand (`hand_id`) and linear.
- If a hand switches `spell_id`, its fade restarts at 0.
- An unknown `spell_id` raises `KeyError`.

## Ghosts (hands that disappeared)
When the tracker loses a hand there is no new geometry, so call `ghost_spells(seen_ids)` **once per frame** with the ids detected this frame. For each missing hand it decreases the fade by `1/fade_out_frames` and returns a **copy** of its last transform with the faded opacity (the stored transform is never modified). When the fade reaches 0 the hand is forgotten. Hands that never had a transform or spell are dropped. Calling it twice in one frame makes the fade run at double speed.

## Rendering (`render`)
1. Clamp `transform.opacity` to `[0, 1]`. If it is 0 or less, return immediately.
2. Compute the drawing window: a square around the palm, with radius `spell_radius * extent`, where `extent` is the largest `scale_mult + |offset|` among the layers (so no layer is cut). Clip it to the frame. If it is fully off screen, return.
3. For each layer, in order (first = bottom):
   - layer center = palm + `offset` rotated with the hand, in spell radii;
   - `build_affine(angle, scale * scale_mult, ...)` then one `cv2.warpAffine` into the window (transparent border);
   - multiply by `opacity * layer.opacity_mult` if below 1;
   - composite onto the frame.
4. Return the same `frame_bgr` array (modified in place).

### Alpha compositing (premultiplied)
```python
# At load time:
img[:, :, :3] *= img[:, :, 3:4]            # color * alpha, alpha in 0..1

# At render time:
result = layer_color + frame * (1 - alpha)
result = clip(result, 0, 255)
```
Each step uses `out=` buffers, so no new arrays are created.

### Multi-hand
`render_all(frame, spells, animation_time)` takes `(SpellTransform, gesture)` pairs, one per hand (ghosts are `(transform, None)`), and draws them one after another. They share the scratch buffers, so rendering is **not thread-safe**.

## Per-frame usage
```python
spells = []
seen = set()
for hand_id, (geometry, recognized) in hands.items():
    g = to_active(recognized)
    spell_id = SPELL_FOR.get(recognized.gesture, "shield")
    spells.append((renderer.compute_transform(geometry, g, None, t, hand_id, spell_id), g))
    seen.add(hand_id)
spells += renderer.ghost_spells(seen)       # once per frame, also when no hand is detected
renderer.render_all(frame, spells, t)       # t = seconds since start
```
`render_all` must be called every frame, even with no hands, otherwise ghosts never fade.

## Requirements
- [x] Load and cache spell assets at startup
- [x] Compute transform from geometry + gesture + time (`motion` accepted but unused)
- [x] Dual rotation (hand-relative + per-layer animation spin, with `follow_hand` option)
- [x] Scale with hand size (clamped, with a fallback for bad measurements)
- [x] Alpha composite with configurable opacity (spell, gesture confidence and layer opacity)
- [x] Support multiple simultaneous spells (multi-hand, multiple spell types, multiple layers)
- [x] Gesture-gated rendering (only `PoseState.ACTIVE` fades a spell in)
- [x] Fade in/out on gesture transitions, and ghost fade-out when a hand is lost
- [x] No per-frame allocations for the large buffers (preallocated flat buffers, `out=` everywhere). Small Python objects are still created per frame (transforms, ghost copies)
- [ ] Glow effect (`glow_enabled` exists but is not implemented)

## Configuration
Defaults of `SpellConfig`:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `layers` | (required) | list of `LayerConfig`, first = bottom |
| `base_scale` | 1.5 | Spell radius = hand size x this |
| `opacity` | 0.9 | Base opacity |
| `fade_in_frames` | 5 | Frames to fade in |
| `fade_out_frames` | 8 | Frames to fade out |
| `min_radius_px` | 20.0 | Smallest allowed spell radius (also the fallback for bad hand sizes) |
| `glow_enabled` | False | Reserved |

Defaults of `LayerConfig`: `rotation_speed=0.0`, `scale_mult=1.0`, `offset=(0, 0)`, `follow_hand=True`, `opacity_mult=1.0`, `key_white=False`, `tint=None`.

Current preset (`make_spells()`), one spell called `"shield"` (`base_scale=1.5`, `opacity=0.9`):
| Layer | Asset | `rotation_speed` | `scale_mult` |
|-------|-------|------------------|--------------|
| 1 (bottom) | `assets/spells/outer/dark_red.png` | 0.8 rad/s | 1.0 |
| 2 (top) | `assets/spells/inner/orange.png` | -1.4 rad/s | 0.65 |

The two rings spin in opposite directions. This replaces the old `inner_asset`, `outer_asset`, `inner_rotation_speed` and `outer_rotation_speed` fields.

## Testing
Tests in `tests/test_rendering.py`. They generate their own PNGs in pytest's temp folder, so no camera or real assets are needed.
- **Pure functions:** `layer_angle`, `build_affine` (center lands on target, rotation, scale, `out` buffer reused), `composite_premultiplied` (transparent, opaque, half alpha, additive, clipping).
- **`load_asset`:** errors, premultiplication, 512x512 float32, aspect ratio and centering, grayscale/BGR inputs, `key_white`, `tint`, cache.
- **Init:** empty spell, missing asset, extent calculation.
- **Transform:** scale and clamping (including NaN, inf, 0, negative), fade in/out, independent hands, confidence, spell switching, unknown spell.
- **Ghosts:** copy not mutated, fade-out then forgotten, invisible hands dropped.
- **Render:** nothing drawn (zero opacity, off screen), disc drawn, opacity blend and clamp, clipping at corners, offset rotates with the hand, `scale_mult`, layer order, `opacity_mult`, angle and scale passed to `build_affine`, scratch buffers do not leak, `render_all`, full hand lifecycle.
- **Visual (manual, not automated):** check the spell on a real hand at several sizes and distances.

## Known limitations
- `glow_enabled` and the `motion` parameter are not used yet.
- `compute_transform` changes internal state, so call it once per hand per frame.
- Shared scratch buffers: use one renderer from one thread.
- Assets must have the circle touching the image edges, or the spell will look smaller or larger than the hand-based radius.
