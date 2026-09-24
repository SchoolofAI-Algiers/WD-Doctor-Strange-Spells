# Subsystem: Capture (Webcam Acquisition)

## Responsibility
Acquire frames from webcam, handle device selection, provide consistent frame format, measure FPS.

## Interface
```python
class Capture:
    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        ...
    
    def read(self) -> np.ndarray:  # BGR frame
        ...
    
    def get_fps(self) -> float:
        ...
    
    def release(self):
        ...
```

## Requirements
- [ ] Open default webcam (device 0) with configurable resolution
- [ ] Support device enumeration (list available cameras)
- [ ] Convert to consistent format (BGR, uint8) for downstream
- [ ] Non-blocking read with timeout
- [ ] FPS measurement (rolling average over 30 frames)
- [ ] Graceful handling of camera disconnect/reconnect
- [ ] Context manager support (`with Capture() as cap:`)

## Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `device_id` | 0 | Camera index |
| `width` | 640 | Frame width (lower = faster) |
| `height` | 480 | Frame height |
| `target_fps` | 30 | Requested FPS |
| `buffer_size` | 1 | Frame buffer (1 = lowest latency) |

## Performance Targets
- Frame read latency: <5ms
- Steady FPS within ±2 of target
- Memory: <10MB for buffers

## Dependencies
- `cv2.VideoCapture`
- Optional: `cv2.CAP_DSHOW` on Windows for lower latency

## Testing
- Unit: Mock `cv2.VideoCapture` to test read/release logic
- Integration: Real camera smoke test (manual)
- Property: Frame shape matches requested resolution

## Future Extensions
- Multi-camera support (stereo, wide-angle)
- Async frame grabbing with thread + queue
- Auto-exposure/white-balance lock for consistency