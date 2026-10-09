import time

from src.capture.capture import Capture

N_FRAMES = 300
TARGET_FPS = 30

with Capture(width=640, height=480, fps=TARGET_FPS) as cam:
    for _ in range(30):  # warm-up, so startup frames don't skew the numbers
        cam.read()

    read_times = []
    for _ in range(N_FRAMES):
        t0 = time.perf_counter()
        frame = cam.read()
        read_times.append((time.perf_counter() - t0) * 1000)  # milliseconds

    fps = cam.get_fps()

print(f"Average read time: {sum(read_times) / len(read_times):.1f} ms")
print(f"Worst read time:   {max(read_times):.1f} ms")
print(f"Measured FPS:      {fps:.1f} (target {TARGET_FPS}, allowed {TARGET_FPS - 2} to {TARGET_FPS + 2})")
print(f"Frame size:        {frame.nbytes / 1_000_000:.2f} MB")
