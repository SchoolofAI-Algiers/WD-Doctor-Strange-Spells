import sys

import cv2
import numpy as np
from src.capture.capture import Capture, list_available_cameras

if __name__ == "__main__":
    available_cameras = list_available_cameras()

    if not available_cameras:
        print("No available cameras found.")
        sys.exit(1)

    print("Cameras found:", available_cameras)

    with (
        Capture(width=640, height=480) as cam
    ):  # we create an object of capture with width and height given, it will execute enter at beggining and exit at the end
        print("Camera opened successfully!")

        # check the format guarantee before the loop (check by read)
        frame = cam.read()
        assert frame.shape == (480, 640, 3), (
            f"Unexpected shape {frame.shape}"
        )  # if cond is false display msg
        assert frame.dtype == np.uint8, f"Unexpected dtype {frame.dtype}"

        print("Press q to quit, d to fake a disconnect, t to test the timeout")

        while True:
            try:
                frame = cam.read()
            except TimeoutError as e:
                print("Timeout caught:", e)
                break

            fps = cam.get_fps()
            cv2.putText(
                frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            cv2.imshow("Webcam Test", frame)

            key = cv2.waitKey(1) & 0xFF
            closed = (
                cv2.getWindowProperty("Webcam Test", cv2.WND_PROP_VISIBLE) < 1
            )  # checks if window is closed
            if key == ord("q") or closed:
                break

            if key == ord("d"):
                cam.cap.release()  # simulate the camera dying, read() should reconnect by itself, we are in this case where read (false,none)

            if key == ord("t"):
                cam.device_id = 99  # a camera that doesn't exist
                cam.cap.release()  # read() can't reconnect, so it should raise TimeoutError

    cv2.destroyAllWindows()

    # after the with block the camera is released, so reading must be refused
    try:
        cam.read()
    except RuntimeError as e:
        print("Correctly refused:", e)
