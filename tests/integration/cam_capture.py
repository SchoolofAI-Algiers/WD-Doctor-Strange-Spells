from src.capture.capture import Capture, list_available_cameras
import cv2
import time
import sys

if __name__ == "__main__":
    available_cameras = list_available_cameras()

    if not available_cameras:
        print("No available cameras found.")
        sys.exit(1)

    print("Cameras found:", list_available_cameras())

    with Capture(width=640, height=480) as cam: #with is used to ensure that the camera is properly released when the block is exited, even if an error occurs. This is important for resource management and preventing issues with accessing the camera in future runs of the program.
        print("Camera opened successfully!")

        while True:
            frame = cam.read()

            fps = cam.get_fps()
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Webcam Test", frame)

            key = cv2.waitKey(1) & 0xFF
            closed = cv2.getWindowProperty("Webcam Test", cv2.WND_PROP_VISIBLE) < 1 #checks if window is closed 
            if key == ord("q") or closed:
                break

    cv2.destroyAllWindows()