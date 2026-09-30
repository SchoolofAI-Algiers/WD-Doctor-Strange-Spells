import cv2  # OpenCV  used for image processing and computer vision tasks
import sys
from collections import deque
import time

def list_available_cameras(max_index: int = 5) -> list[int]: # 5 is random chosen as max of cameras to ckeck
    """Return the camera indices that actually open."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i) #opencv identifies cameras by their index, starting from 0. The first camera is usually index 0, the second camera is index 1, and so on. The maximum index to check is specified by max_index.
        if cap.isOpened():
            found.append(i)
        cap.release()
    return found

class Capture:

    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480, fps: int = 30,buffer_size: int = 1):
        # Initialize the video capture object
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size #buffer size is the number of frames that can be stored in the camera's internal buffer before they are processed. A larger buffer size can help prevent dropped frames, but it can also introduce latency. A smaller buffer size can reduce latency, but it may also increase the risk of dropped frames if the processing cannot keep up with the frame rate.
        
        self.cap = cv2.VideoCapture(device_id)  # captures video from device_id (0 = default camera) , it returns a VideoCapture object that can be used to read frames from the camera
        self._times = deque(maxlen=30)  # max 30 latest timestamps

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open your device's camera {device_id}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)


    def read(self): #reads a frame from the camera
        ok, frame = self.cap.read() #read returns a boolean (ok), 1 if it was successful, 0 if it failed, and the frame itself (frame)

        if not ok:
            raise RuntimeError("Failed to read frame from camera")
        
        self._times.append(time.monotonic())   # append time frame was captured to the deque

        return frame

    
    def get_fps(self) -> float: #result is a float representing the frames per second (FPS) of the video capture

        if len(self._times) < 2: # if we don't have at least 2 timestamps, we can't calculate FPS
            return 0.0
        elapsed = self._times[-1] - self._times[0] # calculate the elapsed time between the first and last timestamps in the deque [0] is first and [-1] is last, so we subtract the first timestamp from the last timestamp to get the total time elapsed during the capture of frames
        if elapsed <= 0.0:
            return 0.0 # to avoid division by zero or negative elabsed time
        
        return (len(self._times) - 1) / elapsed # number of frames / elapsed time = frames per second (FPS), -1 as 5 frames for ex gives us 4 gaps between them

    def release(self):
        self.cap.release() #if we dont release the camera, it will remain locked and unavailable for other applications or future runs of the program. This can lead to errors or unexpected behavior when trying to access the camera again.
    
    def __enter__(self):  # runs when the `with` block starts always
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # runs when the `with` block ends, even after an error
        self.release()


