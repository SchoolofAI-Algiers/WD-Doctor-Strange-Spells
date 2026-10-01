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

        self._times = deque(maxlen=30)  # max 30 latest timestamps
        self._closed = False  # becomes True after release()
        self.cap = None # VideoCapture object
        self._open()  # opens the camera and applies the settings, raises RuntimeError if it can't open

    def _open(self): #this needs to be a separate function because we need to call it from __init__ and _reconnect so its cant be in __init__ directly.
        #open is called when we create an object+ when reconnecting cam as it has default settings of the cam
        self.cap = cv2.VideoCapture(self.device_id)  # captures video from device_id (0 = default camera) .

        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"Could not open your device's camera {self.device_id}")

#request the camera to use specified  width, height,fps and buffer size.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

    def _reconnect(self):  # drops the old handle (we affected none) and tries to open the camera again with the same settings
        if self.cap is not None:
            self.cap.release()
            self.cap = None #we will call open that creates it again
        try:
            self._open()
        except RuntimeError: #if error of this type happens, we pass and keep executing code 

            pass  # camera still missing, read() will keep trying until its timeout runs out



    def read(self, timeout: float = 2.0): #reads a frame from the camera, raises TimeoutError if no frame arrives within timeout seconds
        if self._closed: #when its realised we set it to true , we cant rfead if not camera
            raise RuntimeError("Capture has been released")

#timout is the max limit of waiting for a frame to arive
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.cap is not None:
                ok, frame = self.cap.read() #read returns a boolean (ok), True if it was successful, False if it failed, and the frame itself (frame)

                if ok and frame is not None:
                    self._times.append(time.monotonic())   # append time frame was captured to the deque
                    return frame

            self._reconnect()  # read failed or camera is missing, try to recover
            time.sleep(0.1) #it gives time to usb to be back and camera to be reconnected.

        raise TimeoutError(f"No frame from camera {self.device_id} within {timeout}s")

    
    def get_fps(self) -> float: #result is a float representing the frames per second (FPS) of the video capture

        if len(self._times) < 2: # if we don't have at least 2 timestamps, we can't calculate FPS
            return 0.0
        elapsed = self._times[-1] - self._times[0] # calculate the elapsed time between the first and last timestamps in the deque [0] is first and [-1] is last, so we subtract the first timestamp from the last timestamp to get the total time elapsed during the capture of frames
        if elapsed <= 0.0:
            return 0.0 # to avoid division by zero or negative elabsed time
        
        return (len(self._times) - 1) / elapsed # number of frames / elapsed time = frames per second (FPS), -1 as 5 frames for ex gives us 4 gaps between them

    def release(self):
        self._closed = True #setting closed to true (camera closed) 
        if self.cap is not None:
            self.cap.release() #if we dont release the camera, it will remain locked and unavailable for other applications or future runs of the program. This can lead to errors or unexpected behavior when trying to access the camera again.
            self.cap = None
    
    def __enter__(self):  # runs when the `with` block starts always
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # runs when the `with` block ends, even after an error
        self.release()

