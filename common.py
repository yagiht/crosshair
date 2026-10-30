"""Shared bits for cursor_lab: the camera, MediaPipe, and a few constants.

cursor_lab is a separate experiment. Nothing here imports from FIVES, and FIVES
doesn't import anything from here.
"""

import threading
import time
import urllib.request
from pathlib import Path

LAB = Path(__file__).parent
DATA_DIR = LAB / "data"
RUNS_DIR = LAB / "runs"
MODEL_FILE = LAB / "models" / "hand_landmarker.task"     # the lab's own copy of MediaPipe's model
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")

# MediaPipe's 21 hand points: the ones we use
INDEX_TIP = 8
INDEX_KNUCKLE = 5
MIDDLE_KNUCKLE = 9

# Same idea as FIVES' CURSOR_MARGIN: only the middle of the picture maps to the window.
MARGIN = 0.15

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30


def make_landmarker():
    """MediaPipe's hand finder, one hand, frames in order, on the CPU (like FIVES)."""
    import mediapipe as mp
    if not MODEL_FILE.exists():
        MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading the hand model (first run only)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    vision = mp.tasks.vision
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_FILE),
                                          delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    return vision.HandLandmarker.create_from_options(options)


class Camera:
    """Reads the webcam on its own thread and keeps only the newest picture, with the time
    it arrived. Old pictures would mean wrong timestamps, and timestamps matter here: the
    model learns from how points move over TIME."""

    def __init__(self):
        import cv2
        self.cam = cv2.VideoCapture(CAMERA_INDEX)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cam.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        if not self.cam.isOpened():
            raise SystemExit("Couldn't open the camera. Check CAMERA_INDEX in common.py "
                             "and the camera permission for your terminal.")
        self.cond = threading.Condition()
        self.frame, self.stamp, self.seq = None, 0.0, 0
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, frame = self.cam.read()
            stamp = time.monotonic()
            if not ok:
                time.sleep(0.01)
                continue
            with self.cond:
                self.frame, self.stamp, self.seq = frame, stamp, self.seq + 1
                self.cond.notify_all()

    def read(self, last_seq):
        """Wait for a picture newer than last_seq. Returns (frame, stamp, seq)."""
        with self.cond:
            while self.seq == last_seq and self.running:
                self.cond.wait(timeout=1.0)
            return self.frame, self.stamp, self.seq

    def close(self):
        self.running = False
        self.cam.release()


def to_window(p, w, h):
    """Camera point (0..1) -> window pixels, using only the middle of the picture."""
    span = 1 - 2 * MARGIN
    x = min(max((p[0] - MARGIN) / span, 0.0), 1.0)
    y = min(max((p[1] - MARGIN) / span, 0.0), 1.0)
    return int(x * (w - 1)), int(y * (h - 1))
