"""The window-closing game. It RECORDS training data and MEASURES how fast you are.

    python game.py                 # play with plain tracking (records data)
    python game.py --model         # play with the trained model predicting your finger
    python game.py --seed 1        # same windows in the same order (fair before/after comparison)
    python game.py --raw           # no smoothing on the dot (see exactly what MediaPipe sees)
    python game.py --no-precise    # no precision pointer (the dot is always 1:1 with your finger)
    python game.py --margin 0.15   # how much of the picture maps to the window (FIVES CURSOR_MARGIN)

How to play: point with your index finger. Little windows pop up; hover the dot
over one until its ring fills to close it. Small windows test precision, far ones
test speed. Every game saves your hand movement to data/ (training data) and your
average time to runs/scores.csv (the benchmark).

Keys in the game window: q = quit early (still saves).

The dot you SEE is smoothed (One Euro filter, like FIVES) and ignores one-frame glitches.
What gets RECORDED is always the raw MediaPipe points, so your training data stays honest.
"""

import argparse
import csv
import math
import random
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

import common as C
import features as F

W, H = 960, 720
DWELL = 0.30          # seconds to hover a window to close it
BG = (18, 12, 8)
BLUE = (255, 170, 60)
WHITE = (235, 235, 235)
DIM = (110, 90, 70)


class OneEuro:
    """Smart smoothing (same as FIVES): heavy when the point is nearly still, light when it moves fast.
    min_cutoff: steadiness when still (lower = steadier). beta: how fast it loosens up (higher = less lag)."""

    def __init__(self, min_cutoff=1.5, beta=8.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def update(self, p, t):
        x, y = float(p[0]), float(p[1])
        if self.x is None:
            self.x, self.y, self.t = x, y, t
            return np.array([x, y], np.float32)
        dt = max(t - self.t, 1e-3)
        self.t = t
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx += a_d * ((x - self.x) / dt - self.dx)
        self.dy += a_d * ((y - self.y) / dt - self.dy)
        a = self._alpha(self.min_cutoff + self.beta * math.hypot(self.dx, self.dy), dt)
        self.x += a * (x - self.x)
        self.y += a * (y - self.y)
        return np.array([self.x, self.y], np.float32)

    def reset(self):
        self.x = self.y = self.t = None
        self.dx = self.dy = 0.0


class PrismPointer:
    """Precise when slow, pinned when fast (the "PRISM" technique from VR research).

      slow, careful hand -> the cursor moves only a fraction as far (min_scale..1): precise aiming
      fast hand          -> the cursor moves 1:1 AND catches back up to your fingertip,
                            so any gap built up while aiming slowly disappears
      still hand         -> nothing moves (no slow drift behind your back)

    Works on camera positions (0..1). update() returns the offset (cursor - fingertip); add it to
    every hand point and the rest of FIVES works unchanged. The offset is usually 0, so the cursor
    is right under your finger; it only opens up a little while you aim slowly.
    """

    def __init__(self, precise_speed=0.35, min_scale=0.35, catchup_speed=0.9, lo=0.0, hi=1.0):
        self.precise_speed = precise_speed      # below this hand speed (picture widths / s): slowed down
        self.min_scale = min_scale              # the slowest the cursor goes, as a fraction of your hand
        self.catchup_speed = catchup_speed      # at this speed the gap closes fastest
        self.lo, self.hi = lo, hi
        self.off = (0.0, 0.0)
        self.prev = None
        self.t = None
        self.last_scale = 1.0

    def scale(self, speed: float) -> float:
        """How far the cursor moves for each bit your hand moves, at this hand speed."""
        return min(1.0, max(self.min_scale, speed / max(1e-6, self.precise_speed)))

    def catchup(self, speed: float) -> float:
        """What fraction of the gap to close this frame (0 when slow, up to 0.5 when fast)."""
        span = max(1e-6, self.catchup_speed - self.precise_speed)
        return 0.5 * min(1.0, max(0.0, (speed - self.precise_speed) / span))

    def update(self, hand, t, frozen=False):
        """hand = smoothed fingertip (0..1). frozen = keep the cursor still (open palm). Returns the offset."""
        if self.prev is not None:
            dt = max(t - self.t, 1e-3)
            dx, dy = hand[0] - self.prev[0], hand[1] - self.prev[1]
            if frozen:                          # cursor holds still while the hand moves: absorb it
                ox, oy = self.off[0] - dx, self.off[1] - dy
            else:
                speed = math.hypot(dx, dy) / dt
                k = self.last_scale = self.scale(speed)
                ox, oy = self.off[0] + (k - 1) * dx, self.off[1] + (k - 1) * dy
                c = self.catchup(speed)
                ox, oy = ox * (1 - c), oy * (1 - c)
            # keep the cursor inside the usable area (pushing past the edge just slides the gap)
            cx = min(self.hi, max(self.lo, hand[0] + ox))
            cy = min(self.hi, max(self.lo, hand[1] + oy))
            self.off = (cx - hand[0], cy - hand[1])
        self.prev, self.t = hand, t
        return self.off

    def hand_gone(self):
        """The hand left the picture: when it comes back the cursor is under your finger again."""
        self.prev = None
        self.off = (0.0, 0.0)


GLITCH_JUMP = 0.2     # a fingertip "teleporting" more than 20% of the picture in one frame is a tracking glitch


def new_target(rng, last):
    """A random window. One in three is small (precision), and it's never right next to the last one."""
    small = rng.random() < 1 / 3
    w, h = (34, 26) if small else (90, 66)
    while True:
        x, y = rng.randint(40, W - 40 - w), rng.randint(90, H - 40 - h)
        if last is None or np.hypot(x - last[0], y - last[1]) > 180:
            return (x, y, w, h)


def draw_window(img, rect, fill):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 40, 25), -1)
    cv2.rectangle(img, (x, y), (x + w, y + 9), BLUE, -1)                    # title bar
    cv2.rectangle(img, (x, y), (x + w, y + h), BLUE, 1)
    s = min(w, h) // 5
    cx, cy = x + w // 2, y + 5 + h // 2
    cv2.line(img, (cx - s, cy - s), (cx + s, cy + s), WHITE, 2)             # the X
    cv2.line(img, (cx - s, cy + s), (cx + s, cy - s), WHITE, 2)
    if fill > 0:
        cv2.ellipse(img, (cx, cy), (s + 8, s + 8), -90, 0, int(360 * fill), WHITE, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="store_true", help="steer with the trained model's prediction")
    ap.add_argument("--windows", type=int, default=40, help="windows per game")
    ap.add_argument("--seed", type=int, default=None, help="same number = same windows, for fair comparisons")
    ap.add_argument("--raw", action="store_true", help="no smoothing / glitch guard on the dot you see")
    ap.add_argument("--cutoff", type=float, default=1.5, help="One Euro min_cutoff (FIVES uses 1.5)")
    ap.add_argument("--beta", type=float, default=8.0, help="One Euro beta (FIVES uses 8.0)")
    ap.add_argument("--no-precise", action="store_true", help="turn the precision pointer off")
    ap.add_argument("--min-scale", type=float, default=0.35, help="precision: slowest cursor speed (FIVES PRECISE_MIN_SCALE)")
    ap.add_argument("--precise-speed", type=float, default=0.35, help="precision: below this hand speed it slows (FIVES PRECISE_SPEED)")
    ap.add_argument("--margin", type=float, default=0.22, help="edge margin of the picture (FIVES CURSOR_MARGIN)")
    args = ap.parse_args()

    predictor = None
    if args.model:
        import torch
        from model import load
        path = C.RUNS_DIR / "cursor_predictor.pt"
        if not path.exists():
            raise SystemExit("No trained model yet. Run  python train.py  first.")
        predictor, info = load(path)
        torch.set_num_threads(1)
        print(f"Using {path.name} (test error {info['test_error']:.4f})")

    smoother = None if args.raw else OneEuro(args.cutoff, args.beta)
    C.MARGIN = args.margin                      # same mapping as FIVES' CURSOR_MARGIN
    accel = None if args.no_precise else PrismPointer(args.precise_speed, args.min_scale, 0.9, C.MARGIN, 1 - C.MARGIN)
    last_raw = None                             # last good raw fingertip, for spotting glitches
    glitches = held = 0
    rng = random.Random(args.seed)
    landmarker = C.make_landmarker()
    cam = C.Camera()
    times, frames = [], []                      # the recording
    recent = deque(maxlen=F.HISTORY)            # (t, signal) for the model
    target, last_target = None, None
    spawned = hover_since = None
    close_times = []
    seq, last_ms = 0, -1
    fps, last = 0.0, time.monotonic()
    cv2.namedWindow("cursor lab", cv2.WINDOW_AUTOSIZE)
    print("Point at the windows to close them. q quits.")

    try:
        while len(close_times) < args.windows:
            frame, stamp, seq = cam.read(seq)
            if frame is None:
                continue
            frame = cv2.flip(frame, 1)          # mirror, so moving right moves the dot right
            ms = max(int(stamp * 1000), last_ms + 1)
            last_ms = ms
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(image, ms)

            canvas = np.full((H, W, 3), BG, np.uint8)
            pointer = None
            if result.hand_landmarks:
                pts = np.array([(lm.x, lm.y) for lm in result.hand_landmarks[0]], np.float32)
                times.append(stamp)
                frames.append(pts)
                sig = F.signal(pts[None])[0]
                if recent and stamp - recent[-1][0] > F.MAX_GAP:
                    recent.clear()
                recent.append((stamp, sig))
                tip = pts[C.INDEX_TIP]
                if smoother is not None:        # glitch guard: only for the dot you see
                    jumped = last_raw is not None and np.abs(tip - last_raw).max() > GLITCH_JUMP
                    if jumped and held < 2:     # a teleport: hold the dot still for up to 2 frames
                        glitches += 1
                        held += 1
                        tip = last_raw
                    else:                       # normal (or it stayed there: it was real, follow it)
                        last_raw, held = tip, 0
                aim = tip                       # where the dot should go before smoothing
                if predictor is not None:
                    cv2.circle(canvas, C.to_window(tip, W, H), 5, DIM, -1)     # faint: raw finger
                    if len(recent) == F.HISTORY:
                        wt = np.array([r[0] for r in recent])
                        ws = np.array([r[1] for r in recent])
                        x = torch.from_numpy(F.window_features(wt, ws)[None])
                        with torch.no_grad():
                            aim = tip + predictor(x)[0].numpy()
                pointer = aim if smoother is None else smoother.update(aim, stamp)
                if accel is not None:           # slow hand = precise, fast hand = pinned (like FIVES)
                    off = accel.update(pointer, stamp)
                    pointer = np.array([pointer[0] + off[0], pointer[1] + off[1]], np.float32)
            else:
                times.append(stamp)
                frames.append(np.full((21, 2), np.nan, np.float32))
                recent.clear()
                if smoother is not None:
                    smoother.reset()                    # hand lost: start fresh when it comes back
                last_raw, held = None, 0
                if accel is not None:
                    accel.hand_gone()                   # bring your hand back: the dot is under your finger again

            now = time.monotonic()
            if target is None:
                target, spawned, hover_since = new_target(rng, last_target), now, None
            fill = 0.0
            if pointer is not None:
                px, py = C.to_window(pointer, W, H)
                x, y, w, h = target
                if x <= px <= x + w and y <= py <= y + h:
                    hover_since = hover_since or now
                    fill = (now - hover_since) / DWELL
                    if fill >= 1:
                        close_times.append(now - spawned)
                        last_target, target = target, None
                        fill = 0.0
                else:
                    hover_since = None
            if target is not None:
                draw_window(canvas, target, fill)
            if pointer is not None:
                cv2.circle(canvas, (px, py), 9, BLUE, 2)
                cv2.circle(canvas, (px, py), 3, WHITE, -1)

            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now
            avg = np.mean(close_times) if close_times else 0
            mode = (("MODEL" if predictor is not None else "PLAIN") + ("  RAW" if args.raw else "")
                    + ("" if accel is None else "  PRECISE"))
            cv2.putText(canvas, f"{mode}   closed {len(close_times)}/{args.windows}   avg {avg:.2f}s   "
                        f"{fps:.0f} fps{'' if pointer is not None else '   NO HAND'}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
            cv2.imshow("cursor lab", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()
        landmarker.close()

    mode = ("model" if predictor is not None else "plain") + ("_raw" if args.raw else "") + ("" if accel is None else "_precise")
    if glitches:
        print(f"Ignored {glitches} glitchy frames on screen (they're still in the recording, raw)")
    stamp_name = time.strftime("%Y%m%d_%H%M%S")
    if len(times) > 30:
        C.DATA_DIR.mkdir(exist_ok=True)
        out = C.DATA_DIR / f"session_{stamp_name}_{mode}.npz"
        np.savez_compressed(out, t=np.array(times), pts=np.array(frames))
        print(f"Saved {len(times)} frames of hand movement to data/{out.name}")
    if close_times:
        avg = float(np.mean(close_times))
        C.RUNS_DIR.mkdir(exist_ok=True)
        scores = C.RUNS_DIR / "scores.csv"
        new = not scores.exists()
        with open(scores, "a", newline="") as f:
            wr = csv.writer(f)
            if new:
                wr.writerow(["time", "mode", "seed", "windows", "avg_seconds"])
            wr.writerow([stamp_name, mode, args.seed, len(close_times), f"{avg:.3f}"])
        print(f"{mode}: closed {len(close_times)} windows, average {avg:.2f} s each (saved to runs/scores.csv)")


if __name__ == "__main__":
    main()
