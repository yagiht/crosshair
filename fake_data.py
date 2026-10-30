"""Makes FAKE hand recordings, so you can test train.py without a camera.

    python fake_data.py        # writes data/fake_*.npz
    python train.py            # should train fine and beat "no prediction"

Fake data is only for checking that the code runs. Delete data/fake_*.npz before
training for real, because a model trained on fake hands won't help your real hand.
"""

import numpy as np

from common import DATA_DIR, INDEX_KNUCKLE, INDEX_TIP, MIDDLE_KNUCKLE


def min_jerk(p0, p1, n):
    """A smooth human-like reach from p0 to p1 over n frames."""
    s = np.linspace(0, 1, n)[:, None]
    return p0 + (p1 - p0) * (10 * s**3 - 15 * s**4 + 6 * s**5)


def session(rng, seconds=180):
    fps = 30
    path = []
    pos = np.array([0.5, 0.5])
    while len(path) < seconds * fps:
        goal = rng.uniform(0.2, 0.8, 2)
        n = int(rng.uniform(0.25, 0.9) * fps)
        over = goal + (goal - pos) * rng.uniform(0, 0.12)            # slight overshoot...
        path += list(min_jerk(pos, over, n))
        path += list(min_jerk(over, goal, int(0.2 * fps)))           # ...then correct
        path += [goal + rng.normal(0, 0.002, 2) for _ in range(int(rng.uniform(0.1, 0.6) * fps))]
        pos = goal
    path = np.array(path)
    T = len(path)
    t = np.cumsum(rng.normal(1 / fps, 0.003, T).clip(0.02, 0.05))
    tip = path + rng.normal(0, 0.0015, path.shape)                   # tracking wobble
    pts = np.repeat(tip[:, None, :], 21, axis=1)
    knuckles = path + np.array([0.0, 0.12]) + rng.normal(0, 0.001, path.shape)
    pts[:, INDEX_KNUCKLE] = knuckles + [-0.02, 0]
    pts[:, MIDDLE_KNUCKLE] = knuckles + [0.02, 0]
    pts[:, INDEX_TIP] = tip
    for start in rng.integers(0, T - 20, 6):                         # hand lost a few times
        pts[start:start + rng.integers(3, 15)] = np.nan
    return t, pts.astype(np.float32)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    DATA_DIR.mkdir(exist_ok=True)
    for k in range(3):
        t, pts = session(rng)
        np.savez_compressed(DATA_DIR / f"fake_{k}.npz", t=t, pts=pts)
    print("Wrote 3 fake recordings to data/. Now run:  python train.py")
