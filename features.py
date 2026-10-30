"""Turns recorded hand movement into training examples. Pure numpy, no camera.

One EXAMPLE = "here are the last HISTORY frames of your hand" (the input) plus
"this is where your fingertip actually was HORIZON seconds later" (the answer).
The answer is already in the recording, so no labeling by hand is needed.

Everything is measured RELATIVE to the fingertip's current spot, so the model
learns "how hands move", not "where your hand happens to be in the picture".
"""

import numpy as np

from common import INDEX_KNUCKLE, INDEX_TIP, MIDDLE_KNUCKLE

HISTORY = 8          # frames the model looks back at (about 0.27 s at 30 fps)
HORIZON = 0.06       # seconds into the future to predict (about 2 frames)
MAX_GAP = 0.12       # a bigger gap between frames means the hand was lost: don't bridge it

N_FEATURES = HISTORY * 4 + (HISTORY - 1)


def signal(pts):
    """(T, 21, 2) hand points -> (T, 4): fingertip x, y and knuckle-middle x, y."""
    tip = pts[:, INDEX_TIP]
    knuckles = (pts[:, INDEX_KNUCKLE] + pts[:, MIDDLE_KNUCKLE]) / 2
    return np.concatenate([tip, knuckles], axis=1)


def window_features(times, sig):
    """The input for ONE prediction. times: (HISTORY,), sig: (HISTORY, 4), oldest first."""
    now_tip = sig[-1, :2]
    rel = sig.copy()
    rel[:, 0:2] -= now_tip
    rel[:, 2:4] -= now_tip
    dts = np.diff(times)
    return np.concatenate([rel.reshape(-1), dts]).astype(np.float32)


def baseline_velocity(times, sig, horizon=HORIZON):
    """The simple non-ML guess: keep moving at the speed of the last two frames."""
    dt = max(times[-1] - times[-2], 1e-3)
    v = (sig[-1, :2] - sig[-2, :2]) / dt
    return v * horizon


def make_examples(t, pts, horizon=HORIZON):
    """All usable examples from one recording.

    t: (T,) seconds. pts: (T, 21, 2), with NaN rows where no hand was found.
    Returns X (N, N_FEATURES), Y (N, 2) = where the fingertip went, relative to now,
    and V (N, 2) = the constant-velocity guess (to compare the model against).
    """
    sig = signal(pts)
    valid = ~np.isnan(sig).any(axis=1)
    gaps_ok = np.concatenate([[False], np.diff(t) <= MAX_GAP]) & valid & np.concatenate([[False], valid[:-1]])
    # run[i] = how many frames in a row (ending at i) are valid and close together
    run = np.zeros(len(t), dtype=int)
    for i in range(len(t)):
        if valid[i]:
            run[i] = run[i - 1] + 1 if i and gaps_ok[i] else 1
    X, Y, V = [], [], []
    T = len(t)
    for i in range(HISTORY - 1, T):
        if run[i] < HISTORY:
            continue
        target_time = t[i] + horizon
        j = i + 1
        while j < T and t[j] < target_time and run[j] == run[j - 1] + 1:
            j += 1
        if j >= T or run[j] != run[j - 1] + 1:
            continue                                   # the hand was lost before the answer time
        a = (target_time - t[j - 1]) / max(t[j] - t[j - 1], 1e-6)
        future_tip = sig[j - 1, :2] + a * (sig[j, :2] - sig[j - 1, :2])
        w_t, w_s = t[i - HISTORY + 1:i + 1], sig[i - HISTORY + 1:i + 1]
        X.append(window_features(w_t, w_s))
        Y.append(future_tip - sig[i, :2])
        V.append(baseline_velocity(w_t, w_s, horizon))
    if not X:
        return (np.zeros((0, N_FEATURES), np.float32), np.zeros((0, 2), np.float32),
                np.zeros((0, 2), np.float32))
    return np.array(X), np.array(Y, np.float32), np.array(V, np.float32)
