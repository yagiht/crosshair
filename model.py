"""The prediction model: a small neural network (a "multi-layer perceptron").

In:  the last few frames of your fingertip and knuckles (from features.py).
Out: how far your fingertip will move in the next HORIZON seconds (x, y).

It is tiny on purpose (about 13,000 numbers), so it runs in well under a
millisecond on any laptop.
"""

import torch
from torch import nn

from features import HORIZON, HISTORY, N_FEATURES


class CursorPredictor(nn.Module):
    def __init__(self, hidden=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2),
        )
        # Inputs and answers are tiny numbers (fractions of the picture). Neural nets
        # learn best with numbers around 1, so we rescale using averages measured on the
        # training data. Stored as "buffers" so they're saved inside the model file.
        self.register_buffer("x_mean", torch.zeros(N_FEATURES))
        self.register_buffer("x_std", torch.ones(N_FEATURES))
        self.register_buffer("y_std", torch.ones(2))

    def forward(self, x):
        z = (x - self.x_mean) / self.x_std
        return self.net(z) * self.y_std


def save(model, path, info):
    torch.save({"state": model.state_dict(), "hidden": model.net[0].out_features,
                "history": HISTORY, "horizon": HORIZON, "info": info}, path)


def load(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt["history"] != HISTORY or abs(ckpt["horizon"] - HORIZON) > 1e-9:
        raise SystemExit(f"{path} was trained with HISTORY={ckpt['history']}, HORIZON={ckpt['horizon']}, "
                         f"but features.py now says HISTORY={HISTORY}, HORIZON={HORIZON}. Retrain.")
    model = CursorPredictor(ckpt["hidden"])
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, ckpt["info"]
