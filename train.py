"""Train the cursor predictor on everything in data/.

    python train.py               # train and save runs/cursor_predictor.pt
    python train.py --epochs 100  # train longer

It prints how far off three guesses are, on recordings the model never trained on:
  no prediction   = where the cursor is today (always HORIZON seconds behind)
  constant speed  = "keep moving the way you were" (no ML)
  model           = the neural network
Lower is better. If the model doesn't beat "constant speed", record more varied data.
"""

import argparse
import time

import numpy as np
import torch

import features as F
from common import DATA_DIR, MARGIN, RUNS_DIR
from model import CursorPredictor, save

SCREEN_POINTS = 1512   # MacBook screen width in points, only used to make errors easy to picture


def to_points(err):
    """Error in picture fractions -> approximate screen points (same mapping as the cursor)."""
    return err * SCREEN_POINTS / (1 - 2 * MARGIN)


def mean_error(pred, true):
    return float(np.mean(np.linalg.norm(pred - true, axis=1)))


def load_data():
    files = sorted(DATA_DIR.glob("*.npz"))
    if not files:
        raise SystemExit("No recordings in data/. Run  python game.py  first "
                         "(or  python fake_data.py  to test the pipeline).")
    train, val = [], []
    for f in files:
        d = np.load(f)
        X, Y, V = F.make_examples(d["t"], d["pts"])
        if len(X) < 50:
            print(f"  {f.name}: only {len(X)} examples, skipping")
            continue
        cut = int(len(X) * 0.8)        # the last 20% of each recording is the test set
        train.append((X[:cut], Y[:cut], V[:cut]))
        val.append((X[cut:], Y[cut:], V[cut:]))
        print(f"  {f.name}: {len(X)} examples")
    if not train:
        raise SystemExit("Not enough usable data yet. Record a longer session.")
    stack = lambda parts: tuple(np.concatenate(a) for a in zip(*parts))
    return stack(train), stack(val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    torch.manual_seed(0)

    print("Loading recordings:")
    (Xtr, Ytr, Vtr), (Xva, Yva, Vva) = load_data()
    print(f"{len(Xtr)} training examples, {len(Xva)} test examples\n")

    model = CursorPredictor(args.hidden)
    model.x_mean.copy_(torch.tensor(Xtr.mean(0)))
    model.x_std.copy_(torch.tensor(Xtr.std(0) + 1e-6))
    model.y_std.copy_(torch.tensor(Ytr.std(0) + 1e-6))

    xtr, ytr = torch.tensor(Xtr), torch.tensor(Ytr)
    xva = torch.tensor(Xva)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = torch.nn.SmoothL1Loss()
    y_scale = model.y_std

    best, best_state = float("inf"), None
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(xtr))
        for k in range(0, len(order), 256):
            idx = order[k:k + 256]
            loss = loss_fn(model(xtr[idx]) / y_scale, ytr[idx] / y_scale)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            err = mean_error(model(xva).numpy(), Yva)
        if err < best:
            best, best_state = err, {k: v.clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:3d}   test error {to_points(err):6.1f} pts")

    model.load_state_dict(best_state)
    hold = mean_error(np.zeros_like(Yva), Yva)
    vel = mean_error(Vva, Yva)
    print(f"\nTrained in {time.time() - start:.0f} s. Average distance from where your finger actually went "
          f"{F.HORIZON * 1000:.0f} ms later:")
    print(f"  no prediction   {to_points(hold):6.1f} pts")
    print(f"  constant speed  {to_points(vel):6.1f} pts")
    print(f"  model           {to_points(best):6.1f} pts   ({100 * (1 - best / hold):.0f}% less lag than no prediction)")
    if best >= vel:
        print("\nThe model isn't beating 'constant speed' yet. Record more, and mix slow, fast, "
              "and tiny movements.")

    RUNS_DIR.mkdir(exist_ok=True)
    out = RUNS_DIR / "cursor_predictor.pt"
    save(model, out, {"test_error": best, "hold_error": hold, "velocity_error": vel,
                      "examples": len(Xtr)})
    print(f"\nSaved {out.relative_to(out.parent.parent)}. Try it:  python game.py --model")


if __name__ == "__main__":
    main()
