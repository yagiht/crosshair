# Crosshair

**Predicting hand motion to make webcam hand tracking precise and lag-free.**

Crosshair is an experiment in predicting where a fingertip will be 60 ms in the future, so a
hand-tracking cursor feels like it keeps up with your hand instead of trailing behind it.

It's the research side of [FIVES](https://github.com/yagiht/fives), a webcam hand-tracking interface
that replaces the mouse with gestures. FIVES tracks the hand at ~28 fps with ~21 ms of processing, so the
cursor always shows where the finger *was*. A prediction model is one way to close that gap.

> Status: **in progress.** The current best model is close to, but not yet better than, a simple
> constant-velocity baseline. Results below are reported honestly on held-out data.

## How it works

<img width="1902" height="1458" alt="Adobe Express - Screen Recording 2026-10-30 at 12 14 29 AM" src="https://github.com/user-attachments/assets/4868186a-1620-4f1c-b391-255800c651a4" />


```
webcam ─► MediaPipe (21 hand points) ─► fingertip track ─► last 8 frames ─► model ─► position 2 frames (~60 ms) ahead
```

1. **Collect data with a game.** `game.py` pops up small "windows" to close by hovering a fingertip
   over them, a mix of long fast moves and small precise ones. Every game records the raw MediaPipe
   hand landmarks (positions only, no images) to `data/`, and logs the average close time to
   `runs/scores.csv` as a usability benchmark.
2. **Build examples.** Each example is the fingertip's last 8 positions (16 numbers) and the answer is
   where it actually was 2 frames later. Frames where tracking was lost (NaN) are skipped.
3. **Train and evaluate** on a time-based split: first 80% of a session for training, last 20% for
   testing, so the model is always scored on movement it has never seen.
4. **Compare against baselines** that need no learning:
   - *no prediction*: "the finger stays where it is" (what a plain cursor does today)
   - *constant velocity*: "the finger keeps moving at its current speed"

The game's cursor uses the same pipeline as FIVES: One Euro filtering, a glitch guard against one-frame
tracking jumps, a PRISM-style precision pointer (slow hand = scaled-down motion, fast hand = 1:1), and
interpolation between camera frames. Only the on-screen dot is filtered; the recordings stay raw.

## Results so far

One session (~1 minute, 1,776 examples; 1,420 train / 356 test). Error = average distance between the
predicted and actual fingertip, in screen pixels (1512 px wide), **on the held-out test set**:

| Predictor | Test error |
|---|---|
| No prediction | 26.8 px |
| **Constant velocity (baseline to beat)** | **15.8 px** |
| Linear model, raw positions | 27.7 px |
| Linear model, normalized inputs | 17.3 px |
| MLP (16→64→64→2, ReLU) | 22.9 px |

What I learned:
- **Normalization mattered most.** Raw positions (~0.5) hide the useful signal, the tiny frame-to-frame
  movements (~0.02). Making every input relative to the current position and scaling ×10 cut the linear
  model's error from 27.7 to 17.3 px with no other change.
- **The bigger model overfit.** The MLP reached a much lower training loss (0.0034 vs 0.027) but did
  worse on the test set. With one minute of data, it memorized that session.
- **A low loss can hide a bad model.** On synthetic data, plain SGD reached a tiny loss while the
  motion weights were only half learned (they get ~10× smaller gradients); Adam fixed it.

**Next:** record 10–15 sessions, filter tracking glitches (teleports > 15% of the frame), retrain, and
test whether the model beats constant velocity, both in pixels and in real game times (`--model`).

## Run it

Needs Python **3.12** (MediaPipe doesn't support 3.13 yet) and a webcam. Tested on macOS.

```bash
git clone https://github.com/yagiht/crosshair.git
cd crosshair
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python game.py                  # play: point at the windows, hover to close (q quits)
python train.py                 # train on everything in data/
python game.py --model --seed 1 # play with the trained model steering
python game.py --seed 1         # same windows without it, to compare
```

`python game.py --help` lists the cursor options (`--raw`, `--no-glide`, `--no-precise`, `--margin`, …).
On macOS, allow your terminal to use the camera the first time.

## Files

| File | What it does |
|---|---|
| `game.py` | data-collection game and benchmark |
| `common.py` | camera, MediaPipe setup, shared settings |
| `features.py` | turns recordings into training examples |
| `model.py` | the prediction network |
| `train.py` | training and evaluation against the baselines |
| `fake_data.py` | synthetic recordings for testing the pipeline without a camera |
| `lessons/` | my step-by-step PyTorch learning scripts, from tensors to the experiments above |

## How this was built

I built this while learning PyTorch, with help from Claude (Anthropic's AI). The game and the training
pipeline (`game.py`, `common.py`, `features.py`, `model.py`, `train.py`) were written with Claude. The
`lessons/` scripts and the experiments in the results table are my own work, written step by step while
learning: tensors, training loops, batching, normalization, and diagnosing overfitting.
