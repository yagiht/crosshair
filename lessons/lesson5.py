import numpy as np
import torch
from torch import nn

d = np.load("data/session_20261028_005720_plain.npz")
tip = d["pts"][:, 8, :]

inputs = []
answers = []

for i in range(7, len(tip) - 2):
    history = tip[i-7 : i+1]
    answer = tip[i+2]

    if np.isnan(history).any() or np.isnan(answer).any():
        continue

    inputs.append(history.reshape(-1))
    answers.append(answer)

inputs = torch.tensor(np.array(inputs))
answers = torch.tensor(np.array(answers))
print(inputs.shape, answers.shape)

now = inputs[:, -2:]                         # where the finger is now
inputs = (inputs - now.repeat(1, 8)) * 10    # every past position, relative to now, ×10
answers = (answers - now) * 10               # how far it moves next, ×10

last = inputs[:, -2:]          # the most recent position (the last 2 of the 16 numbers)
prev = inputs[:, -4:-2]        # the position one frame before that

no_pred = last                          # opponent 1: "you'll stay where you are"
constant = last + (last - prev) * 2     # opponent 2: "you'll keep the same speed for 2 more frames"

def miss(guess):
    return (guess - answers).norm(dim=1).mean().item() * 1512

last = inputs[:, -2:]
print(last.shape)

split = int(len(inputs) * 0.8)
train_x, test_x = inputs[:split], inputs[split:]
train_y, test_y = answers[:split], answers[split:]
print(train_x.shape, test_x.shape)

model = nn.Sequential(
    nn.Linear(16, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 2),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for step in range(2000):
    guesses = model(train_x)
    loss = nn.functional.mse_loss(guesses, train_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        print(step, loss.item())

def px(guess, truth):
    return (guess - truth).norm(dim=1).mean().item() * 1512 / 10

with torch.no_grad():
    t_prev = test_x[:, -4:-2]                 # one frame back, relative to now
    print("TEST  no prediction: ", px(torch.zeros_like(test_y), test_y), "px")
    print("TEST  constant speed:", px(-t_prev * 2, test_y), "px")
    print("TEST  your model:    ", px(model(test_x), test_y), "px")