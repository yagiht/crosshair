import torch
from torch import nn

torch.manual_seed(0)


positions = torch.rand(200, 2)
moves = (torch.rand(200, 2) - 0.5) * 0.2

inputs = torch.cat([positions, moves], dim=1)
answers = positions + moves

# print(inputs.shape, answers.shape)
# print(inputs[0], answers[0])


layer = nn.Linear(4, 2)
optimizer = torch.optim.Adam(layer.parameters(), lr=0.05)


for step in range(500):
    guesses = layer(inputs)
    loss = nn.functional.mse_loss(guesses, answers)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(step, loss.item())


test = torch.tensor([0.20, 0.30, 0.05, -0.02])
print("test guess:", layer(test))
print("true answer: 0.25, 0.28")

print(layer.weight)
print(layer.bias)