import torch
from torch import nn

#neural network layer takes four inputs for 2 outputs
layer = nn.Linear(4,2)
# print(layer.weight)
# print(layer.bias)

#variable tensor
inputs = torch.tensor([0.50, 0.61, 0.05, 0.01])

#instantiates guess, so it will get two outputs from inputs
guess = layer(inputs)
print(guess)

#adds an answer so pyTorch knows how wrong it was
answer = torch.tensor([0.55, 0.62])

#this is the error.. the wrongness factor.
error = guess - answer
print(error)

#takes two inputs... subtracts, squares, then takes average.
loss = nn.functional.mse_loss(guess, answer)
print(loss)


optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)

for step in range(20):
    guess = layer(inputs)
    loss = nn.functional.mse_loss(guess, answer)

    optimizer.zero_grad()   # forget the last step's nudges
    loss.backward()         # work out each knob's nudge
    optimizer.step()        # nudge the knobs

    print(step, loss.item())

print("final guess:", layer(inputs))