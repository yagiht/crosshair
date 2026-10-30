# import PyTorch Library
import torch

#creates frame, a list of numbers that was put in a tensor (a fancy container that pytorch can convert things in)
frame = torch.tensor([0.42, 0.61])
print(frame)

#prints the shape of frames
print(frame.shape)
#output: torch.Size([2]) one row, so no firrst number, then the length

frames = torch.tensor([[0.40, 0.60], [0.45, 0.60], [0.50, 0.61]])

#prints a grid... the inner brackets decide each row
print(frames)

print(frames.shape)
#print [3,2] as theres 3 rows, 2 length

# #step 7 (":" means all)
# print(frames[0])
# print(frames[2, 0])
# print(frames[:, 0])

print(frames * 100)

#n colon: after... is everything after the index of colon(include)
#n colon: before. is evefrything until inded of colon (don't include)
moves = frames[1:] - frames[:-1]
print(moves)
print(moves.shape)