import torch
from torch.autograd import Variable, Function
import numpy as np

# Create a Variable
x = Variable(torch.ones(2, 2), requires_grad=True)
print(x)

y = np.array([[0., 1., 3., 2., 5.],
             [9., 4., 2., 1., 5.]])

V = Variable(torch.from_numpy(y))
print(V)
V1 = V + 2
print(V1) # Created from an operations
          # Will have a grad_fn
print("grad_fn")
print(V1.grad_fn) # But why does it not have a grad_fn?

Z = V1 * V * 3
print(Z)
print(Z.mean())