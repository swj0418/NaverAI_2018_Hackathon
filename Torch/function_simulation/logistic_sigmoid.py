import matplotlib.pyplot as plt
import numpy as np


inputVector = np.random.randn(2000)
outputVector = []

def Logit_Sigmoid(x : np.float32):
    return 1. / (1. + np.exp(-x))
google
for i in range(inputVector.size):
    outputVector.append(Logit_Sigmoid(inputVector[i]))
    print(Logit_Sigmoid(inputVector[i]))

plt.plot(inputVector)
plt.yscale('logit')
plt.show()