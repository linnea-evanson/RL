import torch

t = [i for i in range(100)]
b = torch.tensor([t,t,t,t])
print(b)
print(b.shape)

b= b.t()  #transpose
print("b.t", b)
print(b.shape)
reshaped = b.reshape(10,10,4)   #reshape
print(reshaped)
print(reshaped.shape)

reshaped = torch.flip(reshaped, [0,1])   #flip (change order)
print(reshaped)

import matplotlib.pyplot as plt
import numpy as np
#plt.plot([i for i in range(200)], [0.5/(i/100+1) for i in range(200)])
#plt.show()

plt.plot([1000*np.exp(-(x)/100) for x in range(100)])
plt.show()

#plt.plot([1/(2*x+1) for x in range(100)])
#plt.show()