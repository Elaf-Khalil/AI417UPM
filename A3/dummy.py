import random
import numpy as np
import torch as tr

seed = 4310030                     # <<<<<<<<<<<<<<<< Your UPM ID Goes Here
random.seed(seed)
np.random.seed(seed)
tr.manual_seed(seed)

class SimpleFeedForwardNet(tr.nn.Module):

  def __init__(self):
    super().__init__() 
    self.linear1 = tr.nn.Linear(784, 128, bias=True)       # 784 -> 16 - Layer 1 -- Affine Transformation (Linear with Bias)
    self.linear2 = tr.nn.Linear(128, 64, bias=True)       # 16 -> 16  - Layer 2 -- Linear Transformation (no bias)
    self.linear3 = tr.nn.Linear(64, 10, bias=True)        # 16 -> 10  - Layer 3 -- Affine Transformation (Linear with Bias)
    self.init_weights()

  def init_weights(self):
    tr.nn.init.xavier_uniform_(self.linear1.weight)
    tr.nn.init.xavier_uniform_(self.linear2.weight)
    tr.nn.init.xavier_uniform_(self.linear3.weight)
    tr.nn.init.zeros_(self.linear1.bias)
    tr.nn.init.zeros_(self.linear2.bias)
    tr.nn.init.zeros_(self.linear3.bias)
  
  def forward(self, x):
    x = self.linear1(x)
    x = self.linear2(x)
    x = self.linear3(x)
    return x

model = SimpleFeedForwardNet()                                          # Architecture
optimizer = tr.optim.SGD(model.parameters(), lr=0.01, maximize=False)   # Optimizer


loss_fn = tr.nn.CrossEntropyLoss()                                      # Objective Function [DO NOT CHANGE !]
