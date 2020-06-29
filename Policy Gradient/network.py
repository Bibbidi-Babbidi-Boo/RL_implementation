import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(4, 32)
        self.lin2 = nn.Linear(32,16)
        self.lin3 = nn.Linear(16,2)
    def forward(self, t):
        t = self.lin1(t)
        t = F.relu(t)
        t = self.lin2(t)
        t = F.relu(t)
        t = self.lin3(t)
        # t = F.relu(t)
        t = nn.Softmax(t)
        return t

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
