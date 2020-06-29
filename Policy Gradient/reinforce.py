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
from torch.distributions import Categorical

from network import Net

env = gym.make('CartPole-v0').unwrapped

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.lin2 = nn.Linear(128,2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, t):
        t = self.lin1(t)
        t = self.dropout(t)
        t = F.relu(t)
        t = self.lin2(t)
        t = F.softmax(t, dim=1)
        return t

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = net(state)
    m = Categorical(probs)
    action = m.sample()
    net.saved_log_probs.append(m.log_prob(action))
    return action.item()

def compute_reward(returns):
    gamma = 0.99
    for i_act in range(len(net.saved_log_probs)):
        R = 0
        for i_cost in range(i_act, len(net.rewards)):
            R -= gamma**(i_cost-i_act)*net.rewards[i_cost]
        R += (i_cost-i_act)/2
        R = R*net.saved_log_probs[i_act]
        returns.append(R)
    net.saved_log_probs = []
    net.rewards = []
    return returns

def backprop(returns, batch_range):
    # returns = sum(returns)/batch_range
    reward_total = torch.cat(returns).sum()/batch_range
    # reward_total.requires_grad=True
    # returns = torch.tensor([returns])
    # returns.requires_grad = True
    print("total rewards = ", reward_total.item())
    reward_total.backward()
    optimizer.step()
    net.saved_log_probs = []
    net.rewards = []

net = Net()
optimizer = optim.Adam(net.parameters(), lr=1e-4)

def main():
    batch_range = 100
    for i_train in range(10000):
        returns = []
        for i_batch in range(batch_range):
            observation = env.reset()
            for env_ren in range(1000):
                action = select_action(observation)
                observation, reward, done, _ = env.step(action)
                env.render()
                net.rewards.append(reward)
                if done:
                    print("Episode finished after {} timesteps".format(env_ren+1), i_train)
                    break
            returns = compute_reward(returns)
        backprop(returns, batch_range)
        torch.save(net.state_dict(), 'policy_gradient.pth')

if __name__=='__main__':
    main()

env.close()
