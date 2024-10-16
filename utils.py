import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# Replay buffer for storing experience
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done = zip(*transitions)
        return (
            torch.FloatTensor(obs),
            torch.FloatTensor(action),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(next_obs),
            torch.FloatTensor(done).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)