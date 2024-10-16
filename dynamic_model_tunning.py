from dynamic_model import DynamicsModel
import pickle
from utils import ReplayBuffer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import gym

# get dataset
print("Loading dataset...")
replay_buffer_env = pickle.load(open('replay_buffer.pkl', 'rb'))
print("Dataset loaded")
print(replay_buffer_env)

# split dataset
print("Splitting dataset...")
replay_buffer_train = ReplayBuffer(len(replay_buffer_env))
replay_buffer_test = ReplayBuffer(len(replay_buffer_env))
# replay_buffer_train.buffer = replay_buffer_env.buffer[:int(0.8*len(replay_buffer_env))]
# replay_buffer_test.buffer = replay_buffer_env.buffer[int(0.8*len(replay_buffer_env)):]
for i in range(len(replay_buffer_env)):
    if i<0.8*len(replay_buffer_env):
        replay_buffer_train.add(replay_buffer_env.buffer[i])
    else:
        replay_buffer_test.add(replay_buffer_env.buffer[i])
print("Dataset split")

env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
action_dim = 1
dynamics_model = DynamicsModel(obs_dim, action_dim)
BATCH_SIZE = 256
dynamics_optimizer = optim.Adam(dynamics_model.parameters(), lr=3e-4)



# train model
print("Training model...")
num_epochs = 1000
for epoch in range(num_epochs):
    if len(replay_buffer_train) > BATCH_SIZE:
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = replay_buffer_train.sample(BATCH_SIZE)
        predicted_next_obs = dynamics_model(obs_batch, action_batch)
        dynamics_loss = F.mse_loss(predicted_next_obs, next_obs_batch)
        dynamics_optimizer.zero_grad()
        dynamics_loss.backward()
        dynamics_optimizer.step()
print("Model trained")

# test model
print("Testing model...")
# on test set
total_loss = 0
for epoch in range(10):
    if len(replay_buffer_test) > BATCH_SIZE:
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = replay_buffer_test.sample(BATCH_SIZE)
        predicted_next_obs = dynamics_model(obs_batch, action_batch)
        dynamics_loss = F.mse_loss(predicted_next_obs, next_obs_batch)
        total_loss += dynamics_loss.item()
print("Test loss: ", total_loss/10)

# save model
print("Saving model...")
torch.save(dynamics_model.state_dict(), 'dynamics_model.pth')
