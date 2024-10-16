import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

# Set up environment and random seed
env = gym.make('CartPole-v1')
env.reset(seed=1)
torch.manual_seed(1)
writer = SummaryWriter('logfile/')

# Hyperparameters
learning_rate = 0.01
gamma = 0.99
model_rollouts = 10
k_step = 1
use_only_D_env = True

# Define Policy Network
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n

        self.l1 = nn.Linear(self.state_space, 128, bias=False)
        self.l2 = nn.Linear(128, 128, bias=False)
        self.l3 = nn.Linear(128, self.action_space, bias=False)

    def forward(self, x):
        # x = torch.relu(self.l1(x))
        # x = torch.relu(self.l2(x))
        # x = self.l3(x)
        # return torch.softmax(x, dim=-1)
        x = torch.relu(self.l1(x))
        x = torch.dropout(x, p=0.6, train=True)
        x = self.l3(x)
        return torch.softmax(x, dim=-1)

# Define Predictive Model
class PredictiveModel(nn.Module):
    def __init__(self):
        super(PredictiveModel, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n

        self.l1 = nn.Linear(self.state_space + 1, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, self.state_space + 1)  # Predict next state and reward

    def forward(self, state, action):
        if action.dim() == 1:
            action = action.unsqueeze(-1)  # Ensure action has batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension if missing
        x = torch.cat((state, action), dim=-1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        next_state = x[:, :-1]
        reward = x[:, -1]
        return next_state, reward

policy = Policy()
predictive_model = PredictiveModel()
optimizer_policy = optim.Adam(policy.parameters(), lr=learning_rate)
optimizer_model = optim.Adam(predictive_model.parameters(), lr=learning_rate)

# Datasets to store environment and model data
D_env = []
D_model = []

# Select Action
def select_action(state):
    state = torch.from_numpy(state).float()
    probs = policy(state)
    c = Categorical(probs)
    action = c.sample()
    return action.item(), c.log_prob(action)

# Train Predictive Model
def train_model():
    if len(D_env) < 100:
        return
    batch = np.random.choice(len(D_env), 64)
    states, actions, next_states, rewards = zip(*[D_env[i] for i in batch])

    states = torch.tensor(np.array(states), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(-1)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)

    pred_next_states, pred_rewards = predictive_model(states, actions)
    loss = nn.MSELoss()(pred_next_states, next_states) + nn.MSELoss()(pred_rewards, rewards)

    optimizer_model.zero_grad()
    loss.backward()
    optimizer_model.step()

# Update Policy
def update_policy():
    if(use_only_D_env):
        
        if len(D_env) < 100:
            return
        batch = np.random.choice(len(D_env[-100:]), 64)
        states, actions, _,rewards = zip(*[D_env[i-100] for i in batch])
    else:
        if len(D_model) < 100:
            return
        batch = np.random.choice(len(D_model), 64)
        states, actions, rewards = zip(*[D_model[i] for i in batch])

    states = torch.tensor(np.array(states), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32)

    log_probs = []
    for i in range(len(states)):
        _, log_prob = select_action(states[i].numpy())
        log_probs.append(log_prob)

    loss = -torch.sum(torch.stack(log_probs) * rewards)

    optimizer_policy.zero_grad()
    loss.backward()
    optimizer_policy.step()

# Gather initial data with random policy
state = env.reset()
done = False
while len(D_env) < 1000:
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    D_env.append((state, action, next_state, reward))
    state = next_state
    if done:
        state = env.reset()


# Main Training Loop
def main(epochs):
    no_of_episodes = 0
    for epoch in range(epochs):
        # Train model on real environment data
        train_model()

        # Interact with the environment
        state = env.reset()
        done = False
        while not done:
            action, _ = select_action(state)
            next_state, reward, done, _ = env.step(action)
            # print("reward:", reward)
            D_env.append((state, action, next_state, reward))
            state = next_state
        no_of_episodes += 1

        # Model rollouts
        for _ in range(model_rollouts):
            state = np.array(D_env[np.random.choice(len(D_env))][0])
            for _ in range(k_step):
                action, _ = select_action(state)
                action_tensor = torch.tensor([action], dtype=torch.float32).unsqueeze(0)
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                next_state, reward = predictive_model(state_tensor, action_tensor)
                D_model.append((state, action, reward.item()))
                state = next_state.detach().numpy()[0]

        # Update policy
        update_policy()

        # Log progress
        if epoch % 10 == 0:
            print(f'Epoch {epoch} completed')
            print("D_env length:", len(D_env))
            print(f"epoch avg reward: {np.sum([x[3] for x in D_env]) / no_of_episodes}")

# Run the training
main(epochs=1000)

# Plot Results
# (This part would require modifying to plot specific results from MDPO, e.g., model loss, reward history, etc.)