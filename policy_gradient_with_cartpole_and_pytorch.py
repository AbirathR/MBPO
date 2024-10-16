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
print("action_space: ", env.action_space.shape)
env.seed(1)
torch.manual_seed(1)
writer = SummaryWriter('logfile/')

# Hyperparameters
learning_rate = 0.01
gamma = 0.99

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n

        self.l1 = nn.Linear(self.state_space, 128, bias=False)
        self.l2 = nn.Linear(128, 128, bias=False)
        self.l3 = nn.Linear(128, self.action_space, bias=False)

        self.gamma = gamma

        # Episode policy and reward history
        self.policy_history = []
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.dropout(x, p=0.6, train=True)
        x = self.l3(x)
        return torch.softmax(x, dim=-1)

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

# Select Action
def select_action(state):
    state = torch.from_numpy(state).float()
    probs = policy(state)
    c = Categorical(probs)
    action = c.sample()

    # Add log probability of chosen action to history
    policy.policy_history.append(c.log_prob(action))
    return action.item()

# Update Policy
def update_policy(episode):
    R = 0
    rewards = []

    # Discount future rewards back to the present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0, R)

    # Scale rewards
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    # Calculate loss
    loss = torch.sum(torch.stack(policy.policy_history) * -rewards)

    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save and initialize episode history counters
    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.reward_episode))
    writer.add_scalar('data/reward', np.sum(policy.reward_episode), episode)
    policy.policy_history = []
    policy.reward_episode = []

# Main Training Loop
def main(episodes):
    running_reward = 10
    for episode in range(episodes):
        state = env.reset()
        done = False

        for time in range(1, 1001):
            action = select_action(state)
            state, reward, done, _ = env.step(action)

            # Save reward
            policy.reward_episode.append(reward)
            if done:
                break

        # Used to determine when the environment is solved
        running_reward = (running_reward * 0.99) + (time * 0.01)

        update_policy(episode)

        if episode % 50 == 0:
            print('Episode {}	Last length: {:5d}	Average length: {:.2f}'.format(episode, time, running_reward))

        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward, time))
            break
    writer.close()

# Run the training
episodes = 1000
main(episodes)

# Plot Results
window = int(episodes / 20)

fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9, 9])
rolling_mean = pd.Series(policy.reward_history).rolling(window).mean()
std = pd.Series(policy.reward_history).rolling(window).std()
ax1.plot(rolling_mean)
ax1.fill_between(range(len(policy.reward_history)), rolling_mean - std, rolling_mean + std, color='orange', alpha=0.2)
ax1.set_title('Episode Length Moving Average ({}-episode window)'.format(window))
ax1.set_xlabel('Episode')
ax1.set_ylabel('Episode Length')

ax2.plot(policy.reward_history)
ax2.set_title('Episode Length')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Episode Length')

fig.tight_layout(pad=2)
plt.show()