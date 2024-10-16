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
env.seed(1)
torch.manual_seed(1)
writer = SummaryWriter('logfile/')

# Hyperparameters
learning_rate = 0.01
gamma = 0.99

class EpisodeData:
    def __init__(self):
        self.policy_history = []
        self.reward_episode = []

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

# policy = Policy()
# optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
# D_env = []

# Select Action
def select_action(policy,state,policy_history):
    state = torch.from_numpy(state).float()
    probs = policy(state)
    c = Categorical(probs)
    action = c.sample()

    # Add log probability of chosen action to history
    policy_history.append(c.log_prob(action))
    return action.item()

# Update Policy
def update_policy(episode, episode_data,loss_history,reward_history,optimizer):
    reward_episode= episode_data.reward_episode
    policy_history= episode_data.policy_history
    R = 0
    rewards = []

    # Discount future rewards back to the present using gamma
    for r in reward_episode[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)

    # Scale rewards
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    # Calculate loss
    loss = torch.sum(torch.stack(policy_history) * -rewards)

    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save and initialize episode history counters
    loss_history.append(loss.item())
    reward_history.append(np.sum(reward_episode))
    writer.add_scalar('data/reward', np.sum(reward_episode), episode)
    # policy.policy_history = []
    # policy.reward_episode = []

def gather_env_data(env, policy, D_env):
    state = env.reset()
    done = False
    episode_data = EpisodeData()
    while not done:
        action = select_action(policy,state, episode_data.policy_history)
        state, reward, done, _ = env.step(action)
        episode_data.reward_episode.append(reward)
    D_env.append(episode_data)


def mbpo(epochs):
    policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    D_env = []
    running_reward = 10
    for epoch in range(epochs):
        episode_data = EpisodeData()
        state = env.reset()
        done = False

        for time in range(1, 1001):
            action = select_action(policy,state,episode_data.policy_history)
            state, reward, done, _ = env.step(action)

            # Save reward
            episode_data.reward_episode.append(reward)
            if done:
                break

        # Used to determine when the environment is solved
        running_reward = (running_reward * 0.99) + (time * 0.01)

        update_policy(epoch, episode_data, policy.loss_history, policy.reward_history, optimizer)
        D_env.append(episode_data)
        # episode.policy_history = []
        # policy.reward_episode = []

        if episode % 50 == 0:
            print('Episode {}	Last length: {:5d}	Average length: {:.2f}'.format(episode, time, running_reward))

        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward, time))
            break
    writer.close()
    return policy, D_env

# Main Training Loop
# def main(episodes):
#     policy = Policy()
#     optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
#     D_env = []
#     running_reward = 10
#     for episode in range(episodes):
#         episode_data = EpisodeData()
#         state = env.reset()
#         done = False

#         for time in range(1, 1001):
#             action = select_action(policy,state,episode_data.policy_history)
#             state, reward, done, _ = env.step(action)

#             # Save reward
#             episode_data.reward_episode.append(reward)
#             if done:
#                 break

#         # Used to determine when the environment is solved
#         running_reward = (running_reward * 0.99) + (time * 0.01)

#         update_policy(episode, episode_data, policy.loss_history, policy.reward_history, optimizer)
#         D_env.append(episode_data)
#         # episode.policy_history = []
#         # policy.reward_episode = []

#         if episode % 50 == 0:
#             print('Episode {}	Last length: {:5d}	Average length: {:.2f}'.format(episode, time, running_reward))

#         if running_reward > env.spec.reward_threshold:
#             print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward, time))
#             break
#     writer.close()
#     return policy, D_env

# Run the training
episodes = 1000
policy,D_env = mbpo(episodes)

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