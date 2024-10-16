import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# Set up the CartPole environment
env = gym.make("CartPole-v1")

# Hyperparameters
GAMMA = 0.99
TAU = 0.005
LR = 3e-4
DM_LR = 1e-4
BUFFER_SIZE = 100000
BATCH_SIZE = 256
ALPHA = 0.2
MODEL_ROLLOUTS = 400
K_STEP = 5
GRAD_UPDATES = 10

# Neural network for Q-function
class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Neural network for the policy
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)  # Stabilize log_std values
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)  # Squash action
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

# Neural network for the dynamics model
class DynamicsModel(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(DynamicsModel, self).__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, obs_dim)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

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
            torch.FloatTensor(np.array(obs)),
            torch.FloatTensor(np.array(action)).unsqueeze(-1) if action.ndim == 1 else torch.FloatTensor(action),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(next_obs),
            torch.FloatTensor(done).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)

# Initialize networks and optimizers
obs_dim = env.observation_space.shape[0]
action_dim = 1
policy_net = PolicyNetwork(obs_dim, action_dim)
q_net1 = QNetwork(obs_dim, action_dim)
q_net2 = QNetwork(obs_dim, action_dim)
q_net1_target = QNetwork(obs_dim, action_dim)
q_net2_target = QNetwork(obs_dim, action_dim)
q_net1_target.load_state_dict(q_net1.state_dict())
q_net2_target.load_state_dict(q_net2.state_dict())
dynamics_model = DynamicsModel(obs_dim, action_dim)
# load model
dynamics_model.load_state_dict(torch.load('dynamics_model.pth'))

policy_optimizer = optim.Adam(policy_net.parameters(), lr=LR)
q_optimizer1 = optim.Adam(q_net1.parameters(), lr=LR)
q_optimizer2 = optim.Adam(q_net2.parameters(), lr=LR)
dynamics_optimizer = optim.Adam(dynamics_model.parameters(), lr=DM_LR)

replay_buffer_env = ReplayBuffer(BUFFER_SIZE)
replay_buffer_model = ReplayBuffer(BUFFER_SIZE)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Train dynamics model
    if len(replay_buffer_env) > BATCH_SIZE:
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = replay_buffer_env.sample(BATCH_SIZE)
        predicted_next_obs = dynamics_model(obs_batch, action_batch)
        dynamics_loss = F.mse_loss(predicted_next_obs, next_obs_batch)
        dynamics_optimizer.zero_grad()
        dynamics_loss.backward()
        dynamics_optimizer.step()

    # Interact with the environment
    obs = env.reset()
    done = False
    while not done:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        action, _ = policy_net.sample(obs_tensor)
        action = action.detach().cpu().numpy()[0]
        next_obs, reward, done, _ = env.step(int(action.item() > 0))
        replay_buffer_env.add((obs, action, reward, next_obs, float(done)))
        obs = next_obs

    # Model rollouts
    for _ in range(MODEL_ROLLOUTS):
        if len(replay_buffer_env) > 0:
            obs = random.choice(replay_buffer_env.buffer)[0]
            for _ in range(K_STEP):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action, _ = policy_net.sample(obs_tensor)
                action = action.detach()
                next_obs = dynamics_model(obs_tensor, action).detach().cpu().numpy()[0]
                reward = 1.0 if abs(next_obs[0]) < 2.4 else 0.0  # Reward based on CartPole conditions
                done = float(abs(next_obs[0]) >= 2.4)
                replay_buffer_model.add((obs, action.numpy(), reward, next_obs, done))
                if done:
                    break
                obs = next_obs

    # Update policy and Q networks
    for _ in range(GRAD_UPDATES):
        if len(replay_buffer_model) > BATCH_SIZE:
            obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = replay_buffer_model.sample(BATCH_SIZE)

            # Update Q networks
            with torch.no_grad():
                next_action, next_log_prob = policy_net.sample(next_obs_batch)
                q_target = torch.min(
                    q_net1_target(next_obs_batch, next_action),
                    q_net2_target(next_obs_batch, next_action),
                )
                q_target = reward_batch + GAMMA * (1 - done_batch) * (q_target - ALPHA * next_log_prob)

            q_loss1 = F.mse_loss(q_net1(obs_batch, action_batch), q_target)
            q_loss2 = F.mse_loss(q_net2(obs_batch, action_batch), q_target)
            q_optimizer1.zero_grad()
            q_optimizer2.zero_grad()
            q_loss1.backward()
            q_loss2.backward()
            q_optimizer1.step()
            q_optimizer2.step()

            # Update policy network
            new_action, log_prob = policy_net.sample(obs_batch)
            q_new_action = torch.min(q_net1(obs_batch, new_action), q_net2(obs_batch, new_action))
            policy_loss = (ALPHA * log_prob - q_new_action).mean()
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            # Soft update target networks
            for target_param, param in zip(q_net1_target.parameters(), q_net1.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

            for target_param, param in zip(q_net2_target.parameters(), q_net2.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

    print(f"Epoch {epoch + 1} completed.")

env.close()