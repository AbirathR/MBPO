import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Hyperparameters
E = 1000   # Environment steps per epoch
M = 400    # Model rollouts per epoch
B = 1      # Ensemble size (not used as ensemble size is 1)
G = 40     # Policy updates per epoch
k = 1      # Model horizon
hidden_size = 200  # Hidden layer size
num_hidden_layers = 4  # Number of hidden layers

# Dynamics Model
class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DynamicsModel, self).__init__()
        layers = [nn.Linear(state_dim + action_dim, hidden_size), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, state_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        next_state = self.model(x)
        return next_state

# Policy Network (Actor)
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        layers = [nn.Linear(state_dim, hidden_size), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers += [nn.Linear(hidden_size, action_dim), nn.Tanh()]  # For continuous actions in [-1, 1]
        self.model = nn.Sequential(*layers)

    def forward(self, state):
        action = self.model(state)
        return action

# Value Network (Critic)
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        layers = [nn.Linear(state_dim, hidden_size), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, state):
        value = self.model(state)
        return value

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)

# Training Loop
env = gym.make('CartPole-v1')

state_dim = env.observation_space.shape[0]
action_dim = 1  # We'll discretize actions for CartPole

# Initialize models
dynamics_model = DynamicsModel(state_dim, action_dim)
policy_net = PolicyNetwork(state_dim, action_dim)
value_net = ValueNetwork(state_dim)

# Optimizers
dynamics_optimizer = optim.Adam(dynamics_model.parameters(), lr=1e-3)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
value_optimizer = optim.Adam(value_net.parameters(), lr=1e-3)

# Replay buffers
real_buffer = ReplayBuffer()
model_buffer = ReplayBuffer()

num_epochs = 1000

episode_rewards = []

for epoch in range(num_epochs):
    # Collect E environment steps
    total_env_steps = 0
    epoch_rewards = []
    while total_env_steps < E:
        state = env.reset()
        done = False
        episode_reward = 0
        while not done and total_env_steps < E:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = policy_net(state_tensor).detach().numpy()[0]
            # For CartPole, actions are discrete (0 or 1)
            action = 0 if action < 0 else 1

            next_state, reward, done, _ = env.step(action)
            real_buffer.push(state, [action], reward, next_state, done)
            episode_reward += reward
            total_env_steps += 1

            state = next_state

        epoch_rewards.append(episode_reward)

    # Train dynamics model with real data
    dynamics_epochs = 50  # Number of epochs to train the dynamics model
    dynamics_batch_size = 256
    for _ in range(dynamics_epochs):
        if len(real_buffer) >= dynamics_batch_size:
            states, actions, rewards, next_states, dones = real_buffer.sample(dynamics_batch_size)
            states = torch.FloatTensor(states)
            actions = torch.FloatTensor(actions)
            next_states = torch.FloatTensor(next_states)

            predicted_next_states = dynamics_model(states, actions)
            loss = nn.MSELoss()(predicted_next_states, next_states)
            dynamics_optimizer.zero_grad()
            loss.backward()
            dynamics_optimizer.step()

    # Generate synthetic data using the dynamics model
    if len(real_buffer) >= M:
        model_buffer = ReplayBuffer()  # Clear model buffer each epoch
        # Sample M initial states from real buffer
        initial_states, _, _, _, _ = real_buffer.sample(M)
        initial_states = torch.FloatTensor(initial_states)

        state = initial_states
        for _ in range(k):
            action = policy_net(state).detach()
            next_state = dynamics_model(state, action).detach()

            # Placeholder rewards and dones
            reward = torch.ones((M, 1))  # Assuming reward of 1 for CartPole
            done = torch.zeros((M, 1))   # Assuming not done

            model_buffer.push(state.numpy(), action.numpy(), reward.numpy(), next_state.numpy(), done.numpy())

            state = next_state

    # Train policy with synthetic data
    policy_batch_size = 256
    for _ in range(G):
        if len(model_buffer) >= policy_batch_size:
            states, actions, rewards, next_states, dones = model_buffer.sample(policy_batch_size)
            states = torch.FloatTensor(states)
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)

            # Update value network
            target_values = rewards + 0.99 * value_net(next_states) * (1 - dones)
            values = value_net(states)
            value_loss = nn.MSELoss()(values, target_values.detach())
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

            # Update policy network
            policy_loss = -value_net(states).mean()
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

    # Log progress
    avg_reward = np.mean(epoch_rewards)
    episode_rewards.append(avg_reward)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Average Reward: {avg_reward}")

# Close the environment
env.close()

# Visualization of rewards
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards)
plt.xlabel('Epoch')
plt.ylabel('Average Reward')
plt.title('MBPO on CartPole-v1')
plt.grid(True)
plt.show()
