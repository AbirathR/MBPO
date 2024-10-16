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
hidden_size = 128  # Hidden layer size
learning_rate = 1e-2

gamma = 0.99  # Discount factor

# Dynamics Model
class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DynamicsModel, self).__init__()
        self.state_space = state_dim
        self.action_space = action_dim
        self.l1 = nn.Linear(self.state_space + 1, 128, bias=False)
        self.l2 = nn.Linear(128, 128, bias=False)
        self.l3 = nn.Linear(128, self.state_space, bias=False)

    def forward(self, state, action):
        action = action.view(-1, 1)  # Ensure action has the correct shape
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.l1(x))
        x = torch.dropout(x, p=0.6, train=True)
        next_state = self.l3(x)
        return next_state

# Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.state_space = state_dim
        self.action_space = action_dim
        self.l1 = nn.Linear(self.state_space, 128, bias=False)
        self.l2 = nn.Linear(128, 128, bias=False)
        self.l3 = nn.Linear(128, self.action_space, bias=False)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.dropout(x, p=0.6, train=True)
        action = torch.softmax(self.l3(x), dim=-1)
        return action

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
action_dim = env.action_space.n

# Initialize models
dynamics_model = DynamicsModel(state_dim, action_dim)
policy_net = PolicyNetwork(state_dim, action_dim)

# Optimizers
dynamics_optimizer = optim.Adam(dynamics_model.parameters(), lr=learning_rate)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

# Replay buffers
real_buffer = ReplayBuffer()
model_buffer = ReplayBuffer()

num_epochs = 1000

# Policy history and reward history for policy gradient
policy_history = []
reward_episode = []

# Training Loop
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
            action_probs = policy_net(state_tensor)
            action_distribution = torch.distributions.Categorical(action_probs)
            action = action_distribution.sample().item()

            next_state, reward, done, _ = env.step(action)
            real_buffer.push(state, [action], reward, next_state, done)
            reward_episode.append(reward)
            policy_history.append(action_distribution.log_prob(torch.tensor(action)))
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
            actions = torch.FloatTensor(actions).view(-1, 1)  # Adjust action dimensions to match the model input
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
            action = torch.argmax(action, dim=1).view(-1, 1)  # Adjust action dimensions to match the model input
            next_state = dynamics_model(state, action).detach()

            # Placeholder rewards and dones
            reward = torch.ones((M, 1))  # Assuming reward of 1 for CartPole
            done = torch.zeros((M, 1))   # Assuming not done

            model_buffer.push(state.numpy(), action.numpy(), reward.numpy(), next_state.numpy(), done.numpy())

            state = next_state

    # Update policy using policy gradient method
    R = 0
    rewards = []
    # Discount future rewards back to the present using gamma
    for r in reward_episode[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)

    # Scale rewards
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    # Calculate policy gradient loss
    policy_loss = []
    for log_prob, reward in zip(policy_history, rewards):
        policy_loss.append(-log_prob * reward)
    policy_loss = torch.cat(policy_loss).sum()

    # Update policy network
    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    # Clear history
    policy_history = []
    reward_episode = []

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
plt.title('MBPO with Policy Gradient on CartPole-v1')
plt.grid(True)
plt.show()