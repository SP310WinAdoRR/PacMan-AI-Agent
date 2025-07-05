import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pacman_env import PacManRetroEnv, CNNPolicy  # Assume your env+policy code is in pacman_env.py

# ============ Replay Buffer ============
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Remove extra batch dimension if present
        state = np.squeeze(state, axis=0) if state is not None and state.shape[0] == 1 else state
        next_state = np.squeeze(next_state, axis=0) if next_state is not None and next_state.shape[0] == 1 else next_state
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)

# ============ Hyperparameters ============
EPISODES = 500
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 64
MEMORY_CAPACITY = 10000
TARGET_UPDATE_FREQ = 10
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 5000

# ============ Main Training ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = PacManRetroEnv(render_mode=None)
policy_net = CNNPolicy(env.action_space).to(device)
target_net = CNNPolicy(env.action_space).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayBuffer(MEMORY_CAPACITY)

steps_done = 0

def select_action(state):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if random.random() > eps_threshold:
        with torch.no_grad():
            q_values = policy_net(state)
            return q_values.argmax(1).item()
    else:
        return random.randrange(env.action_space.n)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

    states = states.to(device)
    actions = actions.unsqueeze(1).to(device)
    rewards = rewards.unsqueeze(1).to(device)
    next_states = next_states.to(device)
    dones = dones.unsqueeze(1).to(device)

    q_values = policy_net(states).gather(1, actions)
    with torch.no_grad():
        max_next_q = target_net(next_states).max(1)[0].unsqueeze(1)
        expected_q = rewards + (1 - dones) * GAMMA * max_next_q

    loss = nn.MSELoss()(q_values, expected_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ============ Training Loop ============
for episode in tqdm(range(EPISODES), desc="Training"):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    total_reward = 0

    for t in range(2000):  # Reduce max steps per episode for faster training
        action = select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)

        next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

        done = terminated or truncated
        memory.push(state.cpu().numpy(), action, reward, next_state_tensor.cpu().numpy(), done)

        state = next_state_tensor
        total_reward += reward

        # Print reward for debugging
        if reward > 0:
            print(f"\n[DEBUG] Reward: {reward} (Episode {episode}, Step {t})")

        optimize_model()

        if done:
            break

    if episode % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode:3d} | Total Reward: {total_reward:5.1f}")

env.close()
torch.save(policy_net.state_dict(), "dqn_pacman_policy.pth")
print("🏁 Training finished and policy saved.")
