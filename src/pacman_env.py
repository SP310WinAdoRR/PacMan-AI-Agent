import gymnasium as gym
import retro
import numpy as np
import cv2
import torch
import torch.nn as nn
from gymnasium import spaces
from retro.examples.discretizer import Discretizer


class SimpleDiscretizer(Discretizer):
    def __init__(self, env):
        # Just define the combos (actions) directly
        combos = [
            ["UP"],
            ["DOWN"],
            ["LEFT"],
            ["RIGHT"]
        ]
        super().__init__(env=env, combos=combos)

# -----------------------
# 1. Custom Gym Env
# -----------------------
class PacManRetroEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, game="MsPacMan-Nes", render_mode="human"):
        super().__init__()
        self.game = game
        self.render_mode = render_mode
        self.env = SimpleDiscretizer(retro.make(game=self.game))
        self.action_space = self.env.action_space
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1, 84, 84), dtype=np.float32
        )
        self.prev_score = 0
        self.prev_ram = self.env.unwrapped.get_ram()
        self.last_logged_score = 0

    def preprocess(self, obs):
        if obs is None or not isinstance(obs, np.ndarray):
            raise ValueError("❌ Invalid observation passed to preprocess()")
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return obs.astype(np.float32)[None, :, :] / 255.0

    def get_score_from_ram(self, ram):
        try:
            # NES MsPacMan — BCD digits, 6 digits, last digit always zero
            score = (
                ((ram[0x4D] >> 4) & 0x0F) * 100000 +
                (ram[0x4D] & 0x0F) * 10000 +
                ((ram[0x4E] >> 4) & 0x0F) * 1000 +
                (ram[0x4E] & 0x0F) * 100 +
                ((ram[0x4F] >> 4) & 0x0F) * 10
            )
            if score > 999990:
                score = 999990
            return score
        except Exception:
            return 0

    def calculate_reward(self, ram):
        curr_score = self.get_score_from_ram(ram)
        delta = curr_score - self.prev_score
        # Only positive, reasonable deltas are rewarded
        reward = delta if 0 < delta < 1000 else 0
        self.prev_score = curr_score
        return reward


    def risk_signal(self, ram):
        pacman_x = int(ram[0x44]) & 0xFF
        pacman_y = int(ram[0x4C]) & 0xFF
        ghosts_x = [
            int(ram[0x6E]) & 0xFF,
            int(ram[0x70]) & 0xFF,
            int(ram[0x72]) & 0xFF,
            int(ram[0x74]) & 0xFF,
        ]
        ghosts_y = [
            int(ram[0x88]) & 0xFF,
            int(ram[0x8A]) & 0xFF,
            int(ram[0x8C]) & 0xFF,
            int(ram[0x8E]) & 0xFF,
        ]

        for gx, gy in zip(ghosts_x, ghosts_y):
            if gx >= 240 or gy >= 240:
                continue  # invalid ghost pos
            dx = abs(pacman_x - gx)
            dy = abs(pacman_y - gy)
            if dx + dy <= 6:
                return 1
        return 0

    def reset(self, seed=None, options=None):
        obs, _ = self.env.reset()
        self.prev_score = 0
        self.last_logged_score = 0
        self.prev_ram = self.env.unwrapped.get_ram()
        return self.preprocess(obs), {}

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        ram = self.env.unwrapped.get_ram().copy()
        # Print all RAM values for manual mapping
        print(f"[DEBUG] RAM snapshot: {[ram[i] for i in range(0x00, 0xA0)]}")
        # Debug: Print RAM score bytes and decoded score
        print(f"[DEBUG] RAM: 4D={ram[0x4D]:02x} 4E={ram[0x4E]:02x} 4F={ram[0x4F]:02x}")
        print(f"[DEBUG] Decoded score: {self.get_score_from_ram(ram)}")
        # Debug: Print Pac-Man and ghost positions
        pacman_x = int(ram[0x44]) & 0xFF
        pacman_y = int(ram[0x4C]) & 0xFF
        ghosts_x = [int(ram[0x6E]) & 0xFF, int(ram[0x70]) & 0xFF, int(ram[0x72]) & 0xFF, int(ram[0x74]) & 0xFF]
        ghosts_y = [int(ram[0x88]) & 0xFF, int(ram[0x8A]) & 0xFF, int(ram[0x8C]) & 0xFF, int(ram[0x8E]) & 0xFF]
        print(f"[DEBUG] Pac-Man: ({pacman_x},{pacman_y}) | Ghosts: {list(zip(ghosts_x, ghosts_y))}")
        # Print RAM for 0x70-0x7F and 0x40-0x4F to help find correct mapping
        print(f"[DEBUG] RAM 0x70-0x7F: {[ram[i] for i in range(0x70, 0x80)]}")
        print(f"[DEBUG] RAM 0x40-0x4F: {[ram[i] for i in range(0x40, 0x50)]}")
        reward = self.calculate_reward(ram)
        risk = self.risk_signal(ram)
        info["risk"] = risk

        if self.prev_score != self.last_logged_score:
            print(f"\rScore: {self.prev_score} | Risk: {risk}   ", end="", flush=True)
            self.last_logged_score = self.prev_score

        self.prev_ram = ram.copy()
        return self.preprocess(obs), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self.env.render()

    def close(self):
        self.env.close()



# -----------------------
# 2. PyTorch Policy
# -----------------------
class CNNPolicy(nn.Module):
    def __init__(self, action_space):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 9 * 9, 512),
            nn.ReLU(),
            nn.Linear(512, action_space.n)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

# -----------------------
# 3. Main loop
# -----------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PacManRetroEnv(render_mode="human")
    policy = CNNPolicy(env.action_space).to(device)

    obs, _ = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # (B, C, H, W)

    print("🎮 Launching shaped MsPacMan-Nes agent...")

    total_reward = 0
    while True:
        env.render()
        with torch.no_grad():
            logits = policy(obs_tensor)
            raw_action = torch.argmax(logits, dim=1).item()
            action = raw_action

        next_obs, reward, terminated, truncated, info = env.step(action)
        obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward += reward

        print(f"Reward: {reward} | Total: {total_reward} | Risk: {info.get('risk', 0)}")
        if terminated or truncated:
            break

    env.close()
    print("🏁 Episode ended. Total score:", total_reward)


main()
