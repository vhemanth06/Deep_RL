import os
import sys
import random
from collections import deque
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import cv2
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

# Suppress UserWarning for cleaner console output
warnings.filterwarnings("ignore", category=UserWarning)
import ale_py
try:
    gym.register_envs(ale_py)
except:
    pass

# --- Configuration ---
GAMEDIR = "Pong_2"
LOG_DIR = f"{GAMEDIR}/logs"
PLOTS_DIR = f"{GAMEDIR}/plots"
CKPT_DIR = f"{GAMEDIR}/checkpoints"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# --- Logging setup (Tee output to console and file) ---
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj); f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

try:
    logfile = open(os.path.join(LOG_DIR, "output.txt"), "a")
    if not isinstance(sys.stdout, Tee):
        sys.stdout = Tee(sys.stdout, logfile)
    if not isinstance(sys.stderr, Tee):
        sys.stderr = Tee(sys.stderr, logfile)
except Exception as e:
    print(f"Warning: Could not set up log file Tee. Logging only to console. Error: {e}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, *transition):
        # State and Next_State are stored as np.uint8 arrays (max memory efficiency)
        self.buffer.append(tuple(transition))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to Tensors on the correct device (automatically converts uint8 to float32)
        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.int64, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)
        return states, actions, rewards, next_states, dones
    def __len__(self):
        return len(self.buffer)

# --- DQN CNN Architecture (Atari) ---
class DQN_CNN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        c, h, w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
        )
        # Compute conv out size dynamically

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )


    def forward(self, x):
        # x is BxCxHxW (expects un-normalized 0-255 values, converted to float32 tensor)
        x = x.to(self.conv[0].weight.device)
        # CRITICAL: Normalize data here, right before the first layer
        x = self.conv(x / 255.0)
        return self.fc(x)

# --- State Preprocessing for Pong (84x84 Grayscale Stack) ---
def preprocess_pong(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    # CRITICAL FIX: Store un-normalized pixel values as 1-byte integers
    return obs.astype(np.uint8)

# --- Checkpoint Functions ---
def save_checkpoint(policy_net, target_net, optimizer, episode, total_steps, returns, best_mean, best_path, last_path):
    ckpt = {
        "policy_state_dict": policy_net.state_dict(),
        "target_state_dict": target_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "episode": episode,
        "total_steps": total_steps,
        "returns": returns,
        "best_mean": best_mean
    }
    torch.save(ckpt, last_path)
    window = min(len(returns), 100)
    current_mean = np.mean(returns[-window:]) if window > 0 else -float("inf")
    if current_mean > best_mean or current_mean > 18.0:
        torch.save(ckpt, best_path)
        print(f"New best model saved (mean({window})={current_mean:.2f}) → {best_path}")
        best_mean = current_mean
    return best_mean

def load_checkpoint(path, policy_net, target_net, optimizer):
    if not os.path.exists(path):
        return 0, 0, [], -float("inf")
    ckpt = torch.load(path, map_location=device)
    policy_net.load_state_dict(ckpt["policy_state_dict"])
    target_net.load_state_dict(ckpt["target_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("episode", 0), ckpt.get("total_steps", 0), ckpt.get("returns", []), ckpt.get("best_mean", -float("inf"))

# --- DQN Training Loop ---
def dqn_train(env_name="ALE/Pong-v5",
              total_episodes=2000,
              # REDUCED BATCH SIZE TO PREVENT MEMORY SPIKES DURING TRAINING
              batch_size=16,
              gamma=0.99,
              lr=5e-4,
              # REDUCED CAPACITY TO ENSURE LOW MEMORY FOOTPRINT (approx 275MB)
              buffer_capacity=5000,
              target_update_steps=2000,
              epsilon_start=1.0,
              epsilon_end=0.1,
              epsilon_decay=1000000,
              render=False,
              save_every=10,
              use_resume=True,
              resume_best=False,
              max_episode_steps=2000
              ):

    print(f"Starting Pong DQN on {env_name}")
    env = gym.make(env_name, render_mode=("human" if render else None), frameskip=4, repeat_action_probability=0.0)
    n_actions = env.action_space.n

    policy_net = DQN_CNN((4,84,84), n_actions).to(device)
    target_net = DQN_CNN((4,84,84), n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_capacity)

    start_episode = 0
    total_steps = 0
    returns = []
    episode_end_steps = []
    best_mean = -float("inf")

    best_path = os.path.join(CKPT_DIR, "best_model.pth")
    last_path = os.path.join(CKPT_DIR, "last_model.pth")

    # Load checkpoint logic
    if use_resume:
        ckpt_to_load = best_path if resume_best and os.path.exists(best_path) else last_path
        start_episode, total_steps, returns, best_mean = load_checkpoint(ckpt_to_load, policy_net, target_net, optimizer)
        if start_episode > 0:
            print(f"Resumed from {ckpt_to_load} (Episode {start_episode}, Best Mean {best_mean:.2f})")
        else:
            print("No checkpoint found, starting fresh.")

    # Populating initial buffer (a small loop for better stability before training starts)
    if len(buffer) < batch_size:
        # print(f"Populating replay buffer to minimum batch size of {batch_size}...")
        obs, info = env.reset()
        last = preprocess_pong(obs) # uint8 array
        state_stack = np.stack([last]*4, axis=0) # uint8 stack

        while len(buffer) < batch_size:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done_flag = terminated or truncated

            next_pre = preprocess_pong(next_obs) # uint8 array
            next_stack = np.concatenate([state_stack[1:, ...], next_pre[None, ...]], axis=0) # uint8 stack

            buffer.push(state_stack, action, reward, next_stack, float(done_flag))
            state_stack = next_stack

            if done_flag:
                obs, info = env.reset()
                last = preprocess_pong(obs)
                state_stack = np.stack([last]*4, axis=0)

            total_steps += 1
        # print("Buffer minimum size reached. Starting training.")


    for ep in range(start_episode, total_episodes):
        # Episode start
        obs, info = env.reset()
        last = preprocess_pong(obs)
        state_stack = np.stack([last]*4, axis=0) # uint8 stack

        done = False
        total_reward = 0
        step = 0
        n = 100
        while not done:
            # Epsilon calculation
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-total_steps / epsilon_decay)

            # Action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    # Need to load as float32 tensor for division inside forward pass
                    s = torch.tensor(state_stack, dtype=torch.float32, device=device).unsqueeze(0)
                    q = policy_net(s)
                    action = int(torch.argmax(q, dim=1).item())

            # Perform action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done_flag = terminated or truncated

            # --- Max Episode Steps Check ---
            step += 1
            if step >= max_episode_steps:
                truncated = True
                done_flag = True

            done = done_flag

            total_reward += reward
            total_steps += 1

            # Prepare next state and push transition
            next_pre = preprocess_pong(next_obs)
            next_stack = np.concatenate([state_stack[1:, ...], next_pre[None, ...]], axis=0)
            buffer.push(state_stack, action, reward, next_stack, float(done_flag))
            state_stack = next_stack

            # --- Learn (Standard DQN) ---
            if len(buffer) >= batch_size and total_steps % 4 == 0:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)

                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0]

                curr_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
                targets = rewards + gamma * next_q * (1 - dones)

                loss = nn.SmoothL1Loss()(curr_q, targets.detach())

                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                for param in policy_net.parameters():
                    param.grad.data.clamp_(-1., 1.)

                optimizer.step()

                # Target network update
                if total_steps % target_update_steps == 0:
                    target_net.load_state_dict(policy_net.state_dict())


        returns.append(total_reward)
        episode_end_steps.append(total_steps)

        mean_window = min(len(returns), 100)
        mean_reward = np.mean(returns[-mean_window:]) if mean_window > 0 else total_reward

        # FIXED: Added TotalSteps to the print output
        print(f"[DQN] Ep {ep+1}/{total_episodes} Reward={total_reward:.2f} Mean{n}={mean_reward:.2f} Eps={epsilon:.3f}")

        # Save checkpoints
        if (ep + 1) % save_every == 0:
            best_mean = save_checkpoint(policy_net, target_net, optimizer,
                                  ep + 1, total_steps, returns,
                                  best_mean, best_path, last_path)



    env.close()

    # --- Final Plotting (Reward vs. Total Timesteps) ---
    
    if len(returns) < n:
        print("Not enough episodes to generate 100-episode mean plot.")
        return returns

    print(f"Generating final plot of Mean({n}) Reward vs. Total Environment Steps...")
    rolling_means = np.convolve(returns, np.ones(n)/n, mode='valid')
    plot_steps = episode_end_steps[n-1:]
    best_means_line = np.maximum.accumulate(rolling_means)

    plt.figure(figsize=(10, 6))
    plt.plot(plot_steps, rolling_means, label=f"Mean({n}) Reward", color="blue", alpha=0.6)
    plt.plot(plot_steps, best_means_line, label="Best Mean (Cumulative Max)", color="red", linestyle="--", linewidth=1.5)

    plt.xlabel("Total Environment Steps (Frameskip=4)")
    plt.ylabel(f"Mean {n}-Episode Reward ")
    plt.title(f"DQN Training Progress on {env_name}")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(PLOTS_DIR, f"dqn_{env_name.replace('/', '_')}_{timestamp()}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved final plot: {plot_path}")

    return returns

# --- Execution ---
if __name__ == "__main__":

    dqn_train(
        lr=5e-4,
        epsilon_decay=1000000,
        max_episode_steps=2000,
        use_resume=True
    )

    print("DQN Training Finished.")
