import os
import sys
import argparse
import random
from collections import deque
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import cv2
import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
gym.register_envs(ale_py)

GAMEDIR = "Pong"
LOG_DIR = f"{GAMEDIR}/logs"
PLOTS_DIR = f"{GAMEDIR}/plots"
CKPT_DIR = f"{GAMEDIR}/checkpoints"
checkpoint=50

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def preprocess_pong(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) 
    small = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return (small / 255.0).astype(np.float16)

class DQNet_Pong(nn.Module):
    def __init__(self, act_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, act_dim)
        )


    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.int64, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)
        return states, actions, rewards, next_states, dones
    def __len__(self):
        return len(self.buffer)
    
class dqnagent:
    def __init__(self, env, action_dim):
        self.env = env
        self.action_dim = action_dim

        self.policy_net = DQNet_Pong(action_dim).to(device)
        self.target_net = DQNet_Pong(action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.replay_buffer = ReplayBuffer(10000)

        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay = 1000000
        self.epsilon = self.epsilon_start
        self.stepsdone= 0
        self.update_target_steps = 10000
    
    def select_action(self, state):
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       np.exp(-1. * self.stepsdone / self.epsilon_decay)
        self.stepsdone += 1
        if random.random() < self.epsilon:
            # return self.env.action_space.sample()
            return random.randrange(self.action_dim)
        
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.stepsdone % self.update_target_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.item()
    
    def savecheckpoint(self,episode,total_steps,filepath=CKPT_DIR):
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode': episode,
            'total_steps': total_steps,
            'steopsdone': self.stepsdone,
            'replay_buffer': self.replay_buffer,
        }
        torch.save(checkpoint, os.path.join(filepath, f"dqn_pong_checkpoint_ep{episode}.pth"))
        print(f"Checkpoint saved at episode {episode}")
    
    def loadcheckpoint(agent, filepath):
        if not os.path.isfile(filepath):
            print(f"No checkpoint found at {filepath}")
            return False,0,0
        checkpoint = torch.load(filepath)
        agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.stepsdone = checkpoint['stepsdone']
        agent.replay_buffer = checkpoint['replay_buffer']
        episode = checkpoint['episode']
        total_steps = checkpoint['total_steps']
        print(f"Checkpoint loaded from {filepath}")
        return True,episode+1,total_steps

def dqntrain(envname,  num_episodes):
    env = gym.make(envname, frameskip=4, repeat_action_probability=0.0)
    action_dim = env.action_space.n
    agent = dqnagent(env, action_dim)
    total_steps = 0
    load, start_episode, total_steps = dqnagent.loadcheckpoint(agent, os.path.join(CKPT_DIR, "dqn_pong_checkpoint.pth"))
    if load:
        print(f"Resuming training from episode {start_episode}")
    else:
        start_episode = 0
    
    print("Starting DQN training...")
    episode_rewards = []
    steps_per_episode = []
  
    for episode in range(num_episodes):
        obs, info = env.reset()
        state = np.stack([preprocess_pong(obs)] * 4, axis=0)
        done = False
        truncated = False
        episode_reward = 0
        step = 0
        while not done:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_frame = preprocess_pong(next_obs)
            next_state = np.concatenate([state[1:], next_frame[None,:,:]],axis=0)

            agent.replay_buffer.push(state, action, reward, next_state, done)
            loss = agent.train_step()

            state = next_state
            episode_reward += reward
            step += 1
            total_steps += 1

        episode_rewards.append(episode_reward)
        steps_per_episode.append(step)

        if (episode + 1) % checkpoint == 0:
            agent.savecheckpoint(episode , total_steps)
        print(f"[DQN] Ep {episode + 1}/{num_episodes} Reward={episode_reward:.2f} Eps={agent.epsilon:.3f}")

    agent.savecheckpoint(episode - 1, total_steps)
    env.close()

    window = 20
    moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    episodes = np.arange(window, len(episode_rewards) + 1)
    best_idx = np.argmax(moving_avg)
    best_ep = episodes[best_idx]
    best_val = moving_avg[best_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, moving_avg, label=f'{window}-Episode Moving Average', color='b')
    plt.scatter(best_ep, best_val, color='r', s=80, label=f'Best Avg: {best_val:.2f} at Ep {best_ep}')
    plt.title(f'DQN Training on {envname} - {window}-Episode Mean Reward')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    plot_path = os.path.join(PLOTS_DIR, f"dqn_pong_avg_reward_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.show()

if __name__ == "__main__":
    dqntrain("ALE/Pong-v5", num_episodes=5000)