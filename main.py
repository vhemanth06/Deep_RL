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

GAMEDIR = "Pong"
LOG_DIR = f"{GAMEDIR}/logs"
PLOTS_DIR = f"{GAMEDIR}/plots"
CKPT_DIR = f"{GAMEDIR}/checkpoints"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj); f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

logfile = open(os.path.join(LOG_DIR, "output.txt"), "a")
sys.stdout = Tee(sys.stdout, logfile)
sys.stderr = Tee(sys.stderr, logfile)

try:
    import ale_py
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def probe_env(env_name, episodes=3, max_steps=500):
    print(f"Probing environment: {env_name}")
    env = gym.make(env_name)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    for ep in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        for t in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        print(f"Random policy episode {ep+1}: reward = {total_reward}")
    env.close()

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, *transition):
        self.buffer.append(tuple(transition))
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


class DQN_CNN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        c, h, w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
        )
        # compute conv out
        with torch.no_grad():
            o = self.conv(torch.zeros(1, *input_shape))
            conv_out_size = int(o.numel())
        self.fc = nn.Sequential(nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, n_actions))
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class DQN_MLP(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))
    def forward(self, x):
        return self.net(x)


def preprocess_pong(obs):
    # obs = obs[35:195] 
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    return obs.astype(np.float32) / 255.0


# def save_checkpoint(policy_net, target_net, optimizer, episode, total_steps, returns, path):
#     torch.save({
#         "policy_state_dict": policy_net.state_dict(),
#         "target_state_dict": target_net.state_dict(),
#         "optimizer_state_dict": optimizer.state_dict(),
#         "episode": episode,
#         "total_steps": total_steps,
#         "returns": returns
#     }, path)
#     print(f"Saved checkpoint: {path}")
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

    # Save last model
    torch.save(ckpt, last_path)
    print(f"Saved latest checkpoint (episode {episode}) → {last_path}")

    # Check if this is the new best
    current_mean = np.mean(returns[-20:]) if len(returns) >= 20 else returns[-1]
    if current_mean > best_mean:
        torch.save(ckpt, best_path)
        print(f"New best model saved (mean(20)={current_mean:.2f}) → {best_path}")
        best_mean = current_mean

    return best_mean


def load_checkpoint(path, policy_net, target_net, optimizer):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    policy_net.load_state_dict(ckpt["policy_state_dict"])
    target_net.load_state_dict(ckpt["target_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"Loaded checkpoint from {path}")
    return ckpt.get("episode", 0), ckpt.get("total_steps", 0), ckpt.get("returns", [])

def latest_checkpoint():
    files = [os.path.join(CKPT_DIR, f) for f in os.listdir(CKPT_DIR) if f.endswith(".pth")]
    if not files:
        return None
    files.sort(key=lambda x: os.path.getmtime(x))
    return files[-1]

def dqn_train(env_name="ALE/Pong-v5",
              total_episodes=500,
              batch_size=32,
              gamma=0.99,
              lr=1e-4,
              buffer_capacity=100000,
              min_buffer_size=5000,
              target_update_steps=1000,
              epsilon_start=1.0,
              epsilon_end=0.1,
              epsilon_decay=50000,
              render=False,
              save_every=10,
              use_resume=False,
              resume_best=False
              ):

    print(f"Starting DQN on {env_name} for {total_episodes} episodes")
    env = gym.make(env_name, render_mode=("human" if render else None))
    n_actions = env.action_space.n

    is_pixel = len(env.observation_space.shape) == 3  # Pong
    if is_pixel:
        policy_net = DQN_CNN((4,84,84), n_actions).to(device)
        target_net = DQN_CNN((4,84,84), n_actions).to(device)
    else:
        state_dim = env.observation_space.shape[0]
        policy_net = DQN_MLP(state_dim, n_actions).to(device)
        target_net = DQN_MLP(state_dim, n_actions).to(device)

    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_capacity)

    # resume
    # start_episode = 0
    # total_steps = 0
    # returns = []
    # if use_resume:
    #     ckpt = latest_checkpoint()
    #     if ckpt:
    #         start_episode, total_steps, returns = load_checkpoint(ckpt, policy_net, target_net, optimizer)

    # best_mean = -float("inf")
    start_episode = 0
    total_steps = 0
    returns = []
    best_mean = -float("inf")

    if use_resume:
        if resume_best:
            ckpt = os.path.join(CKPT_DIR, "best_model.pth")
            print("Resuming from BEST model checkpoint")
        else:
            ckpt = latest_checkpoint()
            print("Resuming from latest checkpoint")

        if ckpt and os.path.exists(ckpt):
            start_episode, total_steps, returns = load_checkpoint(ckpt, policy_net, target_net, optimizer)
            best_mean = np.mean(returns[-20:]) if len(returns) >= 20 else -float("inf")
            print(f"Resumed from {ckpt} (Episode {start_episode}, Steps {total_steps})")
        else:
            print("No checkpoint found, starting fresh.")

    best_path = os.path.join(CKPT_DIR, "best_model.pth")
    last_path = os.path.join(CKPT_DIR, "last_model.pth")

    for ep in range(start_episode, total_episodes):
        # reset
        obs, info = env.reset()
        if is_pixel:
            last = preprocess_pong(obs)
            state_stack = np.stack([last]*4, axis=0)  # 4x84x84
        else:
            state_stack = np.array(obs, dtype=np.float32)  # state vector

        done = False
        total_reward = 0
        step = 0

        while True:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-total_steps / epsilon_decay)
            # select action
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    if is_pixel:
                        s = torch.tensor(state_stack, dtype=torch.float32, device=device).unsqueeze(0)
                        q = policy_net(s)
                        action = int(torch.argmax(q, dim=1).item())
                    else:
                        s = torch.tensor(state_stack, dtype=torch.float32, device=device).unsqueeze(0)
                        q = policy_net(s)
                        action = int(torch.argmax(q, dim=1).item())

            next_obs, reward, terminated, truncated, info = env.step(action)
            done_flag = terminated or truncated
            total_reward += reward
            total_steps += 1
            step += 1

            if is_pixel:
                next_pre = preprocess_pong(next_obs)
                frame_diff = next_pre - last
                next_stack = np.roll(state_stack, -1, axis=0)
                next_stack[-1] = frame_diff
                buffer.push(state_stack, action, reward, next_stack, float(done_flag))
                state_stack = next_stack
                last = next_pre
            else:
                next_state = np.array(next_obs, dtype=np.float32)
                buffer.push(state_stack, action, reward, next_state, float(done_flag))
                state_stack = next_state

            # learn
            if len(buffer) > min_buffer_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                curr_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
                next_q = target_net(next_states).max(1)[0]
                targets = rewards + gamma * next_q * (1 - dones)
                loss = nn.MSELoss()(curr_q, targets.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if total_steps % target_update_steps == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done_flag:
                break

        returns.append(total_reward)
        mean20 = np.mean(returns[-20:]) if len(returns) >= 1 else total_reward
        print(f"[DQN] Ep {ep+1}/{total_episodes} Reward={total_reward:.2f} Mean20={mean20:.2f} Eps={epsilon:.3f}")

        # save checkpoints
        # if (ep + 1) % 10 == 0:
        #     ckpt_path = f"checkpoints/dqn_{env_name.replace('/', '_')}_ep{ep+1}.pth"
        #     save_checkpoint(policy_net, target_net, optimizer, ep+1, total_steps, returns, ckpt_path)

        # # save best
        # if mean20 > best_mean:
        #     best_mean = mean20
        #     save_checkpoint(policy_net, target_net, optimizer, ep+1, total_steps, returns, os.path.join(CKPT_DIR, "dqn_best.pth"))
        if (ep + 1) % 10 == 0:
            best_mean = save_checkpoint(policy_net, target_net, optimizer,
                                  ep + 1, total_steps, returns,
                                  best_mean, best_path, last_path)
    env.close()

    # plot mean(20) vs episodes (x scaled to episodes)
    mean_returns = [np.mean(returns[max(0,i-19):i+1]) for i in range(len(returns))]
    plt.figure()
    plt.plot(mean_returns, label="mean(20)")
    plt.title(f"DQN on {env_name}")
    plt.xlabel("Episode")
    plt.ylabel("Mean return (20)")
    plt.legend()
    plot_path = os.path.join(PLOTS_DIR, f"dqn_{env_name.replace('/', '_')}_{timestamp()}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot: {plot_path}")

    # If MountainCar, create action map plot
    if ("MountainCar" in env_name) or ("MountainCar-v0" in env_name):
        # action space {0,1,2}, state = [pos, vel]
        print("Generating action-choice heatmap for MountainCar")
        policy_net.eval()
        pos_vals = np.linspace(-1.2, 0.6, 200)
        vel_vals = np.linspace(-0.07, 0.07, 200)
        action_map = np.zeros((len(vel_vals), len(pos_vals)), dtype=int)
        with torch.no_grad():
            for i, v in enumerate(vel_vals):
                states = np.stack([np.stack([p, v]) for p in pos_vals], axis=0)  # (P,2)
                s_tensor = torch.tensor(states, dtype=torch.float32, device=device)
                q = policy_net(s_tensor)
                acts = torch.argmax(q, dim=1).cpu().numpy()
                action_map[i, :] = acts
        plt.figure(figsize=(6,5))
        plt.imshow(action_map, origin="lower", extent=[pos_vals.min(), pos_vals.max(), vel_vals.min(), vel_vals.max()], aspect='auto')
        plt.colorbar(ticks=[0,1,2], label='action')
        plt.title("MountainCar policy action map (velocity vs position)")
        plt.xlabel("Position")
        plt.ylabel("Velocity")
        heatmap_path = os.path.join(PLOTS_DIR, f"mountaincar_actionmap_{timestamp()}.png")
        plt.savefig(heatmap_path)
        plt.close()
        print(f"Saved MountainCar action map: {heatmap_path}")

    return returns

def dqn_sweep(env_name, param_name, param_values, base_kwargs):

    results = {}
    for val in param_values:
        kwargs = base_kwargs.copy()
        kwargs[param_name] = val
        print(f"Running sweep: {param_name}={val}")
        returns = dqn_train(env_name=env_name, total_episodes=kwargs.get("total_episodes", 50),
                            batch_size=kwargs.get("batch_size", 32),
                            gamma=kwargs.get("gamma", 0.99),
                            lr=kwargs.get("lr", 1e-4),
                            buffer_capacity=kwargs.get("buffer_capacity", 100000),
                            min_buffer_size=kwargs.get("min_buffer_size", 5000),
                            target_update_steps=kwargs.get("target_update_steps", 1000),
                            epsilon_start=kwargs.get("epsilon_start", 1.0),
                            epsilon_end=kwargs.get("epsilon_end", 0.1),
                            epsilon_decay=kwargs.get("epsilon_decay", 50000),
                            render=kwargs.get("render", False),
                            save_every=kwargs.get("save_every", 1000),
                            use_resume=False)
        results[val] = [np.mean(returns[max(0,i-19):i+1]) for i in range(len(returns))]

    # plot
    plt.figure()
    for val, curve in results.items():
        plt.plot(curve, label=f"{param_name}={val}")
    plt.title(f"DQN sweep {param_name} on {env_name}")
    plt.xlabel("Episode")
    plt.ylabel("Mean return (20)")
    plt.legend()
    fname = os.path.join(PLOTS_DIR, f"dqn_sweep_{param_name}_{timestamp()}.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved sweep plot: {fname}")


class PolicyMLP(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))
    def forward(self, x):
        return self.net(x)

def compute_returns(rewards, gamma=0.99, reward_to_go=False):
    if reward_to_go:
        out = []
        for i in range(len(rewards)):
            G = 0.0
            pw = 1.0
            for r in rewards[i:]:
                G += pw * r
                pw *= gamma
            out.append(G)
        return np.array(out, dtype=np.float32)
    else:
        G = 0.0
        pw = 1.0
        for r in rewards:
            G += pw * r
            pw *= gamma
        return np.array([G for _ in rewards], dtype=np.float32)

def pg_train(env_name="CartPole-v0", iterations=50, batch_size=5000, lr=1e-2, gamma=0.99,
             reward_to_go=False, adv_norm=False, render=False):
    print(f"PG training on {env_name} | reward_to_go={reward_to_go} adv_norm={adv_norm}")
    env = gym.make(env_name, render_mode=("human" if render else None))
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    policy = PolicyMLP(state_dim, n_actions).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    all_mean_rewards = []

    for it in range(iterations):
        batch_states, batch_actions, batch_weights, batch_episode_rewards = [], [], [], []
        steps = 0
        # collect trajectories until steps >= batch_size
        while steps < batch_size:
            obs, info = env.reset()
            done = False
            states, actions, rewards = [], [], []
            while True:
                s_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits = policy(s_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = int(dist.sample().item())
                next_obs, reward, terminated, truncated, info = env.step(action)
                done_flag = terminated or truncated
                states.append(obs)
                actions.append(action)
                rewards.append(reward)
                obs = next_obs
                if done_flag:
                    break
            steps += len(states)
            batch_states += states
            batch_actions += actions
            batch_episode_rewards.append(sum(rewards))
            batch_weights += list(compute_returns(rewards, gamma, reward_to_go))

        batch_states_t = torch.tensor(np.array(batch_states), dtype=torch.float32, device=device)
        batch_actions_t = torch.tensor(batch_actions, dtype=torch.int64, device=device)
        batch_weights_t = torch.tensor(batch_weights, dtype=torch.float32, device=device)

        # advantage normalization
        if adv_norm:
            mean = batch_weights_t.mean()
            std = batch_weights_t.std() + 1e-8
            batch_weights_t = (batch_weights_t - mean) / std

        logits = policy(batch_states_t)
        dists = torch.distributions.Categorical(logits=logits)
        logp = dists.log_prob(batch_actions_t)
        loss = -(logp * batch_weights_t).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_reward = np.mean(batch_episode_rewards)
        all_mean_rewards.append(mean_reward)
        print(f"[PG] Iter {it+1}/{iterations} Mean reward: {mean_reward:.2f}")

    env.close()
    # plot
    plt.figure()
    plt.plot(all_mean_rewards)
    plt.title(f"Policy Gradient on {env_name} (reward_to_go={reward_to_go}, adv_norm={adv_norm})")
    plt.xlabel("Iteration")
    plt.ylabel("Mean episode reward")
    fname = os.path.join(PLOTS_DIR, f"pg_{env_name.replace('/', '_')}_{timestamp()}.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved PG plot: {fname}")

    return all_mean_rewards

def main():
    # parser = argparse.ArgumentParser(description="Assignment 3: DQN and Policy Gradient utilities")
    parser = argparse.ArgumentParser(
        description="Assignment 3: DQN and Policy Gradient utilities",
        # fromfile_prefix_chars='@'
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_probe = sub.add_parser("probe", help="Print env spaces and run a random agent")
    p_probe.add_argument("--env", type=str, required=True)
    p_probe.add_argument("--episodes", type=int, default=3)

    # dqn
    # p_dqn = sub.add_parser("dqn", help="Run DQN")
    # p_dqn.add_argument("--env", type=str, default="ALE/Pong-v5")
    # p_dqn.add_argument("--episodes", type=int, default=200)  
    # p_dqn.add_argument("--batch_size", type=int, default=32)
    # p_dqn.add_argument("--lr", type=float, default=1e-4)
    # p_dqn.add_argument("--buffer_capacity", type=int, default=100000)
    # p_dqn.add_argument("--min_buffer_size", type=int, default=5000)
    # p_dqn.add_argument("--target_update_steps", type=int, default=1000)
    # p_dqn.add_argument("--epsilon_decay", type=int, default=50000)
    # p_dqn.add_argument("--render", action="store_true")
    # p_dqn.add_argument("--resume", action="store_true")
    p_dqn = sub.add_parser("dqn", help="Run DQN")
    p_dqn.add_argument("--env", type=str, default="ALE/Pong-v5")
    p_dqn.add_argument("--episodes", type=int, default=200)
    p_dqn.add_argument("--batch_size", type=int, default=32)
    p_dqn.add_argument("--lr", type=float, default=1e-4)
    p_dqn.add_argument("--gamma", type=float, default=0.99)
    p_dqn.add_argument("--buffer_capacity", type=int, default=100000)
    p_dqn.add_argument("--min_buffer_size", type=int, default=5000)
    p_dqn.add_argument("--target_update_steps", type=int, default=1000)
    p_dqn.add_argument("--epsilon_start", type=float, default=1.0)
    p_dqn.add_argument("--epsilon_end", type=float, default=0.1)
    p_dqn.add_argument("--epsilon_decay", type=int, default=50000)
    p_dqn.add_argument("--render", action="store_true")
    p_dqn.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint (last_model.pth)")
    p_dqn.add_argument("--resume_best", action="store_true", help="Resume from the best model checkpoint (best_model.pth)")
    p_dqn.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N episodes (default=10)")


    # sweep
    p_sweep = sub.add_parser("dqn_sweep", help="Run DQN hyperparameter sweep")
    p_sweep.add_argument("--env", type=str, default="MountainCar-v0")
    p_sweep.add_argument("--param", type=str, required=True, help="parameter name to sweep (lr|batch_size|buffer_capacity)")
    p_sweep.add_argument("--values", type=str, required=True, help="comma-separated values")
    p_sweep.add_argument("--episodes", type=int, default=50)

    # pg
    p_pg = sub.add_parser("pg", help="Run policy gradient (REINFORCE)")
    p_pg.add_argument("--env", type=str, default="CartPole-v0")
    p_pg.add_argument("--iterations", type=int, default=30)
    p_pg.add_argument("--batch_size", type=int, default=2000)
    p_pg.add_argument("--lr", type=float, default=1e-2)
    p_pg.add_argument("--gamma", type=float, default=0.99)
    p_pg.add_argument("--reward_to_go", action="store_true")
    p_pg.add_argument("--adv_norm", action="store_true")
    p_pg.add_argument("--render", action="store_true")

    args = parser.parse_args()

    if args.cmd == "probe":
        probe_env(args.env, episodes=args.episodes)

    elif args.cmd == "dqn":
        # dqn_train(env_name=args.env,
        #           total_episodes=args.episodes,
        #           batch_size=args.batch_size,
        #           lr=args.lr,
        #           buffer_capacity=args.buffer_capacity,
        #           min_buffer_size=args.min_buffer_size,
        #           target_update_steps=args.target_update_steps,
        #           epsilon_decay=args.epsilon_decay,
        #           render=args.render,
        #           save_every=max(1, args.episodes // 10),
        #           use_resume=args.resume)
        
        dqn_train(
            env_name=args.env,
            total_episodes=args.episodes,
            batch_size=args.batch_size,
            gamma=args.gamma,
            lr=args.lr,
            buffer_capacity=args.buffer_capacity,
            min_buffer_size=args.min_buffer_size,
            target_update_steps=args.target_update_steps,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            render=args.render,
            save_every=args.save_every,
            use_resume=args.resume,
            resume_best=args.resume_best
        )


    elif args.cmd == "dqn_sweep":
        param = args.param
        vals = [float(v) if '.' in v else int(v) for v in args.values.split(",")]
        base = {"total_episodes": args.episodes, "batch_size":32, "lr":1e-4, "buffer_capacity":100000, "min_buffer_size":5000, "target_update_steps":1000}
        if param not in base:
            print(f"Param {param} not supported for sweep. Supported: {list(base.keys())}")
            return
        dqn_sweep(env_name=args.env, param_name=param, param_values=vals, base_kwargs=base)

    elif args.cmd == "pg":
        pg_train(env_name=args.env, iterations=args.iterations, batch_size=args.batch_size, lr=args.lr, gamma=args.gamma,
                 reward_to_go=args.reward_to_go, adv_norm=args.adv_norm, render=args.render)

if __name__ == "__main__":
    main()
