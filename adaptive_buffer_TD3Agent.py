import torch
import torch.nn as nn
import copy
import numpy as np
from collections import namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Experience = namedtuple(
    "Experience", ("state", "action", "next_state", "reward", "done", "priority")
)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, action_dim),
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * torch.tanh(self.net(state))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
        )
        self.net2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        return self.net1(state_action), self.net2(state_action)

    def Q1(self, state, action):
        state_action = torch.cat([state, action], 1)
        return self.net1(state_action)


class AdaptiveBatchReplayBuffer:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_size=int(1e5),
        alpha=0.6,
        beta=0.4,
        beta_increment=0.001,
    ):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))
        self.priorities = np.zeros((max_size, 1), dtype=np.float32)

        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0

        self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        self.priorities[self.ptr] = self.max_priority

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        if self.size == 0:
            return None

        priorities = self.priorities[: self.size]
        probabilities = priorities**self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(
            self.size, batch_size, p=probabilities.flatten(), replace=True
        )
        samples = [
            self.state[indices],
            self.action[indices],
            self.next_state[indices],
            self.reward[indices],
            self.done[indices],
        ]

        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)

        return (
            (torch.FloatTensor(s).to(self.device) for s in samples),
            indices,
            torch.FloatTensor(weights).to(self.device),
        )

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
        self.max_priority = max(self.max_priority, np.max(priorities))


class TD3Agent:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        lr=2e-4,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.1,
        noise_clip=0.2,
        policy_freq=2,
    ):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

        self.replay_buffer = AdaptiveBatchReplayBuffer(state_dim, action_dim)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, batch_size=512):
        self.total_it += 1

        # Sample replay buffer
        samples, indices, weights = self.replay_buffer.sample(batch_size)
        if samples is None:
            return None, None

        state, action, next_state, reward, done = samples

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = (
            weights
            * (
                nn.MSELoss(reduction="none")(current_Q1, target_Q)
                + nn.MSELoss(reduction="none")(current_Q2, target_Q)
            )
        ).mean()
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update priorities
        td_errors = torch.abs(target_Q - current_Q1).detach().cpu().numpy()
        self.replay_buffer.update_priorities(
            indices, td_errors + 1e-6
        )  # small constant to avoid zero priority

        actor_loss = None

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

        return critic_loss.item(), actor_loss.item() if actor_loss is not None else None

    def eval(self, env, num_episodes=10, max_steps=1000):
        total_rewards = []
        for _ in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            for _ in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, _, _ = env.step(action)
                episode_reward += reward
                state = next_state
                if done:
                    break
            total_rewards.append(episode_reward)

        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        return avg_reward, std_reward

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(
            torch.load(filename + "_critic", map_location=device)
        )
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer", map_location=device)
        )
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=device))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer", map_location=device)
        )
        self.actor_target = copy.deepcopy(self.actor)


def load_demos_to_buffer(replay_buffer, demos_path):
    demos = torch.load(demos_path)
    for demo in demos:
        for state, action, next_state, reward, done in demo:
            replay_buffer.add(state, action, next_state, reward, done)
    print(f"Loaded {len(demos)} demos into the replay buffer")
    print(f"replay buffer size: {replay_buffer.size}")
