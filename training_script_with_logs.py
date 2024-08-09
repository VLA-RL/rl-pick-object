import numpy as np
import wandb
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from rlbench.tasks.pick_described_object import PickDescribedObject
from TD3Agent import TD3Agent
import os
import glob
import re

# Hyperparameters
state_dim = 7 + 3 + 7 + 7 + 1  # Adjust based on your state representation
action_dim = 8  # 7 for arm joints + 1 for gripper
max_action = 1.0
lr = 3e-4
gamma = 0.99
tau = 0.005
policy_noise = 0.1
noise_clip = 0.2
policy_freq = 2
batch_size = 1024
num_episodes = 10000
max_steps = 200

# Initialize wandb
wandb.init(
    project="RLBench - Pick Described Object",
    config={
        "lr": lr,
        "gamma": gamma,
        "tau": tau,
        "policy_noise": policy_noise,
        "noise_clip": noise_clip,
        "policy_freq": policy_freq,
        "batch_size": batch_size,
        "num_episodes": num_episodes,
        "max_steps": max_steps,
    },
)

# Initialize environment
env = Environment(
    action_mode=MoveArmThenGripper(
        arm_action_mode=JointVelocity(),
        gripper_action_mode=Discrete(),
    ),
    obs_config=ObservationConfig(),
    robot_setup="panda",
    shaped_rewards=True,
    headless=True,
)
env.launch()

task = env.get_task(PickDescribedObject)

# Initialize TD3Agent
agent = TD3Agent(
    state_dim,
    action_dim,
    max_action,
    lr,
    gamma,
    tau,
    policy_noise,
    noise_clip,
    policy_freq,
)

# Create a directory for saving weights
weights_dir = "./td3_weights"
os.makedirs(weights_dir, exist_ok=True)
existing_weights = glob.glob(
    os.path.join(weights_dir, "td3_pick_described_object_*_actor")
)
if existing_weights:
    # Sort weights by episode number
    def extract_number(filename):
        match = re.search(r"td3_pick_described_object_(\d+)_actor$", filename)
        return int(match.group(1)) if match else 0

    existing_weights.sort(key=extract_number)
    latest_weights = existing_weights[-1]
    latest_weights_base = latest_weights[:-6]  # Remove '_actor' from the end
    print(f"Loading existing weights from {latest_weights_base}")
    agent.load(latest_weights_base)
    # Extract the starting episode number
    start_episode = extract_number(latest_weights)
else:
    print("No existing weights found. Starting from scratch.")
    # Load demos into replay buffer
    start_episode = 0
print(f"Start Episode: {start_episode}")
# Training loop
for episode in range(start_episode, num_episodes):
    task.sample_variation()
    descriptions, obs = task.reset()
    state = np.concatenate(
        [
            task._task.item.get_pose(),
            task._task.dropin_box.get_position(),
            obs.joint_positions,
            obs.gripper_pose,
            [obs.gripper_open],
        ]
    )
    episode_reward = 0
    episode_steps = 0
    total_loss = 0
    actor_loss_sum = 0
    critic_loss_sum = 0

    for step in range(max_steps):
        action = agent.select_action(state)
        action = (action + np.random.normal(0, 0.1, size=action_dim)).clip(
            -max_action, max_action
        )

        next_obs, reward, done = task.step(action)
        next_state = np.concatenate(
            [
                task._task.item.get_pose(),
                task._task.dropin_box.get_position(),
                next_obs.joint_positions,
                next_obs.gripper_pose,
                [next_obs.gripper_open],
            ]
        )

        agent.replay_buffer.add(state, action, next_state, reward, done)

        state = next_state
        episode_reward += reward
        episode_steps += 1

        if agent.replay_buffer.size > batch_size:
            critic_loss, actor_loss = agent.train(batch_size)
            total_loss += critic_loss + (actor_loss if actor_loss is not None else 0)
            critic_loss_sum += critic_loss
            if actor_loss is not None:
                actor_loss_sum += actor_loss

        if done:
            break

    # Log data to wandb
    wandb.log(
        {
            "reward": episode_reward,
            "total_loss": total_loss / episode_steps,
            "actor_loss": actor_loss_sum / episode_steps,
            "critic_loss": critic_loss_sum / episode_steps,
            "episode_length": episode_steps,
        },
        step=episode + 1,
    )

    print(f"Episode {episode + 1}, Reward: {episode_reward}, Steps: {episode_steps}")

    if (episode + 1) % 50 == 0:
        save_path = os.path.join(
            weights_dir, f"td3_pick_described_object_{episode + 1}"
        )
        agent.save(save_path)
        print(f"Saved weights at episode {episode + 1}")

# Save final model
agent.save(os.path.join(weights_dir, "td3_pick_described_object_final"))

env.shutdown()
wandb.finish()
