import numpy as np
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from rlbench.tasks.pick_described_object import PickDescribedObject
from TD3Agent import TD3Agent, load_demos_to_buffer

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
batch_size = 256
num_episodes = 10000
max_steps = 200

# Initialize environment
env = Environment(
    action_mode=MoveArmThenGripper(
        arm_action_mode=JointVelocity(),
        gripper_action_mode=Discrete(),
    ),
    obs_config=ObservationConfig(),
    robot_setup="panda",
    shaped_rewards=True,
    headless=False,
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

# Load demos into replay buffer
demos_path = "./pick_described_object_demos/item_pos_demos.pt"
load_demos_to_buffer(agent.replay_buffer, demos_path)

# Training loop
for episode in range(num_episodes):
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

        if agent.replay_buffer.size > batch_size:
            critic_loss, actor_loss = agent.train(batch_size)

        if done:
            break

    print(f"Episode {episode + 1}, Reward: {episode_reward}, Steps: {step + 1}")

    if (episode + 1) % 100 == 0:
        agent.save(f"./td3_weights/td3_pick_described_object_{episode + 1}")

# Save final model
agent.save("./td3_weights/td3_pick_described_object_final")

env.shutdown()
