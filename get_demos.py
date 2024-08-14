import numpy as np
import torch
from rlbench.environment import Environment
from rlbench.task_environment import TaskEnvironment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from rlbench.tasks.pick_described_object import PickDescribedObject
from rlbench.backend.observation import Observation
import os
from typing import List, Tuple


class DemoCollector:
    def __init__(self, task: TaskEnvironment):
        self.task = task
        self.demo_steps = []

    def process_step(self, obs: Observation):
        gripper_position = self.task._task.robot.arm.get_tip().get_position()
        object_position = self.task._task.target_object.get_position()
        basket_position = self.task._task.dropin_box.get_position()
        grasped_object = self.task._task.robot.gripper.get_grasped_objects()
        object_grasped = (
            grasped_object and grasped_object[0] == self.task._task.target_object
        )
        state = np.concatenate(
            [
                self.task._task.target_object.get_pose(),
                self.task._task.dropin_box.get_position(),
                obs.joint_positions,
                obs.gripper_pose,
                [obs.gripper_open],
            ]
        )

        action = np.concatenate([obs.joint_velocities, [obs.gripper_open]])

        self.demo_steps.append(
            {
                "state": state,
                "action": action,
                "gripper_position": gripper_position,
                "object_position": object_position,
                "basket_position": basket_position,
                "object_grasped": object_grasped,
            }
        )

    def process_demo(self) -> List[Tuple]:
        processed_demo = []
        for i in range(len(self.demo_steps) - 1):
            current_step = self.demo_steps[i]
            next_step = self.demo_steps[i + 1]

            state = current_step["state"]
            action = current_step["action"]
            next_state = next_step["state"]

            # Calculate reward
            reward = 0
            if not current_step["object_grasped"]:
                distance_to_object = np.linalg.norm(
                    next_step["gripper_position"] - next_step["object_position"]
                )
                reward -= distance_to_object
            else:
                distance_to_basket = np.linalg.norm(
                    next_step["object_position"] - next_step["basket_position"]
                )
                reward += 2 - distance_to_basket

            processed_demo.append((state, action, next_state, reward, False))

        final_step = self.demo_steps[-1]
        state = final_step["state"]
        action = final_step["action"]
        reward = 500
        done = True
        processed_demo.append((state, action, state, reward, done))

        self.demo_steps.clear()  # Clear the steps for the next demo
        return processed_demo


def collect_and_save_demos(task: TaskEnvironment, amount: int, save_path: str):
    collector = DemoCollector(task)
    processed_demos = []
    for _ in range(amount):
        task.sample_variation()
        task.get_demos(1, live_demos=True, callable_each_step=collector.process_step)
        processed_demo = collector.process_demo()
        processed_demos.append(processed_demo)

    # Save the demos
    torch.save(processed_demos, save_path)
    print(f"Saved {len(processed_demos)} demos to {save_path}")


def load_demos(load_path: str):
    demos = torch.load(load_path)
    print(f"Loaded {len(demos)} demos from {load_path}")
    return demos


if __name__ == "__main__":
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

    # Get and save demos
    save_dir = "./pick_described_object_demos"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "item_pos_demos.pt")
    collect_and_save_demos(task, 1, save_path)

    # Load and verify the demos
    loaded_demos = load_demos(save_path)

    # Use the demos as needed
    demo = loaded_demos[0]
    print(f"Demo 1 has {len(demo)} steps")
    state, action, next_state, reward, done = demo[0]
    print(f" State shape: {state.shape}")
    print(f" Action shape: {action.shape}")
    print(f" Next state shape: {next_state.shape}")
    for _, _, _, reward, _ in demo:
        print(f" Reward: {reward}")

    env.shutdown()
