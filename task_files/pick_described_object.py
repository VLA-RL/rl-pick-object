# From https://github.com/VLA-RL/RLBench-env-templates
from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, GraspedCondition, ConditionSet
from rlbench.backend.spawn_boundary import SpawnBoundary

GROCERY_NAMES = [
    "chocolate jello",
    "strawberry jello",
    "soup",
    "spam",
    "mustard",
    "sugar",
]


class PickDescribedObject(Task):
    def init_task(self) -> None:
        self.groceries = [Shape(name.replace(" ", "_")) for name in GROCERY_NAMES]
        self.grasp_points = [
            Dummy("%s_grasp_point" % name.replace(" ", "_")) for name in GROCERY_NAMES
        ]
        self.item = Dummy("waypoint0")
        self.over_box = Dummy("waypoint1")
        self.dropin_box = Dummy("waypoint2")

        self.register_graspable_objects(self.groceries)
        self.boundary = SpawnBoundary([Shape("workspace")])
        self.spawn_boundary = SpawnBoundary([Shape("groceries_boundary")])

        self.success_detector = ProximitySensor("success")

    def init_episode(self, index: int) -> List[str]:
        self.spawn_boundary.clear()
        [
            self.spawn_boundary.sample(
                g,
                ignore_collisions=False,
                min_rotation=(0.0, 0.0, -0.5),
                max_rotation=(0.0, 0.0, 0.5),
                min_distance=0.1,
            )
            for g in self.groceries
        ]
        self.item.set_position(self.grasp_points[index].get_position())
        self.item_name = GROCERY_NAMES[index]
        self.target_object = self.groceries[index]
        self.index = index
        self.first_grasped = False
        grasp_condition = GraspedCondition(self.robot.gripper, self.target_object)
        detect_condition = DetectedCondition(self.target_object, self.success_detector)

        self.gripper_position = self.robot.arm.get_tip().get_position()
        self.object_position = self.target_object.get_position()
        self.basket_position = self.dropin_box.get_position()
        self.prev_dist_to_basket = np.linalg.norm(
            self.object_position - self.basket_position
        )
        self.prev_dist_to_object = np.linalg.norm(
            self.gripper_position - self.object_position
        )
        condition_set = ConditionSet(
            [grasp_condition, detect_condition], order_matters=True
        )

        self.register_success_conditions([condition_set])

        return [
            "put the %s in the basket" % GROCERY_NAMES[index],
            "pick up the %s and place in the basket" % GROCERY_NAMES[index],
            "I want to put away the %s" % GROCERY_NAMES[index],
        ]

    def variation_count(self) -> int:
        return len(GROCERY_NAMES)

    def boundary_root(self) -> Object:
        return Shape("boundary_root")

    def base_rotation_bounds(
        self,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        return (0.0, 0.0, -1.0), (0.0, 0.0, 1.0)

    def reward(self) -> float:
        reward = 0.0
        self.gripper_position = self.robot.arm.get_tip().get_position()
        self.object_position = self.target_object.get_position()
        self.basket_position = self.dropin_box.get_position()
        grasped_object = self.robot.gripper.get_grasped_objects()
        object_grasped = grasped_object and grasped_object[0] == self.target_object
        distance_to_basket = np.linalg.norm(self.object_position - self.basket_position)
        distance_to_object = np.linalg.norm(
            self.gripper_position - self.object_position
        )
        if not object_grasped:
            if self.first_grasped:
                if (self.prev_dist_to_basket - distance_to_basket) < 0:
                    reward = -1
                else:
                    reward = 1
            else:
                if (self.prev_dist_to_object - distance_to_object) < 0:
                    reward = -1
                else:
                    reward = 1
        else:
            if not self.first_grasped:
                reward = 50
                self.first_grasped = True
            if (self.prev_dist_to_basket - distance_to_basket) < 0:
                reward += -1
            else:
                reward += 1
        self.prev_dist_to_basket = distance_to_basket
        self.prev_dist_to_object = distance_to_object
        success, term = self.success()
        if success and term:
            return 500.0  # Bonus for completing the task
        elif not success and term:
            return -500.0  # Penalty for failing to complete the task

        return reward  # Small constant penalty to encourage efficiency
