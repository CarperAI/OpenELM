import math
from dataclasses import dataclass
from itertools import combinations


@dataclass
class Walker:
    joints: list[tuple[float, float]]
    muscles: list[list]

    def to_dict(self):
        return self.__dict__

    def validate(self) -> bool:
        """
        Validate the Walker.

        logic for ensuring that the Sodaracer will not break the underlying
        Box2D physics engine
            a) that the strength of muscles is limited
            b) that each joint is connected only to so many muscles
            c) that there is a minimum distance between joints

        Returns:
            bool: Whether the walker is valid or not
        """
        max_muscles_per_joint: int = 10
        max_muscle_strength: int = 10
        min_joint_distance: float = 0.1
        for m in self.muscles:
            # Check a) that the strength of muscles is limited
            if m[-1]["type"] == "muscle":
                if m[-1]["amplitude"] > max_muscle_strength:
                    return False
        # Check b) that each joint is connected only to so many muscles
        for j in self.joints:
            joint_idx: int = self.joints.index(j)
            count: int = 0
            for m in self.muscles:
                count += m[:2].count(joint_idx)
            if count > max_muscles_per_joint:
                return False
        for j1, j2 in combinations(self.joints, r=2):
            # Check c) that there is a minimum distance between joints
            if (
                math.sqrt(((j1[0] - j2[0]) ** 2 + (j1[1] - j2[1]) ** 2))
                < min_joint_distance
            ):
                return False
        return True


class walker_creator:
    """
    Walker Creator Referenced in ELM Paper.

    See https://arxiv.org/abs/2206.08896 (pg.16).
    """

    def __init__(self):
        self.joints: list[tuple[float, float]] = []
        self.muscles: list[list] = []

    def add_joint(self, x: float, y: float) -> tuple[float, float]:
        """Add a spring/joint to the sodaracer."""
        j: tuple[float, float] = (x, y)
        self.joints.append(j)
        return j

    def add_muscle(
        self,
        j0: tuple[float, float],
        j1: tuple[float, float],
        amplitude: float = 0.0,
        phase: float = 0.0,
    ) -> list:
        """Add muscle to sodaracer."""
        # Should raise ValueError in caller if `index` fails.
        muscle_data: list = [self.joints.index(j0), self.joints.index(j1)]
        muscle_data.append(
            {
                "type": "distance",
                "amplitude": amplitude,
                "phase": phase,
            }
        )
        if amplitude != 0.0 or phase != 0.0:
            muscle_data[2]["type"] = "muscle"
        self.muscles.append(muscle_data)
        return muscle_data

    def get_walker(self) -> Walker:
        """Python dictionary with keys such as 'joints' and 'muscles'."""
        return Walker(self.joints, self.muscles)
