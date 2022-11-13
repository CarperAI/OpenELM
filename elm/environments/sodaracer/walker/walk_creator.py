import math
from dataclasses import dataclass, field
from itertools import combinations


@dataclass
class Muscle:
    joints: list[tuple[float, float]]
    type: str = "distance"
    amplitude: float = 0.0
    phase: float = 0.0
    source_id: int = 0
    target_id: int = 0

    def __post_init__(self):
        if self.amplitude == 0.0 and self.phase == 0.0:
            self.type = "muscle"


@dataclass
class Walker:
    joints: list[tuple[float, float]]
    muscles: list[Muscle]
    joint_ids: list[int] = field(default_factory=list)

    def joint_index(self, joint):
        for i in range(len(self.joints)):
            if self.joints[i][0] == joint[0] and self.joints[i][1] == joint[1]:
                return i
        return -1

    def __post_init__(self):
        for m in self.muscles:
            m.source_id = self.joint_index(m.joints[0])
            m.target_id = self.joint_index(m.joints[1])

    def validate(self) -> bool:
        """logic for ensuring that the Sodaracer will not break the underlying Box2D physics engine
            a) that the strength of muscles is limited
            b) that each joint is connected only to so many muscles
            c) that there is a minimum distance between joints
        Returns:
            _type_: bool
        """
        max_muscles_per_joint: int = 10
        max_muscle_strength: int = 10
        min_joint_distance: float = 0.1
        for j in self.joints:
            count: int = 0
            for m in self.muscles:
                if (
                    m.joints[0][0] == j[0]
                    and m.joints[0][1] == j[1]
                    or m.joints[1][0] == j[0]
                    and m.joints[1][1] == j[1]
                ):
                    count += 1
                # Check a) that the strength of muscles is limited
                if m.type == "muscle":
                    if m.amplitude > max_muscle_strength:
                        print("Muscle strength too high")
                        return False
            # Check b) that each joint is connected only to so many muscles
            if count > max_muscles_per_joint:
                print("Too many muscles connected to joint", count)
                return False
        for j1, j2 in combinations(self.joints, r=2):
            # Check c) that there is a minimum distance between joints
            if (
                math.sqrt(((j1[0] - j2[0]) ** 2 + (j1[1] - j2[1]) ** 2))
                < min_joint_distance
            ):
                print(
                    "Joints too close together",
                    j1,
                    j2,
                    self.joint_index(j1),
                    self.joint_index(j2),
                )
                return False
        return True


class Walker2:
    def __init__(self, joints, muscles):
        self.joints = joints
        self.muscles = muscles

    def __str__(self):
        return str(self.serialize_walker())

    def __eq__(self, other):
        # Check if other is a Dictionary
        if isinstance(other, dict):
            return self.serialize_walker() == other
        return self == other


class walker_creator:
    """Walker Creator Referenced in ELM Paper - https://arxiv.org/abs/2206.08896 (pg.16)"""

    def __init__(self):
        self.joints = []
        self.muscles = []

    def add_joint(self, x, y):
        """
        Add a spring/joint to the sodaracer.

        Joints make up the body of the robot by connecting point masses.
        This implementation

        Args:
            x (_type_): _description_
            y (_type_): _description_

        Returns:
            _type_: _description_
        """
        j = (x, y)
        self.joints.append(j)
        return j

    def add_muscle(self, j0, j1, amplitude=0.0, phase=0.0):
        """
        Add muscle to sodaracer.

        _extended_summary_

        Args:
            j0 (_type_): _description_
            j1 (_type_): _description_

        Returns:
            _type_: _description_
        """
        m = Muscle(joints=[j0, j1], amplitude=amplitude, phase=phase)
        self.muscles.append(m)
        return m

    def get_walker(self):
        """Python dictionary with keys such as 'joints' and 'muscles'"""
        return Walker(self.joints, self.muscles)
