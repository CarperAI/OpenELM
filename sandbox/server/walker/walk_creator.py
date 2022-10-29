import numpy as np
from itertools import combinations


class Muscle:
    # {"type": "muscle", "amplitude": 2.12, "phase": 0.0}
    # {"type": "distance"}
    def __init__(self, j0, j1, *args):
        self.j0 = j0
        self.j1 = j1
        self.type = "distance"
        if len(args[0]) == 3:
            isDistance, amplitude, phase = args[0]
            self.type = "muscle"
            self.amplitude = amplitude
            self.phase = phase


class Walker:
    def __init__(self, joints, muscles):
        self.joints = joints
        self.muscles = muscles

    def joint_index(self, joint):
        for i in range(len(self.joints)):
            if np.array_equal(self.joints[i], joint):
                return i
        return -1

    def serialize_walker(self):
        joints = []
        muscles = []
        for j in self.joints:
            joints.append((j[0], j[1]))
        for m in self.muscles:
            if m.type == "distance":
                muscles.append([self.joint_index(m.j0), self.joint_index(m.j1), {"type": m.type}])
            elif m.type == "muscle":
                muscles.append([self.joint_index(m.j0), self.joint_index(m.j1),
                                {"type": m.type, "amplitude": m.amplitude, "phase": m.phase}])
        return {"joints": joints, "muscles": muscles}

    def serialize_walker_sodarace(self):
        walker_dict = {
            "useLEO": True,
            "nodes": [
            ],
            "connections": [
            ],
        }
        for j in self.joints:
            walker_dict["nodes"].append({
                "x": j[0],
                "y": j[1],
            })
        for m in self.muscles:
            if m.type == "distance":
                walker_dict['connections'].append({
                    "sourceID": self.joint_index(m.j0),
                    "targetID": self.joint_index(m.j1),
                    "cppnOutputs": [0, 0, 0, -10.0]
                })
            elif m.type == "muscle":
                walker_dict['connections'].append({
                    "sourceID": self.joint_index(m.j0),
                    "targetID": self.joint_index(m.j1),
                    "cppnOutputs": [0, 0, m.phase, m.amplitude]
                })
        return walker_dict

    def __str__(self):
        return str(self.serialize_walker())

    def __eq__(self, other):
        # Check if other is a Dictionary
        if isinstance(other, dict):
            return self.serialize_walker() == other
        return self == other

    def validate(self):
        """logic for ensuring that the Sodaracer will not break the underlying Box2D physics engine
            a) that each joint is connected only to so many muscles
            b) that the strength of muscles is limited
            c) that there is a minimum distance between joints
        Returns:
            _type_: bool
        """
        max_muscles_per_joint = 10
        max_muscle_strength = 10
        min_joint_distance = 0.1
        for j in self.joints:
            count = 0
            for m in self.muscles:
                if np.array_equal(m.j0, j) or np.array_equal(m.j1, j):
                    count += 1
                # Check b) that the strength of muscles is limited
                if m.type == "muscle":
                    if m.amplitude > max_muscle_strength:
                        print("Muscle strength too high")
                        return False
            # Check a) that each joint is connected only to so many muscles
            if count > max_muscles_per_joint:
                print("Too many muscles connected to joint", count)
                return False
        for j1, j2 in combinations(self.joints, r=2):
            # Check c) that there is a minimum distance between joints
            if np.sqrt(np.sum((j1 - j2) ** 2)) < min_joint_distance:
                print("Joints too close together", j1, j2, self.joint_index(j1), self.joint_index(j2))
                return False
        return True


class walker_creator:
    """Walker Creator Referenced in ELM Paper - https://arxiv.org/abs/2206.08896 (pg.16)
    """
    def __init__(self):
        self.joints = []
        self.muscles = []

    def add_joint(self, x, y):
        """add a spring"""
        j = [x, y]
        self.joints.append(j)
        return j

    def add_muscle(self, j0, j1, *args):
        """add a point mass"""
        m = Muscle(j0, j1, args)
        self.muscles.append(m)
        return m

    def get_walker(self):
        """Python dictionary with keys such as 'joints' and 'muscles'"""
        return Walker(np.array(self.joints), self.muscles)
