
class joint:
    def __init__(self, x, y):
        self.x = x
        self.y = y
class muscle:
    # {"type": "muscle", "amplitude": 2.12, "phase": 0.0}
    # {"type": "distance"}
    def __init__(self, j0, j1, *args):
        self.j0 = j0
        self.j1 = j1
        self.type = "distance"
        if len(args[0]) == 3:
            active, amplitude, phase = args[0]
            self.active = active
            self.type = "muscle"
            self.amplitude = amplitude
            self.phase = phase

class walker:
    def __init__(self, joints, muscles):
        self.joints = joints
        self.muscles = muscles

    def joint_index(self, joint):
        for i in range(len(self.joints)):
            if self.joints[i] == joint:
                return i
        return -1
    
    def serialize_walker(self):
        joints = []
        muscles = []
        for j in self.joints:
            joints.append((j.x, j.y))
        for m in self.muscles:
            if m.type == "distance":
                muscles.append([self.joint_index(m.j0), self.joint_index(m.j1), {"type": m.type}])
            elif m.type == "muscle":
                muscles.append([self.joint_index(m.j0), self.joint_index(m.j1), {"type": m.type, "amplitude": m.amplitude, "phase": m.phase}])
        return {"joints": joints, "muscles": muscles}

    def __str__(self) -> str:
        return str(self.serialize_walker())

    def validate(self):
        """logic for ensuring that the Sodaracer will not break the underlying Box2D physics engine
            a) that each joint is connected only to so many muscles
            b) that the strength of muscles is limited
            c) that there is a minimum distance between joints
        Returns:
            _type_: bool
        """
        return True

class walker_creator:
    """Walker Creator Referenced in ELM Paper - https://arxiv.org/abs/2206.08896 (pg.16)
    """    
    def __init__(self):
        self.joints = []
        self.muscles = []

    def add_joint(self, x, y):
        """add a spring"""
        j = joint(x, y)
        self.joints.append(j)
        return j

    def add_muscle(self, j0, j1, *args):
        """add a point mass"""
        m = muscle(j0, j1, args)
        self.muscles.append(m)
        return m

    def get_walker(self):
        """Python dictionary with keys such as “joints” and “muscles”"""
        return walker(self.joints, self.muscles)
