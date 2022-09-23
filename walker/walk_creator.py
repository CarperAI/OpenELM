class joint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class muscle:
    # {"type": "muscle", "amplitude": 2.12, "phase": 0.0}
    # {"type": "distance"}
    def __init__(self, j0, j1, m_type="distance", amplitude=0.0, phase=0.0):
        self.j0 = j0
        self.j1 = j1
        self.type = m_type
        self.amplitude = amplitude
        self.phase = phase

class walker:
    def __init__(self, joints, muscles):
        self.joints = joints
        self.muscles = muscles

# walker_creator class #
class walker_creator:
    def __init__(self):
        self.joints = []
        self.muscles = []

    def add_joint(self, x, y):
        """add a spring"""
        j = joint(x, y)
        self.joints.append(j)
        return j

    def add_muscle(self, j0, j1, active=True, length=0.0, strength=0.0):
        """add a point mass"""
        m = muscle(j0, j1, active, length, strength)
        self.muscles.append(m)

    def get_walker(self):
        """Python dictionary with keys such as “joints” and “muscles”"""
        return walker(self.joints, self.muscles)

    def validate_walker(self, walker):
        """logic for ensuring that the Sodaracer will not break the underlying Box2D physics engine
            a) that each joint is connected only to so many muscles
            b) that the strength of muscles is limited
            c) that there is a minimum distance between joints
        Returns:
            _type_: bool
        """
        return True
