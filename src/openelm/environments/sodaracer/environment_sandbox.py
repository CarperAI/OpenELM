from openelm.environments.sodaracer.simulator import IESoRWorld
from openelm.environments.sodaracer.walker.runner import (
    make_walker as make_walker_galloper,
)
from Box2D.examples.framework import main


# demo class and script (inherits from original simulator.py class)
class DemoLoadWalkerWorld(IESoRWorld):
    def __init__(self, canvas_size: tuple[int, int] = (200, 150)):
        super().__init__(canvas_size)
        # here we import an example make_walker method, and load it to world
        self.load_body_into_world(make_walker_galloper().to_dict())


if __name__ == "__main__":
    # run with `python environment_sandbox.py backend=pygame`
    main(DemoLoadWalkerWorld)