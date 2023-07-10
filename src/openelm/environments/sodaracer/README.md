# PyIeSOR Simulation

Python version of [iesor-physics](https://github.com/OptimusLime/iesor-physics)

## Setup

Get https://github.com/pybox2d/pybox2d

Use pygame to view your world.
Install the requirements from the OpenELM base directory:

```bash
pip install -e .[sodaracer]
```

View your world:
python simulator.py --backend=pygame

## Loading and visualizing walker on terrain

Run the example script:
```bash
python environment_sandbox.py --backend=pygame
```

Which contains something like:

```python
from openelm.environments.sodaracer.simulator import IESoRWorld
from openelm.environments.sodaracer.walker.runner import (
    make_walker as make_walker_runner,
)
from Box2D.examples.framework import main


# demo class and script (inherits from original simulator.py class)
class DemoLoadWalkerWorld(IESoRWorld):
    def __init__(self, canvas_size: tuple[int, int] = (200, 150)):
        super().__init__(canvas_size)
        # here we import an example make_walker method, and load it to world
        self.load_body_into_world(make_walker_runner().to_dict())


if __name__ == "__main__":
    # run with `python environment_sandbox.py backend=pygame`
    main(DemoLoadWalkerWorld)
```

To load a different walker on the flat terrain (aside from the `runner`), you can change the second import to load the desired walker from implemented walkers.

## TODO

- Testing to compare with C++ results
