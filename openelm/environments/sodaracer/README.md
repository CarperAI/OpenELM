# PyIeSOR Simulation

Python version of [iesor-physics](https://github.com/OptimusLime/iesor-physics)

## Setup

Get https://github.com/pybox2d/pybox2d

Also if you want to view what's happening from a specific json,

```bash
pip install -r requirements_test.txt
```

## How to use

```python
import simulator
import json

myWorld = simulator.IESoRWorld()
with open('example_bodies/jsBody142856.json', 'r') as f:
    initial_dict = myWorld.load_body_into_world(json.load(f), myWorld.canvas)
# 350 is the maximum, so just run it a bunch...
[myWorld.updateWorld(350) for i in range(1000)]
start = initial_dict['startX']
# Could potentially just grab the bones instead, but if it's all
# muscle then you'd need to grab a muscle.
end = min(
    [bone.joint.bodyA.position.x for bone in myWorld.bone_list] +
    [muscle.joint.bodyA.position.x for muscle in myWorld.muscle_list]
)
print(start, end - initial_dict['offsetX'])
```

to simulate your world.

if you want to view it:
```bash
python -m framework_simulator
```

## TODO

- Testing to compare with C++ results
