# Encoding Sodaracers with Python

## Walker creator
> "while a Sodaracer can be constructed in Python by directly adding elements to a Python dictionary with keys such as “joints” and “muscles,” a more Pythonic interface (which was more effective and is what is used in the experiments) is to create a simple class with two methods: “add joint” (to add a spring) and “add muscle” (to add a point mass.) The idea is that such an interface (here encapsulated in a class called “walker creator”) is closer to the training distribution of Python code (although still no Sodarace examples in this format exist)"
* `walk_creator.py`

## Hand-designed seeds
> "Upon initialization, a single hand-designed solution is evaluated and placed into the map."
* `square.py`
* `radial.py`
* `CPPN_fixed.py`
* `CPPN_mutable.py`

## Intermediate Sodaracer representation
> "its translation after being executed into a dictionary of joints and muscles"

`walker_creator.get_walker()` returns a `Walker` class which can be turned into the necessary dictionary using `Walker.to_dict()`
* `test.py`
