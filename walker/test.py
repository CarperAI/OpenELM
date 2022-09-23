from square import make_walker

# Serialized Square Seed #
output_seed = make_walker()  # translation after being executed into a dictionary of joints and muscles

# Test the output format from the paper
expected_seed = {
    "joints": [(0, 0), (0, 10), (10, 10), (10, 0), (5, 5)],
    "muscles": [
    [0, 1, {"type": "distance"}],
    [1, 2, {"type": "distance"}],
    [2, 3, {"type": "distance"}],
    [3, 0, {"type": "distance"}],
    [3, 4, {"type": "distance"}],
    [0, 4, {"type": "muscle", "amplitude": 2.12, "phase": 0.0}],
    [1, 4, {"type": "muscle", "amplitude": 2.12, "phase": 0.0}],
    [2, 4, {"type": "muscle", "amplitude": 2.12, "phase": 0.0}],
    ],
}

print("Working Serialization: ", output_seed == expected_seed)