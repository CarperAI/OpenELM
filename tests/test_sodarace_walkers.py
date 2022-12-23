from openelm.environments.sodaracer.walker import Walker
from openelm.environments.sodaracer.walker.CPPN_fixed import (
    make_walker as make_walker_cppn_fixed,
)
from openelm.environments.sodaracer.walker.CPPN_mutable import (
    make_walker as make_walker_cppn_mutable,
)
from openelm.environments.sodaracer.walker.radial import make_walker as make_walker_radial
from openelm.environments.sodaracer.walker.square import make_walker as make_walker_square


def test_seed_walkers():
    """Test that the Sodarace walkers build correctly."""
    # Test that the walkers build correctly
    square_walker: Walker = make_walker_square()
    radial_walker: Walker = make_walker_radial()
    cppn_fixed_walker: Walker = make_walker_cppn_fixed()
    cppn_mutable_walker: Walker = make_walker_cppn_mutable()

    # Test that the walkers have the correct number of joints and muscles
    assert (len(square_walker.joints), len(square_walker.muscles)) == (5, 8)
    assert (len(radial_walker.joints), len(radial_walker.muscles)) == (9, 16)
    assert (len(cppn_fixed_walker.joints), len(cppn_fixed_walker.muscles)) == (24, 96)
    assert (len(cppn_mutable_walker.joints), len(cppn_mutable_walker.muscles)) == (
        24,
        96,
    )

    # Test muscle loading
    correct_distance_joint = [
        0,
        1,
        {"type": "distance", "amplitude": 0.0, "phase": 0.0},
    ]
    correct_muscle = [2, 4, {"type": "muscle", "amplitude": 2.0, "phase": 0.0}]
    assert square_walker.muscles[0] == correct_distance_joint
    assert square_walker.muscles[-1] == correct_muscle

    # Test full dictionary
    correct_square_dict: dict = {
        "joints": [(0, 0), (0, 10), (10, 10), (10, 0), (5, 5)],
        "muscles": [
            [0, 1, {"type": "distance", "amplitude": 0.0, "phase": 0.0}],
            [1, 2, {"type": "distance", "amplitude": 0.0, "phase": 0.0}],
            [2, 3, {"type": "distance", "amplitude": 0.0, "phase": 0.0}],
            [3, 0, {"type": "distance", "amplitude": 0.0, "phase": 0.0}],
            [3, 4, {"type": "distance", "amplitude": 0.0, "phase": 0.0}],
            [0, 4, {"type": "muscle", "amplitude": 5.0, "phase": 0.0}],
            [1, 4, {"type": "muscle", "amplitude": 10.0, "phase": 0.0}],
            [2, 4, {"type": "muscle", "amplitude": 2.0, "phase": 0.0}],
        ],
    }
    assert square_walker.to_dict() == correct_square_dict
    # Test validation
    assert square_walker.validate()
    assert radial_walker.validate()
    assert cppn_fixed_walker.validate()
    assert cppn_mutable_walker.validate()
