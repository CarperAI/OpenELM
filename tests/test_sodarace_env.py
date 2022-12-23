from openelm.environments.sodaracer.simulator import IESoRWorld
from openelm.environments.sodaracer.walker import Walker
from openelm.environments.sodaracer.walker.CPPN_fixed import (
    make_walker as make_walker_cppn_fixed,
)
from openelm.environments.sodaracer.walker.CPPN_mutable import (
    make_walker as make_walker_cppn_mutable,
)
from openelm.environments.sodaracer.walker.radial import make_walker as make_walker_radial
from openelm.environments.sodaracer.walker.square import make_walker as make_walker_square


def test_sodaracer_eval():
    """Test that the Sodarace seeds give the correct evaluation."""
    square_walker: Walker = make_walker_square()
    radial_walker: Walker = make_walker_radial()
    cppn_fixed_walker: Walker = make_walker_cppn_fixed()
    cppn_mutable_walker: Walker = make_walker_cppn_mutable()

    square_world = IESoRWorld()
    radial_world = IESoRWorld()
    cppn_fixed_world = IESoRWorld()
    cppn_mutable_world = IESoRWorld()

    square_dict = square_world.load_body_into_world(square_walker.to_dict(),
                                                    square_world.canvas)
    assert square_dict == {
        'width': 413.2231404958677,
        'height': 240.0,
        'startX': 0.0,
        'offsetX': -106.61157024793386,
        'startY': -216.0,
        'offsetY': 171.0,
        'mass': 1136.0864722563908
    }

    def run_world(myWorld, initial_dict):
        [myWorld.updateWorld(350) for i in range(1000)]
        start = initial_dict['startX']
        # Could potentially just grab the bones instead, but if it's all
        # muscle then you'd need to grab a muscle.
        end = min(
            [bone.joint.bodyA.position.x for bone in myWorld.bone_list] +
            [muscle.joint.bodyA.position.x for muscle in myWorld.muscle_list]
        )
        return start, abs(end + initial_dict['offsetX'])

    square_start, square_end = run_world(square_world, square_dict)
    assert (square_start, square_end) == (0.0, 7.27755535535573)
