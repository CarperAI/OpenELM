import ast

from openelm.environments.sodaracer.simulator import IESoRWorld
from openelm.environments.sodaracer.walker import Walker
from openelm.environments.sodaracer.walker.CPPN_fixed import (
    make_walker as make_walker_cppn_fixed,
)
from openelm.environments.sodaracer.walker.CPPN_mutable import (
    make_walker as make_walker_cppn_mutable,
)
from openelm.environments.sodaracer.walker.radial import (
    make_walker as make_walker_radial,
)
from openelm.environments.sodaracer.walker.square import (
    make_walker as make_walker_square,
)


def run_world(myWorld, initial_dict):
    # for i in range(100):
    #     myWorld.update_world(350)
    #     end = min(
    #         [bone.joint.bodyA.position.x for bone in myWorld.bone_list]
    #         + [muscle.joint.bodyA.position.x for muscle in myWorld.muscle_list]
    #     )
    #     # print(abs(end + initial_dict["offsetX"]))
    [myWorld.update_world(350) for i in range(1000)]
    start = initial_dict["startX"]
    # Could potentially just grab the bones instead, but if it's all
    # muscle then you'd need to grab a muscle.
    end = min(
        [bone.joint.bodyA.position.x for bone in myWorld.bone_list]
        + [muscle.joint.bodyA.position.x for muscle in myWorld.muscle_list]
    )
    return start, abs(end + initial_dict["offsetX"])


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

    square_dict = square_world.load_body_into_world(square_walker.to_dict())
    assert square_dict == {
        "width": 68.18181818181817,
        "height": 70.4,
        "startX": 0.0,
        "startY": 32.0,
        "offsetX": 40.909090909090914,
        "offsetY": 32.8,
        "mass": 241.58651363350822,
    }

    square_start, square_end = run_world(square_world, square_dict)
    assert (square_start, square_end) == (0.0, 81.81081667813388)

    radial_dict = radial_world.load_body_into_world(radial_walker.to_dict())
    assert radial_dict == {
        "width": 64.83262907589534,
        "height": 68.62412927885356,
        "startX": 3.349189105922829,
        "startY": 32.90044749507858,
        "offsetX": 39.2344963561295,
        "offsetY": 32.78748786549464,
        "mass": 252.54982123779916,
    }

    radial_start, radial_end = run_world(radial_world, radial_dict)
    assert (radial_start, radial_end) == (3.349189105922829, 87.38454747307286)

    cppn_fixed_dict = cppn_fixed_world.load_body_into_world(cppn_fixed_walker.to_dict())
    assert cppn_fixed_dict == {
        "width": 71.59090909090908,
        "height": 21.120000000000005,
        "startX": 0.0,
        "startY": 32.0,
        "offsetX": 39.20454545454546,
        "offsetY": 57.44,
        "mass": 572.6414986337938,
    }

    cppn_fixed_start, cppn_fixed_end = run_world(cppn_fixed_world, cppn_fixed_dict)
    assert (cppn_fixed_start, cppn_fixed_end) == (0.0, 88.8606581254439)

    cppn_mutable_dict = cppn_mutable_world.load_body_into_world(
        cppn_mutable_walker.to_dict()
    )
    assert cppn_mutable_dict == {
        "width": 71.59090909090908,
        "height": 21.120000000000005,
        "startX": 0.0,
        "startY": 32.0,
        "offsetX": 39.20454545454546,
        "offsetY": 57.44,
        "mass": 572.6414986337938,
    }

    cppn_mutable_start, cppn_mutable_end = run_world(
        cppn_mutable_world, cppn_mutable_dict
    )
    assert (cppn_mutable_start, cppn_mutable_end) == (0.0, 458.53709550337356)


def test_square_draw_list(square_walker_dict):
    square_walker: Walker = make_walker_square()
    square_world = IESoRWorld()
    square_dict = square_world.load_body_into_world(square_walker.to_dict())

    _, _ = run_world(square_world, square_dict)
    square_draw_list_dict = ast.literal_eval(square_world.get_world_json())

    assert square_draw_list_dict == square_walker_dict


def test_cppn_fixed_draw_list(cppn_fixed_walker_dict):
    cppn_fixed_walker: Walker = make_walker_cppn_fixed()
    cppn_fixed_world = IESoRWorld()
    cppn_fixed_dict = cppn_fixed_world.load_body_into_world(cppn_fixed_walker.to_dict())

    _, _ = run_world(cppn_fixed_world, cppn_fixed_dict)
    cppn_draw_list_dict = ast.literal_eval(cppn_fixed_world.get_world_json())

    assert cppn_draw_list_dict == cppn_fixed_walker_dict
