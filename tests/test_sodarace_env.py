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
        "width": 413.2231404958677,
        "height": 240.0,
        "startX": 0.0,
        "offsetX": -106.61157024793386,
        "startY": -216.0,
        "offsetY": 171.0,
        "mass": 1136.0864722563908,
    }

    square_start, square_end = run_world(square_world, square_dict)
    assert (square_start, square_end) == (0.0, 7.27755535535573)

    radial_dict = radial_world.load_body_into_world(radial_walker.to_dict())

    assert radial_dict == {
        "width": 413.2228784561031,
        "height": 239.99961951834635,
        "startX": 0.0002620397645793254,
        "offsetX": -106.61170126781613,
        "startY": -215.99996195182018,
        "offsetY": 171.00015219264702,
        "mass": 1182.4111517904973,
    }

    radial_start, radial_end = run_world(radial_world, radial_dict)
    assert (radial_start, radial_end) == (0.0002620397645793254, 95.02245400562137)

    cppn_fixed_dict = cppn_fixed_world.load_body_into_world(cppn_fixed_walker.to_dict())

    assert cppn_fixed_dict == {
        "width": 433.8842975206611,
        "height": 72.0,
        "startX": 0.0,
        "offsetX": -116.94214876033055,
        "startY": -48.0,
        "offsetY": 87.0,
        "mass": 2796.6059242043043,
    }

    cppn_fixed_start, cppn_fixed_end = run_world(cppn_fixed_world, cppn_fixed_dict)

    assert (cppn_fixed_start, cppn_fixed_end) == (0.0, 37.73778779841649)

    cppn_mutable_dict = cppn_mutable_world.load_body_into_world(
        cppn_mutable_walker.to_dict()
    )

    assert cppn_mutable_dict == {
        "width": 433.8842975206611,
        "height": 72.0,
        "startX": 0.0,
        "offsetX": -116.94214876033055,
        "startY": -48.0,
        "offsetY": 87.0,
        "mass": 2796.6059242043043,
    }

    cppn_mutable_start, cppn_mutable_end = run_world(
        cppn_mutable_world, cppn_mutable_dict
    )

    assert (cppn_mutable_start, cppn_mutable_end) == (0.0, 29.67882020808446)


def test_square_draw_list(square_walker_dict):
    square_walker: Walker = make_walker_square()
    square_world = IESoRWorld()
    square_dict = square_world.load_body_into_world(square_walker.to_dict())

    _, _ = run_world(square_world, square_dict)
    square_draw_list_dict = ast.literal_eval(square_world.get_world_json())
    assert square_draw_list_dict == square_walker_dict

# TODO: Remake cppn json
# def test_cppn_fixed_draw_list(cppn_fixed_walker_dict):
#     cppn_fixed_walker: Walker = make_walker_cppn_fixed()
#     cppn_fixed_world = IESoRWorld()
#     cppn_fixed_dict = cppn_fixed_world.load_body_into_world(cppn_fixed_walker.to_dict())

#     _, _ = run_world(cppn_fixed_world, cppn_fixed_dict)
#     cppn_draw_list_dict = ast.literal_eval(cppn_fixed_world.get_world_json())

#     assert cppn_draw_list_dict == cppn_fixed_walker_dict
