"""
Sodarace simualator.

Mostly just https://github.com/OptimusLime/iesor-physics translated to Python.

See here for a Box2D overview:
https://github.com/pybox2d/cython-box2d/blob/master/docs/source/getting_started.md

"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from Box2D import Box2D as b2

from openelm.environments.sodaracer.box2d_examples.framework import Framework, main
from openelm.environments.sodaracer.helpers import (
    Bone,
    DistanceAccessor,
    Entity,
    Muscle,
)


class IESoRWorld(Framework):
    """Class for the Sodarace simulation."""

    name: "IESoRWorld"

    def __init__(self, canvas_size: tuple[int, int] = (150, 200)):
        # super(IESoRWorld, self).__init__()
        """
        Initialize the world.

        Args:
            canvas_size (tuple[int, int], optional): Size of the world canvas.
                Defaults to (200, 150).
        """
        self.interpolation: float = 0.0
        self.radians: float = 0.0
        self.accumulator: float = 0.0
        self.desiredFPS: float = 60.0
        self.simulation_rate: float = 1.0 / self.desiredFPS

        self.static_body_count: int = 0
        # keep a list of body identifiers
        self.body_list: list = []
        # keep a list of shape identifiers
        self.shape_list: list = []
        # keep a reference for every body according to IDs
        self.body_map: dict = {}
        self.bone_list: list = []
        self.muscle_list: list = []

        self.gravity = b2.b2Vec2(0.0, -25.0)  # ??? Magic numbers.
        # Construct a world object, which will hold and simulate the rigid bodies.

        # self.world.gravity = self.gravity
        self.world: b2.b2World = b2.b2World(self.gravity)
        self.world.autoClearForces = False

        self.groundBodyDef: b2.b2BodyDef = b2.b2BodyDef()
        self.groundBodyDef.type = b2.b2_staticBody
        self.groundBodyDef.position = b2.b2Vec2(0.0, -100.0)

        # Add ground body to the world
        self.groundBody: b2.b2Body = self._add_body_to_world(
            "ground", self.groundBodyDef
        )
        # Define the ground box shape.
        self.groundBox: b2.b2PolygonShape = b2.b2PolygonShape()
        # The extents are the half-widths of the box.
        self.groundBox.SetAsBox(3500.0, 10.0)
        # Add the ground fixture to the ground body.
        self._add_shape_to_body(self.groundBody, self.groundBox)

        self.canvas: b2.b2Vec2 = b2.b2Vec2(canvas_size[0], canvas_size[1])

    def get_world_json(self) -> str:
        """Generate a JSON string of the world and every entity in it."""
        root: dict = {}
        shapes: dict = {}
        for body in self.world.bodies:  # type: b2.b2Body
            body_id: str = body.userData
            if body_id is None:
                continue

            fixtures_list = body.fixtures
            for fixture in fixtures_list:  # type: b2.b2Fixture
                single_shape: dict[str, Any] = {}

                if fixture.type == b2.b2Shape.e_circle:
                    circle: b2.b2CircleShape = fixture.shape
                    single_shape["type"] = "circle"
                    single_shape["bodyOffset"] = {
                        "x": body.position.x,
                        "y": body.position.y,
                    }
                    single_shape["rotation"] = body.angle
                    single_shape["center"] = {"x": circle.pos.x, "y": circle.pos.y}
                    single_shape["radius"] = circle.radius
                    single_shape["color"] = "#369"

                elif fixture.type == b2.b2Shape.e_polygon:
                    poly: b2.b2PolygonShape = fixture.shape
                    single_shape["type"] = "polygon"
                    single_shape["points"] = [
                        {"x": point[0], "y": point[1]} for point in poly.vertices
                    ]
                    single_shape["bodyOffset"] = {
                        "x": body.position.x,
                        "y": body.position.y,
                    }
                    single_shape["rotation"] = body.angle
                    single_shape["color"] = "#38F"

                full_id = body_id + fixture.userData
                shapes[full_id] = single_shape

        root["shapes"] = shapes

        joints = {}
        for joint in self.world.joints:  # type: b2.b2Joint
            if joint.userData is None:
                continue

            single_joint = {}
            # Both body ids need to exist
            if (joint.bodyA.userData is not None) and (
                joint.bodyB.userData is not None
            ):
                single_joint["sourceID"] = joint.bodyA.userData
                single_joint["targetID"] = joint.bodyB.userData
                # Drawing information for offsets on the body
                single_joint["sourceOffset"] = {
                    "x": joint.anchorA.x,
                    "y": joint.anchorA.y,
                }
                single_joint["targetOffset"] = {
                    "x": joint.anchorB.x,
                    "y": joint.anchorB.y,
                }
            joints[joint.userData.id] = single_joint

        root["joints"] = joints
        return json.dumps(root, default=vars)

    def Step(self, settings):
        """
        The main physics step.

        Takes care of physics drawing (callbacks are executed after the
        world.Step()) and drawing additional information.
        """
        # self.stepCount += 1
        # # Don't do anything if the setting's Hz are <= 0
        if settings.hz > 0.0:
            timeStep = 1.0 / settings.hz
        else:
            timeStep = 0.0
        self.update_world(timeStep * 1000.0)
        settings.velocityIterations = 10
        settings.positionIterations = 10
        settings.hz = 0.0
        super().Step(settings)
        self.world.ClearForces()

    def update_world(self, ms_update: float) -> int:
        """
        Simulate the world for a given amount of time.

        Args:
            ms_update (float): The amount of time you want to simulate in milliseconds.

        Returns:
            int: The number of times the physics engine was run (step count).
        """
        # Pass in the amount of time you want to simulate in real time, this is
        # the delta time between calls to update.
        # In simulation time, this can be any amount -- but this function
        # won't return until all updates are processed, so it holds up the thread.
        ms_update *= 4  # why...
        # Number of simulation steps during this update
        step_count: int = 0
        # Number of seconds since the last frame
        frame_time: float = ms_update / 1000.0
        # # Set maximum frame time
        frame_time = 0.35 if frame_time > 0.35 else frame_time
        # Accumulate all the time we haven't rendered things in
        self.accumulator += frame_time
        # Aas long as we have a chunk of time to simulate, we need to do the simulation
        while self.accumulator >= self.simulation_rate:
            # Track number of times we run the physics engine
            step_count += 1
            # Move the muscles quicker using this toggle
            speedup: float = 3.0
            # We loop through all our muscles, and update the lengths associated
            # with the connections
            for muscle in self.muscle_list:
                # Get our distance joint -- holder of physical joint info
                distance_joint: b2.b2Joint = muscle.joint
                # Fetch the original length of the distance joint, and add some
                # fraction of that amount to the length, depending on the current
                # location in the muscle cycle
                length_calc = muscle.original_length + (
                    muscle.amplitude
                    * math.cos(self.radians + muscle.phase * 2 * math.pi)
                )
                # Set our length as the calculate value for this joint
                DistanceAccessor.setLength(distance_joint, length_calc)
            # Frame rate, velocity iterations, position iterations
            self.world.Step(self.simulation_rate, 10, 10)
            # Manually clear forces when doing fixed time steps,
            # NOTE: that we disabled auto clear after step inside of the b2World
            self.world.ClearForces()
            # Increment the radians for the muscles
            self.radians += speedup * self.simulation_rate
            # Decrement the accumulator - we ran a chunk just now!
            self.accumulator -= self.simulation_rate
        # Interpolation is basically a measure of how far into the next step we
        # should have simulated, it's meant to ease the drawing stages.
        # (you linearly interpolate the last frame and the current frame using this value)
        self.interpolation = self.accumulator / self.simulation_rate
        # Return the number of times we ran the simulator
        return step_count

    def _add_body_to_world(self, body_id: str, body_def: b2.b2BodyDef) -> b2.b2Body:
        """
        Helper function to add a body to the world.

        Args:
            body_id (str): Unique body id.
            body_def (b2.b2BodyDef): Body definition object.

        Returns:
            b2.b2Body: Body object.
        """
        body: b2.b2Body = self.world.CreateBody(body_def)
        # Store body id and body
        self.body_list.append(body_id)
        self.body_map[body_id] = body
        body.userData = body_id
        return body

    def _add_shape_to_body_fixture(
        self, body: b2.b2Body, fix_def: b2.b2FixtureDef
    ) -> b2.b2Fixture:
        """
        Helper function to add a specified shape as a fixture to a body.

        Args:
            body (b2.b2Body): Body object.
            fix_def (b2.b2FixtureDef): Fixture definition object.

        Returns:
            b2.b2Fixture: Fixture object.
        """
        # Create physics id based on global shape count
        shape_id = self.create_shape_id()
        fixture: b2.b2Fixture = body.CreateFixture(fix_def)
        fixture.userData = shape_id
        return fixture

    def _add_shape_to_body(
        self, body: b2.b2Body, shape: b2.b2Shape, density: float = 0.0
    ) -> b2.b2Fixture:
        """
        Helper function to add a specified shape to a body.

        Args:
            body (b2.b2Body): Body object.
            shape (b2.b2Shape): Shape to add
            density (float): Density of the shape. Default value is 0.0

        Returns:
            b2.b2Fixture: Fixture object.
        """
        # Create physics id based on global shape count
        shape_id = self.create_shape_id()
        fixture: b2.b2Fixture = body.CreateFixture(shape=shape, density=density)
        fixture.userData = shape_id
        return fixture

    def load_body_into_world(self, body: dict) -> dict:
        """
        Load body into world from a dictionary.

        Args:
            body (dict): Dict describing the body to load. The dictionary
                should be a Sodarace walker body, containing lists for both
                "joints" and "muscles".

        Raises:
            e: Exception for any loading error.

        Returns:
            dict: Morphology dictionary containing the body's morphology, including
                the body's height, width, and mass.
        """
        # We use bodycount as a way to identify each physical object being added
        # from the body information
        o_body_count = self.world.bodyCount
        # Morphological properties of the body
        morphology: dict[str, float] = {}
        # Use our body information to create all the body additions to the world
        entities: list[dict[str, Any]] = self._get_body_entities(body, morphology)
        # Map from string ids to entities
        entity_map: dict[str, dict] = {e["bodyID"]: e for e in entities}
        # Add all entities in the body to the world, so that the joints have
        # bodies to connect to
        for entity in entities:
            self.set_body(entity)

        connections = body["muscles"]
        # This determines the cutoff for the connection being fixed (bone)
        # or moving (muscle)
        amplitude_cutoff = 0.2
        # mass = # of nodes + length of connections
        connection_distance_sum = 0.0
        for muscle in connections:
            # To identify a given object in the physical world, we need to start
            # with the current body count, and add the source id number.
            # This allows us to add multiple bodies to the same world
            # (though this is recommended against, since it's easier to create
            # separate worlds)
            source_id = str(o_body_count + muscle[0])
            target_id = str(o_body_count + muscle[1])
            if source_id == target_id:
                continue
            try:
                muscle_properties: dict = muscle[-1]
                # Convert from [-1,1] to [0,1]
                amplitude = (muscle_properties["amplitude"] + 1) / 2.0
                connection_distance = math.sqrt(
                    (entity_map[source_id]["x"] - entity_map[target_id]["x"]) ** 2
                    + (entity_map[source_id]["y"] - entity_map[target_id]["y"]) ** 2
                )
                # Needed to calculate mass
                connection_distance_sum += connection_distance
                properties: dict[str, float] = {}
                if amplitude < amplitude_cutoff:
                    # Add fixed bone connection
                    self.add_distance_joint(source_id, target_id, properties)
                else:
                    # Add moving muscle connection
                    # Gardcoded spring behaviors
                    properties["frequencyHz"] = 3
                    properties["dampingRatio"] = 0.3
                    properties["phase"] = muscle_properties["phase"]
                    # The 0.3 below is some scaling factor based on screen size
                    properties["amplitude"] = 0.3 * amplitude
                    self.add_muscle_joint(source_id, target_id, properties)
            except Exception as e:
                # Not sure if this is needed, but it's here in the original
                print("Oops error", e)
                raise e
        # Add mass to morphology
        morphology["mass"] = morphology["totalNodes"] + connection_distance_sum / 2.0
        # Remove totalNodes since we have mass
        del morphology["totalNodes"]
        return morphology

    def _get_body_entities(
        self, body: dict, initial_morphology: dict
    ) -> list[dict[str, Any]]:
        """
        Helper function to create all bodies in the world.

        Args:
            body (dict): Dict describing the body to load. The dictionary should
                be a Sodarace walker body, containing lists for both
                "joints" and "muscles".
            initial_morphology (dict): Dict to add the initial morphology to.

        Returns:
            list[dict[str, Any]]: Morphology dictionary containing the body's
                height, width, and starting location.
        """
        # First convert our body into a collection of physics objects that need
        # to be added to the environment
        entity_vector: list[dict[str, Any]] = []
        # Use our window width/height for adjusting relative body location

        o_nodes: list = body["joints"]
        # This is the starting body identifier for each node (every node has a
        # unique ID for the source connections)
        # So node 5 is always node 5, but if you already have objects in the
        # physical world, object 5 != node 5, so
        # number of objects + nodeID = true nodeID
        body_id: int = self.world.bodyCount
        # Manipulated values of each node, adjusted for screen coordinates
        x_scaled: float = 0.0
        y_scaled: float = 0.0
        # Adjusting the total size of the object according to window size
        divide_for_max_width: float = 2.2
        divide_for_max_height: float = 2.5

        max_allowed_width: float = self.canvas.x / divide_for_max_width
        max_allowed_height: float = self.canvas.y / divide_for_max_height
        # Starting values, these will be adjusted
        min_x, max_x, min_y, max_y = self.canvas.x, 0.0, self.canvas.y, 0.0

        for node in o_nodes:
            node_x, node_y = node[0], node[1]
            # Here we actually modify the x and y values to fit into a certain
            # sized box depending on the initial canvas size.
            x_scaled = node_x / divide_for_max_width * max_allowed_width
            y_scaled = (1.0 - node_y) / divide_for_max_height * max_allowed_height
            # For each node, we make a circle entity with certain properties,
            # then increment count
            entity_vector.append(
                Entity.circle_entity(
                    str(body_id), x_scaled, y_scaled, max_allowed_width / 35.0, 0
                )
            )
            # What are the min/max x/y values we've seen?
            # Use these to determine how wide/tall the entity is
            min_x = min(min_x, x_scaled)
            max_x = max(max_x, x_scaled)
            min_y = min(min_y, y_scaled)
            max_y = max(max_y, y_scaled)
            body_id += 1
        # We use move_x, move_y to center our entity in world space after creation
        move_x: float = (max_x - min_x) / 2.0
        move_y: float = (max_y - min_y) / 2.0
        # Need to move x, y coordinates for entities to be centered
        for entity in entity_vector:
            entity["x"] = (float(entity["x"]) - min_x) + self.canvas.x / 2 - move_x
            entity["y"] = (float(entity["y"]) - min_y) + self.canvas.y / 2 - move_y
        # Width and height of the object
        initial_morphology["width"] = max_x - min_x
        initial_morphology["height"] = max_y - min_y
        # Starting position?
        initial_morphology["startX"] = min_x
        initial_morphology["startY"] = min_y
        # How much we translated the entity in the x and y direction
        initial_morphology["offsetX"] = -min_x + self.canvas.x / 2 - move_x
        initial_morphology["offsetY"] = -min_y + self.canvas.y / 2 - move_y

        initial_morphology["totalNodes"] = len(o_nodes)
        return entity_vector

    def add_distance_joint(
        self, source_id: str, target_id: str, properties: dict
    ) -> Bone:
        """
        Helper function to add a distance joint between two bodies.

        Args:
            source_id (str): Source body ID.
            target_id (str): Target body ID.
            properties (dict): Dictionary with joint properties, including
                `frequencyHz` and `dampingRatio`.

        Returns:
            Bone: A Bone dataclass object.
        """
        # Connect two body objects together
        body1: b2.b2Body = self.body_map[source_id]
        body2: b2.b2Body = self.body_map[target_id]

        joint = b2.b2DistanceJointDef()
        # Initialize the joints where they're attached at the center of the objects
        joint.Initialize(body1, body2, body1.worldCenter, body2.worldCenter)

        # FrequencyHz determines how the distance joint responds to stretching
        if properties.get("frequencyHz", None) is not None:
            joint.frequencyHz = properties["frequencyHz"]

        if properties.get("dampingRatio", None) is not None:
            joint.dampingRatio = properties["dampingRatio"]

        w_joint = self.world.CreateJoint(joint)

        bone = Bone(str(len(self.bone_list)), w_joint)
        self.bone_list.append(bone)

        # Not sure why this is needed, but it is
        w_joint.userData = bone
        return bone

    def add_muscle_joint(
        self, source_id: str, target_id: str, properties: dict
    ) -> Muscle:
        """
        Helper function to add a muscle joint between two bodies.

        This is a bone (distance joint) with a phase and amplitude.

        Args:
            source_id (str): Source body ID.
            target_id (str): Target body ID.
            properties (dict): Dictionary with properties for the muscle joint,
                including phase and amplitude.

        Returns:
            Muscle: A Muscle dataclass object.
        """
        # Start by adding a standard distance joint
        added_joint = self.add_distance_joint(source_id, target_id, properties)
        # But our muscles have phase and amplitude, which adjust length during updates
        phase = 0 if properties.get("phase", None) is None else properties["phase"]
        # Amplitude is how much change each muscles exhibit
        amplitude = (
            1 if properties.get("amplitude", None) is None else properties["amplitude"]
        )
        muscle = Muscle(str(len(self.muscle_list)), added_joint.joint, amplitude, phase)

        self.muscle_list.append(muscle)
        return muscle

    def set_body(self, entity: dict) -> None:
        """
        Add a body to the world, and register it in the body map.

        Supports circle, polygon, or rectangle entities.

        Args:
            entity (dict): A dictionary containing the entity information.
        """
        # Create a generic body def to hold our body
        body_def = b2.b2BodyDef()
        # This is only called with non-ground entities so the type is dynamic.
        body_def.type = b2.b2_dynamicBody
        # We set the position to the intial x, y coordinates
        body_def.position = b2.b2Vec2(entity["x"], entity["y"])
        # We don't want any rotation on the nodes (they would slide when moving)
        body_def.linearDamping = 1.1
        # Bodies have fixed rotation in iesor.
        body_def.fixedRotation = True
        # All entities appear to have an angle, but just in case:
        if entity.get("angle", None) is not None:
            body_def.angle = entity["angle"]
        # We add our body to the world, and register its identifying information
        # that's useful for drawing
        body: b2.b2Body = self._add_body_to_world(entity["bodyID"], body_def)
        # If we have a radius it's a circle entity.
        if entity.get("radius", None) is not None:
            # This filter prevents circle nodes from colliding on our bodies
            filter = b2.b2Filter()
            filter.categoryBits = 0x0002
            filter.maskBits = 0x0001
            filter.groupIndex = 0

            # We create a fixture def for our new shape
            fixture = b2.b2FixtureDef()
            # We need high density objects so they don't float!
            fixture.density = 25.0
            fixture.friction = 1.0
            # Less bouncing, more staying
            fixture.restitution = 0.1

            node = b2.b2CircleShape()
            node.radius = entity["radius"]
            fixture.shape = node
            fixture.filter = filter

            self._add_shape_to_body_fixture(body, fixture)
        # Now check if the entity is a polygon
        elif entity.get("polyPoints", None) is not None:
            poly = b2.b2PolygonShape()
            poly.vertices = [
                b2.b2Vec2(point["x"], point["y"]) for point in entity["polyPoints"]
            ]
            # Default density to 0
            self._add_shape_to_body(body, poly)
        else:
            # Else this is a rectangle entity
            poly = b2.b2PolygonShape
            # We've got a box, we simply put in the half height/ half width info
            # for our polygon
            poly.SetAsBox(entity["halfWidth"], entity["halfHeight"])
            self._add_shape_to_body(body, poly)

    def create_shape_id(self) -> str:
        """Create a new globally unique shape id, and add it to the shape list."""
        self.static_body_count += 1
        shape_id = str(self.static_body_count)
        self.shape_list.append(shape_id)
        return shape_id


def load_data_file(file_path: Path) -> str:
    """Load object from data file."""
    with open(file_path) as f:
        return f.read()


class SodaraceSimulator:
    """Sodarace simulator class."""

    def __init__(self, body: dict) -> None:
        """
        Initialize the simulator with a body dictionary.

        This simulator only accepts a single sodaracer, to reduce complexity.

        Args:
            body (dict): This dictionary should contain "joints" and "muscles" keys,
                and specify the full location and properties of all entities in
                the body.
        """
        self.world = IESoRWorld()
        self._morphology = self.world.load_body_into_world(body)

    @property
    def morphology(self) -> dict:
        """Get the morphology of the Sodaracer."""
        return self._morphology

    def evaluate(self, time: float) -> float:
        """
        Evaluate the Sodaracer for a given number of timesteps.

        Args:
            time (float): Amount of time (in milliseconds) for which to evaluate
                the Sodaracer. This will be rounded down to the nearest 350ms.

        Returns:
            float: The absolute distance traveled by the Sodaracer.
        """
        for _ in range(int(time // 350)):
            self.world.update_world(350)

        _ = self.morphology["startX"]
        try:
            end = min(
                [bone.joint.bodyA.position[0] for bone in self.world.bone_list]
                + [muscle.joint.bodyA.position[0] for muscle in self.world.muscle_list]
            )
            return abs(end + self.morphology["offsetX"])
        except Exception:
            # print(e)
            # print(self.world.bone_list)
            # print(self.world.muscle_list)
            return -np.inf


if __name__ == "__main__":
    main(IESoRWorld)
