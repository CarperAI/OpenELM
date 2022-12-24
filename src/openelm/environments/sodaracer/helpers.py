from enum import Enum
from typing import Any

import Box2D.Box2D as b2

# omg globals 🤮
# TODO: Figure out a better way than a global, otherwise my code standards guy will be more disappointed in me.
staticBodyCount: int = 0


class EntityType(Enum):
    Circle = 0
    Polygon = 1
    Rectangle = 2


class Entity:
    @staticmethod
    def circle_entity(
        body_id: str, x: float, y: float, radius: float, angle: float
    ) -> dict:
        """
        Create a circle entity.

        Args:
            body_id (str): Unique body id.
            x (float): Center x coordinate.
            y (float): Center y coordinate.
            radius (float): Radius of circle.
            angle (float): Angle of circle.

        Returns:
            dict: A dictionary specifying the circle entity.
        """
        entity: dict[str, Any] = {
            "type": EntityType.Circle,
            "bodyID": body_id,
            "x": x,
            "y": y,
            "radius": radius,
            "angle": angle,
        }
        return entity

    @staticmethod
    def polygon_entity(
        body_id: str,
        x_scale: float,
        y_scale: float,
        points: list[b2.b2Vec2],
        angle: float,
    ) -> dict:
        """
        Create a polygon entity.

        Args:
            body_id (str): Unique body id.
            x_scale (float): Scale of x coordinate.
            y_scale (float): Scale of y coordinate.
            points (list[b2.b2Vec2]): List of points in the polygon.
            angle (float): Angle of polygon.

        Returns:
            dict: A dictionary specifying the polygon entity.
        """
        entity: dict[str, Any] = {
            "type": EntityType.Polygon,
            "bodyID": body_id,
            "x": x_scale,
            "y": y_scale,
            "angle": angle,
            "polyPoints": [
                {"x": point.x, "y": point.y} for point in points
            ]
        }
        return entity

    @staticmethod
    def rectangle_entity(
        body_id: str,
        x_scale: float,
        y_scale: float,
        half_width: float,
        half_height: float,
        angle: float,
    ):
        """
        Create a rectangle entity.

        Args:
            body_id (str): Unique body id.
            x_scale (float): Scale of x coordinate.
            y_scale (float): Scale of y coordinate.
            half_width (float): Half width of rectangle.
            half_height (float): Half height of rectangle.
            angle (float): Angle of rectangle.
        """
        entity: dict[str, Any] = {
            "type": EntityType.Rectangle,
            "bodyID": body_id,
            "x": x_scale,
            "y": y_scale,
            "angle": angle,
            "halfWidth": half_width,
            "halfHeight": half_height,
        }
        return entity


class BodyInformation:
    # {
    # public:
    #     BodyInformation();
    #     //Keep an ID present for each body chunk
    #     std::string GenomeID;
    #
    #     //Required Nodes inside the structure
    #     std::vector<b2Vec2> nodes;
    #
    #     //All connections required for the sodaracer
    #     std::vector<b2Vec2> connections;
    #
    #     //this is for mapping between an index and a
    #     //connection number (required for pruning)
    #     std::map<int,int> indexToConnection;
    #
    #     //Is LEO used in the generation of this body
    #     //(important for output tracking)
    #     bool useLEO;
    #
    # private:
    #     ~BodyInformation();
    # };
    def __init__(self):
        self.genome_id: str = ""
        self.nodes = list()
        self.connections = list()
        self.index_to_connection = dict()
        self.useLEO = False


class PhysicsId:
    def __init__(self, pid: str):
        self.__id = pid

    def id(self):
        return self.__id


class DistanceAccessor(b2.b2DistanceJoint):
    # It was either this or access the name mangled
    # __getLength from the b2DistanceJoint so...
    getLength = b2._swig_new_instance_method(b2._Box2D.b2DistanceJoint___GetLength)
    setLength = b2._swig_new_instance_method(b2._Box2D.b2DistanceJoint___SetLength)


class Muscle:
    def __init__(
        self, muscle_id: str, joint: b2.b2Joint, phase: float, amplitude: float
    ):
        # store the id, joint, phase, and amplitude -- pretty simple!
        self.muscle_id = muscle_id
        self.joint = joint
        self.phase = phase
        self.amplitude = amplitude

        # pull the original length from our distance joint
        self.o_length = DistanceAccessor.getLength(
            joint
        )  # ((b2DistanceJoint*)joint)->GetLength();

    # getters to retrieve inner variables
    def id(self) -> str:
        return self.muscle_id

    def get_joint(self) -> b2.b2Joint:
        return self.joint

    def get_original_length(self) -> float:
        return self.o_length

    def get_phase(self) -> float:
        return self.phase

    def get_amplitude(self) -> float:
        return self.amplitude


class Bone:
    def __init__(self, muscle_id: str, joint: b2.b2Joint):
        self.bone_id = muscle_id
        self.joint = joint

    def get_joint(self) -> b2.b2Joint:
        return self.joint

    def id(self) -> str:
        return self.bone_id
