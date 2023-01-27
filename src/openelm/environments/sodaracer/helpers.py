from dataclasses import dataclass
from enum import Enum
from typing import Any

import Box2D.Box2D as b2


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
            "polyPoints": [{"x": point.x, "y": point.y} for point in points],
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


class DistanceAccessor(b2.b2DistanceJoint):
    """Helper class to get/set the length of a distance joint."""

    # It was either this or access the name mangled
    # __getLength from the b2DistanceJoint so...
    getLength = b2._swig_new_instance_method(b2._Box2D.b2DistanceJoint___GetLength)
    setLength = b2._swig_new_instance_method(b2._Box2D.b2DistanceJoint___SetLength)


@dataclass
class Muscle:
    id: str
    joint: b2.b2Joint
    phase: float
    amplitude: float

    def __post_init__(self):
        self.original_length: float = DistanceAccessor.getLength(self.joint)


@dataclass
class Bone:
    id: str
    joint: b2.b2Joint
