# -*- coding: utf-8 -*-
#
# Python version Copyright (c) 2015 John Stowers
#
# This software is provided 'as-is', without any express or implied
# warranty.  In no event will the authors be held liable for any damages
# arising from the use of this software.
# Permission is granted to anyone to use this software for any purpose,
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions:
# 1. The origin of this software must not be misrepresented; you must not
# claim that you wrote the original software. If you use this software
# in a product, an acknowledgment in the product documentation would be
# appreciated but is not required.
# 2. Altered source versions must be plainly marked as such, and must not be
# misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.

import cv2
import numpy as np

from Box2D import (b2Color, b2DistanceJoint, b2MouseJoint, b2PulleyJoint)
from Box2D.b2 import (staticBody, dynamicBody, kinematicBody, polygonShape,
                      circleShape, loopShape, edgeShape)


def cvcolor(color):
    return int(255.0 * color[2]), int(255.0 * color[1]), int(255.0 * color[0])


def cvcoord(pos):
    return tuple(map(int, pos))


class OpencvDrawFuncs(object):

    def __init__(self, w, h, ppm, fill_polygon=True, flip_y=True):
        self._w = w
        self._h = h
        self._ppm = ppm
        self._colors = {
            staticBody: (255, 255, 255),
            dynamicBody: (127, 127, 127),
            kinematicBody: (127, 127, 230),
        }
        self._fill_polygon = fill_polygon
        self._flip_y = flip_y
        self.screen = np.zeros((self._h, self._w, 3), np.uint8)

    def install(self):
        polygonShape.draw = self._draw_polygon
        circleShape.draw = self._draw_circle
        loopShape.draw = self._draw_loop
        edgeShape.draw = self._draw_edge

    def draw_world(self, world):
        for body in world.bodies:
            for fixture in body.fixtures:
                fixture.shape.draw(body, fixture)
        for joint in world.joints:
            self._draw_joint(joint)

    def clear_screen(self, screen=None):
        if screen is None:
            self.screen.fill(0)
        else:
            self.screen = screen

    def _fix_vertices(self, vertices):
        if self._flip_y:
            return [(v[0], self._h - v[1]) for v in vertices]
        else:
            return [(v[0], v[1]) for v in vertices]

    def _draw_joint(self, joint):
        bodyA, bodyB = joint.bodyA, joint.bodyB
        xf1, xf2 = bodyA.transform, bodyB.transform
        x1, x2 = xf1.position, xf2.position
        p1, p2 = joint.anchorA, joint.anchorB
        color = b2Color(0.5, 0.8, 0.8)

        x1, x2, p1, p2 = self._fix_vertices((x1 * self._ppm, x2 * self._ppm,
                                             p1 * self._ppm, p2 * self._ppm))

        if isinstance(joint, b2DistanceJoint):
            cv2.line(self.screen, cvcoord(p1), cvcoord(p2), cvcolor(color), 1)
        elif isinstance(joint, b2PulleyJoint):
            s1, s2 = joint.groundAnchorA, joint.groundAnchorB
            s1, s2 = self._fix_vertices((s1 * self._ppm, s2 * self._ppm))
            cv2.line(self.screen, cvcoord(s1), cvcoord(p1), cvcolor(color), 1)
            cv2.line(self.screen, cvcoord(s2), cvcoord(p2), cvcolor(color), 1)
            cv2.line(self.screen, cvcoord(s1), cvcoord(s2), cvcolor(color), 1)
        elif isinstance(joint, b2MouseJoint):
            pass  # don't draw it here
        else:
            cv2.line(self.screen, cvcoord(x1), cvcoord(p1), cvcolor(color), 1)
            cv2.line(self.screen, cvcoord(p1), cvcoord(p2), cvcolor(color), 1)
            cv2.line(self.screen, cvcoord(x2), cvcoord(p2), cvcolor(color), 1)

    def _draw_polygon(self, body, fixture):
        polygon = fixture.shape

        transform = body.transform
        vertices = self._fix_vertices([transform * v * self._ppm
                                       for v in polygon.vertices])

        pts = np.array(vertices, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(self.screen, [pts], True, self._colors[body.type])

        if self._fill_polygon:
            lightc = np.array(self._colors[body.type], dtype=int) * 0.5
            cv2.fillPoly(self.screen, [pts], lightc)

    def _draw_circle(self, body, fixture):
        circle = fixture.shape
        position = self._fix_vertices(
            [body.transform * circle.pos * self._ppm])[0]
        cv2.circle(self.screen, cvcoord(position), int(
            circle.radius * self._ppm), self._colors[body.type], 1)

    def _draw_edge(self, body, fixture):
        edge = fixture.shape
        v = [body.transform * edge.vertex1 * self._ppm,
             body.transform * edge.vertex2 * self._ppm]
        vertices = self._fix_vertices(v)
        cv2.line(self.screen, cvcoord(vertices[0]),
                 cvcoord(vertices[1]), self._colors[body.type], 1)

    def _draw_loop(self, body, fixture):
        loop = fixture.shape
        transform = body.transform
        vertices = self._fix_vertices([transform * v * self._ppm
                                       for v in loop.vertices])
        v1 = vertices[-1]
        for v2 in vertices:
            cv2.line(self.screen, cvcoord(v1), cvcoord(v2),
                     self._colors[body.type], 1)
            v1 = v2
