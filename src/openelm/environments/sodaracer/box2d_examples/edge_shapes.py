#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C++ version Copyright (c) 2006-2007 Erin Catto http://www.box2d.org
# Python version Copyright (c) 2010 kne / sirkne at gmail dot com
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

from .framework import (Framework, Keys, main)
from Box2D import (b2BodyDef, b2CircleShape, b2Color, b2EdgeShape,
                   b2FixtureDef, b2PolygonShape, b2RayCastCallback, b2Vec2,
                   b2_dynamicBody, b2_pi)
from math import cos, sin, pi, sqrt
from random import random

VERTEX_COUNT = 80


def get_sinusoid_vertices(x1, vertices):
    y1 = 2.0 * cos(x1 / 10.0 * pi)
    for i in range(vertices):
        x2 = x1 + 0.5
        y2 = 2.0 * cos(x2 / 10.0 * pi)
        yield (x1, y1), (x2, y2)
        x1, y1 = x2, y2


def get_octagon_vertices(w):
    b = w / (2.0 + sqrt(2.0))
    s = sqrt(2.0) * b
    return [(0.5 * s, 0), (0.5 * w, b),
            (0.5 * w, b + s), (0.5 * s, w),
            (-0.5 * s, w), (-0.5 * w, b + s),
            (-0.5 * w, b), (-0.5 * s, 0.0), ]

# for more information, see raycast.py


class RayCastCallback(b2RayCastCallback):

    def __init__(self, **kwargs):
        super(RayCastCallback, self).__init__()
        self.fixture = None

    def ReportFixture(self, fixture, point, normal, fraction):
        self.fixture = fixture
        self.point = b2Vec2(point)
        self.normal = b2Vec2(normal)
        return fraction


class EdgeShapes (Framework):
    name = "Edge Shapes"
    description = "Press 1-5 to drop stuff, and d to delete"
    p1_color = b2Color(0.4, 0.9, 0.4)
    s1_color = b2Color(0.8, 0.8, 0.8)
    s2_color = b2Color(0.9, 0.9, 0.4)

    def __init__(self):
        super(EdgeShapes, self).__init__()
        self.ground = self.world.CreateStaticBody(
            shapes=[b2EdgeShape(vertices=v)
                    for v in get_sinusoid_vertices(-20.0, VERTEX_COUNT)])

        self.shapes = [
            b2PolygonShape(vertices=[(-0.5, 0), (0.5, 0), (0, 1.5)]),
            b2PolygonShape(vertices=[(-0.1, 0), (0.1, 0), (0, 1.5)]),
            b2PolygonShape(vertices=get_octagon_vertices(1.0)),
            b2PolygonShape(box=(0.5, 0.5)),
            b2CircleShape(radius=0.5),
        ]

        self.angle = 0
        self.callback = RayCastCallback()

    @property
    def bodies(self):
        return [body for body in self.world.bodies
                if body != self.ground]

    def CreateShape(self, shapeindex):
        try:
            shape = self.shapes[shapeindex]
        except IndexError:
            return

        pos = (10.0 * (2.0 * random() - 1.0), 10.0 * (2.0 * random() + 1.0))
        defn = b2BodyDef(
            type=b2_dynamicBody,
            fixtures=b2FixtureDef(shape=shape, friction=0.3),
            position=pos,
            angle=(b2_pi * (2.0 * random() - 1.0)),
        )

        if isinstance(shape, b2CircleShape):
            defn.angularDamping = 0.02

        self.world.CreateBody(defn)

    def DestroyBody(self):
        if not self.world.locked:
            for body in self.bodies:
                self.world.DestroyBody(body)
                break

    def Keyboard(self, key):
        if key in (Keys.K_1, Keys.K_2, Keys.K_3, Keys.K_4, Keys.K_5):
            self.CreateShape(key - Keys.K_1)
        elif key == Keys.K_d:
            self.DestroyBody()

    def Step(self, settings):
        super(EdgeShapes, self).Step(settings)

        # Set up the raycast line
        length = 25.0
        point1 = b2Vec2(0, 10)
        d = (length * cos(self.angle), length * sin(self.angle))
        point2 = point1 + d

        callback = self.callback
        callback.fixture = None

        self.world.RayCast(callback, point1, point2)

        # The callback has been called by this point, and if a fixture was hit it will have been
        # set to callback.fixture.
        point1 = self.renderer.to_screen(point1)
        point2 = self.renderer.to_screen(point2)
        if callback.fixture:
            cb_point = self.renderer.to_screen(callback.point)
            self.renderer.DrawPoint(cb_point, 5.0, self.p1_color)
            self.renderer.DrawSegment(point1, cb_point, self.s1_color)

            head = b2Vec2(cb_point) + 0.5 * callback.normal
            self.renderer.DrawSegment(cb_point, head, self.s2_color)
        else:
            self.renderer.DrawSegment(point1, point2, self.s1_color)

        if not settings.pause or settings.singleStep:
            self.angle += 0.25 * b2_pi / 180

if __name__ == "__main__":
    main(EdgeShapes)
