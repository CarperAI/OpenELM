#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C++ version Copyright (c) 2006-2007 Erin Catto http://www.box2d.org
# Python version by Ken Lauer / sirkne at gmail dot com
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

from .framework import (Framework, main)
from Box2D import (b2CircleShape, b2EdgeShape, b2FixtureDef, b2PolygonShape)


def create_bridge(world, ground, size, offset, plank_count, friction=0.6, density=1.0):
    """
    Create a bridge with plank_count planks,
    utilizing rectangular planks of size (width, height).
    The bridge should start at x_offset, and continue to
    roughly x_offset+width*plank_count.
    The y will not change.
    """
    width, height = size
    x_offset, y_offset = offset
    half_height = height / 2
    plank = b2FixtureDef(
        shape=b2PolygonShape(box=(width / 2, height / 2)),
        friction=friction,
        density=density,
    )

    bodies = []
    prevBody = ground
    for i in range(plank_count):
        body = world.CreateDynamicBody(
            position=(x_offset + width * i, y_offset),
            fixtures=plank,
        )
        bodies.append(body)

        world.CreateRevoluteJoint(
            bodyA=prevBody,
            bodyB=body,
            anchor=(x_offset + width * (i - 0.5), y_offset)
        )

        prevBody = body

    world.CreateRevoluteJoint(
        bodyA=prevBody,
        bodyB=ground,
        anchor=(x_offset + width * (plank_count - 0.5), y_offset),
    )
    return bodies


class Bridge (Framework):
    name = "Bridge"
    numPlanks = 30  # Number of planks in the bridge

    def __init__(self):
        super(Bridge, self).__init__()

        # The ground
        ground = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-40, 0), (40, 0)])
        )

        create_bridge(self.world, ground, (1.0, 0.25),
                      (-14.5, 5), self.numPlanks, 0.2, 20)

        fixture = b2FixtureDef(
            shape=b2PolygonShape(vertices=[(-0.5, 0.0),
                                           (0.5, 0.0),
                                           (0.0, 1.5),
                                           ]),
            density=1.0
        )
        for i in range(2):
            self.world.CreateDynamicBody(
                position=(-8 + 8 * i, 12),
                fixtures=fixture,
            )

        fixture = b2FixtureDef(shape=b2CircleShape(radius=0.5), density=1)
        for i in range(3):
            self.world.CreateDynamicBody(
                position=(-6 + 6 * i, 10),
                fixtures=fixture,
            )

if __name__ == "__main__":
    main(Bridge)
