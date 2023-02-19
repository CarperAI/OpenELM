# import simulator
# import json

# myWorld = simulator.IESoRWorld()

# world_json = myWorld.get_world_json()
# print(world_json)


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

from Box2D.examples.framework import (Framework, Keys, main)
from Box2D.examples.bridge import create_bridge
from math import sqrt

from Box2D import Box2D as b2
from Box2D import (b2CircleShape, b2EdgeShape, b2FixtureDef, b2PolygonShape,
                   b2_pi)

        # self.world = b2.World(gravity=(0, -10))
        # self.ground_body = self.world.create_static_body(position=(0, 10))
        # self.ground_body.create_polygon_fixture(box=(50, 10))



class Web(Framework):
    name = "Web"
    description = "This demonstrates a soft distance joint. Press: (b) to delete a body, (j) to delete a joint"
    bodies = []
    joints = []

    def __init__(self):
        super(Web, self).__init__()

        # The ground
        ground = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-40, 0), (40, 0)])
        )

        fixture = b2FixtureDef(shape=b2PolygonShape(box=(0.5, 0.5)),
                               density=5, friction=0.2)

        self.bodies = [self.world.CreateDynamicBody(
            position=pos,
            fixtures=fixture
        ) for pos in ((-5, 5), (5, 5), (5, 15), (-5, 15))]

        bodies = self.bodies



if __name__ == "__main__":
    main(Web)