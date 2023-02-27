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
from Box2D import (b2DistanceJointDef, b2EdgeShape, b2FixtureDef,
                   b2PolygonShape)


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

        # Create the joints between each of the bodies and also the ground
        #         bodyA      bodyB   localAnchorA localAnchorB
        sets = [(ground,    bodies[0], (-10, 0), (-0.5, -0.5)),
                (ground,    bodies[1], (10, 0),  (0.5, -0.5)),
                (ground,    bodies[2], (10, 20), (0.5, 0.5)),
                (ground,    bodies[3], (-10, 20), (-0.5, 0.5)),
                (bodies[0], bodies[1], (0.5, 0), (-0.5, 0)),
                (bodies[1], bodies[2], (0, 0.5), (0, -0.5)),
                (bodies[2], bodies[3], (-0.5, 0), (0.5, 0)),
                (bodies[3], bodies[0], (0, -0.5), (0, 0.5)),
                ]

        # We will define the positions in the local body coordinates, the length
        # will automatically be set by the __init__ of the b2DistanceJointDef
        self.joints = []
        for bodyA, bodyB, localAnchorA, localAnchorB in sets:
            dfn = b2DistanceJointDef(
                frequencyHz=4.0,
                dampingRatio=0.5,
                bodyA=bodyA,
                bodyB=bodyB,
                localAnchorA=localAnchorA,
                localAnchorB=localAnchorB,
            )
            self.joints.append(self.world.CreateJoint(dfn))

    def Keyboard(self, key):
        if key == Keys.K_b:
            for body in self.bodies:
                # Gets both FixtureDestroyed and JointDestroyed callbacks.
                self.world.DestroyBody(body)
                break

        elif key == Keys.K_j:
            for joint in self.joints:
                # Does not get a JointDestroyed callback!
                self.world.DestroyJoint(joint)
                self.joints.remove(joint)
                break

    def FixtureDestroyed(self, fixture):
        super(Web, self).FixtureDestroyed(fixture)
        body = fixture.body
        if body in self.bodies:
            print(body)
            self.bodies.remove(body)
            print("Fixture destroyed, removing its body from the list. Bodies left: %d"
                  % len(self.bodies))

    def JointDestroyed(self, joint):
        if joint in self.joints:
            self.joints.remove(joint)
            print("Joint destroyed and removed from the list. Joints left: %d"
                  % len(self.joints))

if __name__ == "__main__":
    main(Web)
