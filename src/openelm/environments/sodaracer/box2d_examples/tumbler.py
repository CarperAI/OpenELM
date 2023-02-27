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
from Box2D import (b2FixtureDef, b2PolygonShape, b2_pi)


class Tumbler (Framework):
    name = "Tumbler"
    description = ''
    count = 800

    def __init__(self):
        Framework.__init__(self)

        ground = self.world.CreateBody()

        body = self.world.CreateDynamicBody(
            position=(0, 10),
            allowSleep=False,
            shapeFixture=b2FixtureDef(density=5.0),
            shapes=[
                b2PolygonShape(box=(0.5, 10, (10, 0), 0)),
                b2PolygonShape(box=(0.5, 10, (-10, 0), 0)),
                b2PolygonShape(box=(10, 0.5, (0, 10), 0)),
                b2PolygonShape(box=(10, 0.5, (0, -10), 0)),
            ]
        )

        self.joint = self.world.CreateRevoluteJoint(bodyA=ground, bodyB=body,
                                                    localAnchorA=(0, 10), localAnchorB=(0, 0),
                                                    referenceAngle=0, motorSpeed=0.05 * b2_pi,
                                                    enableMotor=True, maxMotorTorque=1.0e8)

    def Step(self, settings):
        Framework.Step(self, settings)

        self.count -= 1
        if self.count == 0:
            return

        self.world.CreateDynamicBody(
            position=(0, 10),
            allowSleep=False,
            fixtures=b2FixtureDef(
                density=1.0, shape=b2PolygonShape(box=(0.125, 0.125))),
        )

if __name__ == "__main__":
    main(Tumbler)
