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

from .framework import (Framework, Keys, main)
from Box2D import (b2EdgeShape, b2FixtureDef, b2PolygonShape, b2_dynamicBody,
                   b2_kinematicBody, b2_staticBody)


class BodyTypes(Framework):
    name = "Body Types"
    description = "Change body type keys: (d) dynamic, (s) static, (k) kinematic"
    speed = 3  # platform speed

    def __init__(self):
        super(BodyTypes, self).__init__()

        # The ground
        ground = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-20, 0), (20, 0)])
        )

        # The attachment
        self.attachment = self.world.CreateDynamicBody(
            position=(0, 3),
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(box=(0.5, 2)), density=2.0),
        )

        # The platform
        fixture = b2FixtureDef(
            shape=b2PolygonShape(box=(4, 0.5)),
            density=2,
            friction=0.6,
        )

        self.platform = self.world.CreateDynamicBody(
            position=(0, 5),
            fixtures=fixture,
        )

        # The joints joining the attachment/platform and ground/platform
        self.world.CreateRevoluteJoint(
            bodyA=self.attachment,
            bodyB=self.platform,
            anchor=(0, 5),
            maxMotorTorque=50,
            enableMotor=True
        )

        self.world.CreatePrismaticJoint(
            bodyA=ground,
            bodyB=self.platform,
            anchor=(0, 5),
            axis=(1, 0),
            maxMotorForce=1000,
            enableMotor=True,
            lowerTranslation=-10,
            upperTranslation=10,
            enableLimit=True
        )

        # And the payload that initially sits upon the platform
        # Reusing the fixture we previously defined above
        fixture.shape.box = (0.75, 0.75)
        self.payload = self.world.CreateDynamicBody(
            position=(0, 8),
            fixtures=fixture,
        )

    def Keyboard(self, key):
        if key == Keys.K_d:
            self.platform.type = b2_dynamicBody
        elif key == Keys.K_s:
            self.platform.type = b2_staticBody
        elif key == Keys.K_k:
            self.platform.type = b2_kinematicBody
            self.platform.linearVelocity = (-self.speed, 0)
            self.platform.angularVelocity = 0

    def Step(self, settings):
        super(BodyTypes, self).Step(settings)

        if self.platform.type == b2_kinematicBody:
            p = self.platform.transform.position
            v = self.platform.linearVelocity
            if ((p.x < -10 and v.x < 0) or (p.x > 10 and v.x > 0)):
                v.x = -v.x
                self.platform.linearVelocity = v

if __name__ == "__main__":
    main(BodyTypes)
