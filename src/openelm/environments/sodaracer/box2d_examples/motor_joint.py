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

from math import sin

from .framework import (Framework, Keys, main)
from Box2D import (b2Color, b2EdgeShape, b2FixtureDef, b2PolygonShape)


class MotorJoint (Framework):
    name = "MotorJoint"
    description = 'g to stop/go'
    count = 800

    def __init__(self):
        Framework.__init__(self)

        ground = self.world.CreateStaticBody(
            shapes=[b2EdgeShape(vertices=[(-20, 0), (20, 0)])],
        )

        # Define motorized body
        body = self.world.CreateDynamicBody(
            position=(0, 8),
            allowSleep=False,
            fixtures=b2FixtureDef(density=2.0, friction=0.6,
                                  shape=b2PolygonShape(box=(2.0, 0.5)),),
        )

        self.joint = self.world.CreateMotorJoint(bodyA=ground, bodyB=body,
                                                 maxForce=1000, maxTorque=1000)

        self.go = False
        self.time = 0.0

    def Keyboard(self, key):
        if key == Keys.K_g:
            self.go = not self.go

    def Step(self, settings):
        Framework.Step(self, settings)

        if self.go and settings.hz > 0.0:
            self.time += 1.0 / settings.hz

        linear_offset = (6 * sin(2.0 * self.time), 8.0 +
                         4.0 * sin(1.0 * self.time))
        angular_offset = 4.0 * self.time

        self.joint.linearOffset = linear_offset
        self.joint.angularOffset = angular_offset

        renderer = self.renderer
        renderer.DrawPoint(renderer.to_screen(
            linear_offset), 4, b2Color(0.9, 0.9, 0.9))

if __name__ == "__main__":
    main(MotorJoint)
