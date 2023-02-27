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
from Box2D import (b2EdgeShape, b2FixtureDef, b2PolygonShape, b2RopeJointDef)

# From the original C++ testbed example:
# "This test shows how a rope joint can be used to stabilize a chain of bodies
#  with a heavy payload. Notice that the rope joint just prevents excessive
#  stretching and has no other effect.  By disabling the rope joint you can see
#  that the Box2D solver has trouble supporting heavy bodies with light bodies.
#  Try playing around with the densities, time step, and iterations to see how
#  they affect stability.  This test also shows how to use contact filtering.
#  Filtering is configured so that the payload does not collide with the
#  chain."


class Rope (Framework):
    name = "Rope Joint Test"
    description = "Press j to toggle the rope joint."

    def __init__(self):
        super(Rope, self).__init__()

        # The ground
        ground = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-40, 0), (40, 0)]))

        shape = b2PolygonShape(box=(0.5, 0.125))
        fd = b2FixtureDef(
            shape=shape,
            friction=0.2,
            density=20,
            categoryBits=0x0001,
            maskBits=(0xFFFF & ~0x0002),
        )

        N = 10
        y = 15.0

        prevBody = ground
        for i in range(N):
            if i < N - 1:
                body = self.world.CreateDynamicBody(
                    position=(0.5 + i, y),
                    fixtures=fd,
                )
            else:
                shape.box = (1.5, 1.5)
                fd.density = 100
                fd.categoryBits = 0x0002
                body = self.world.CreateDynamicBody(
                    position=(i, y),
                    fixtures=fd,
                    angularDamping=0.4,
                )

            self.world.CreateRevoluteJoint(
                bodyA=prevBody,
                bodyB=body,
                anchor=(i, y),
                collideConnected=False,
            )

            prevBody = body

        extraLength = 0.01
        self.rd = rd = b2RopeJointDef(
            bodyA=ground,
            bodyB=body,
            maxLength=N - 1.0 + extraLength,
            localAnchorA=(0, y),
            localAnchorB=(0, 0)
        )
        self.rope = self.world.CreateJoint(rd)

    def Step(self, settings):
        super(Rope, self).Step(settings)

        if self.rope:
            self.Print('Rope ON')
        else:
            self.Print('Rope OFF')

    def Keyboard(self, key):
        if key == Keys.K_j:
            if self.rope:
                self.world.DestroyJoint(self.rope)
                self.rope = None
            else:
                self.rope = self.world.CreateJoint(self.rd)

if __name__ == "__main__":
    main(Rope)
