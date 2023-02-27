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
from math import sqrt

from Box2D import (b2FixtureDef, b2PolygonShape,
                   b2Transform, b2Mul,
                   b2_pi)


class ApplyForce (Framework):
    name = "ApplyForce"
    description = "Use w, a, and d to control the ship."

    def __init__(self):
        super(ApplyForce, self).__init__()
        self.world.gravity = (0.0, 0.0)

        # The boundaries
        ground = self.world.CreateBody(position=(0, 20))
        ground.CreateEdgeChain(
            [(-20, -20),
             (-20, 20),
             (20, 20),
             (20, -20),
             (-20, -20)]
        )

        #  TODO: make note of transform.R.set() -> transform.angle =
        xf1 = b2Transform()
        xf1.angle = 0.3524 * b2_pi
        xf1.position = xf1.R * (1.0, 0.0)

        xf2 = b2Transform()
        xf2.angle = -0.3524 * b2_pi
        xf2.position = xf2.R * (-1.0, 0.0)
        self.body = self.world.CreateDynamicBody(
            position=(0, 2),
            angle=b2_pi,
            angularDamping=5,
            linearDamping=0.1,
            shapes=[b2PolygonShape(vertices=[xf1 * (-1, 0), xf1 * (1, 0),
                                             xf1 * (0, .5)]),
                    b2PolygonShape(vertices=[xf2 * (-1, 0), xf2 * (1, 0),
                                             xf2 * (0, .5)])],
            shapeFixture=b2FixtureDef(density=2.0),
        )

        gravity = 10.0
        fixtures = b2FixtureDef(shape=b2PolygonShape(box=(0.5, 0.5)),
                                density=1, friction=0.3)
        for i in range(10):
            body = self.world.CreateDynamicBody(
                position=(0, 5 + 1.54 * i), fixtures=fixtures)

            # For a circle: I = 0.5 * m * r * r ==> r = sqrt(2 * I / m)
            r = sqrt(2.0 * body.inertia / body.mass)

            self.world.CreateFrictionJoint(
                bodyA=ground,
                bodyB=body,
                localAnchorA=(0, 0),
                localAnchorB=(0, 0),
                collideConnected=True,
                maxForce=body.mass * gravity,
                maxTorque=body.mass * r * gravity
            )

    def Keyboard(self, key):
        if not self.body:
            return

        if key == Keys.K_w:
            f = self.body.GetWorldVector(localVector=(0.0, -200.0))
            p = self.body.GetWorldPoint(localPoint=(0.0, 2.0))
            self.body.ApplyForce(f, p, True)
        elif key == Keys.K_a:
            self.body.ApplyTorque(50.0, True)
        elif key == Keys.K_d:
            self.body.ApplyTorque(-50.0, True)

if __name__ == "__main__":
    main(ApplyForce)
