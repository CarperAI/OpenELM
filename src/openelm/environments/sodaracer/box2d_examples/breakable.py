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
from Box2D import (b2Cross, b2EdgeShape, b2FixtureDef, b2PolygonShape, b2_pi)


class Breakable (Framework):
    name = "Breakable bodies"
    description = "With enough of an impulse, the single body will split [press b to manually break it]"
    _break = False  # Flag to break
    broke = False  # Already broken?

    def __init__(self):
        super(Breakable, self).__init__()

        # The ground
        self.world.CreateBody(shapes=b2EdgeShape(vertices=[(-40, 0), (40, 0)]))

        # The breakable body
        self.shapes = (b2PolygonShape(box=(0.5, 0.5, (-0.5, 0), 0)),
                       b2PolygonShape(box=(0.5, 0.5, (0.5, 0), 0))
                       )
        self.body = self.world.CreateDynamicBody(
            position=(0, 40),
            angle=0.25 * b2_pi,
            shapes=self.shapes,
            shapeFixture=b2FixtureDef(density=1),
        )

        self.fixtures = self.body.fixtures

    def PostSolve(self, contact, impulse):
        # Already broken?
        if self.broke:
            return

        # If the impulse is enough to split the objects, then flag it to break
        if max(impulse.normalImpulses) > 40:
            self._break = True

    def Break(self):
        # Create two bodies from one
        body = self.body
        center = body.worldCenter

        body.DestroyFixture(self.fixtures[1])
        self.fixture2 = None

        body2 = self.world.CreateDynamicBody(
            position=body.position,
            angle=body.angle,
            shapes=self.shapes[1],
            shapeFixture=b2FixtureDef(density=1),
        )
        # Compute consistent velocities for new bodies based on cached
        # velocity.
        velocity1 = (self.velocity +
                     b2Cross(self.angularVelocity, body.worldCenter - center))
        velocity2 = (self.velocity +
                     b2Cross(self.angularVelocity, body2.worldCenter - center))

        body.angularVelocity = self.angularVelocity
        body.linearVelocity = velocity1
        body2.angularVelocity = self.angularVelocity
        body2.linearVelocity = velocity2

    def Step(self, settings):
        super(Breakable, self).Step(settings)
        if self._break:
            self.Break()
            self.broke = True
            self._break = False
        if not self.broke:
            self.velocity = self.body.linearVelocity
            self.angularVelocity = self.body.angularVelocity

    def Keyboard(self, key):
        if key == Keys.K_b and not self.broke:
            self._break = True

if __name__ == "__main__":
    main(Breakable)
