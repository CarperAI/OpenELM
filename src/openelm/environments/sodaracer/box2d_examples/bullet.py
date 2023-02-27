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

from .framework import (Framework, main)
from Box2D import (b2EdgeShape, b2FixtureDef, b2PolygonShape, b2Random)


class Bullet (Framework):
    name = "Bullet"
    description = 'A test for very fast moving objects (bullets)'

    def __init__(self):
        super(Bullet, self).__init__()

        ground = self.world.CreateStaticBody(
            position=(0, 0),
            shapes=[b2EdgeShape(vertices=[(-10, 0), (10, 0)]),
                    b2PolygonShape(box=(0.2, 1, (0.5, 1), 0))]
        )

        self._x = 0.20352793
        self.body = self.world.CreateDynamicBody(
            position=(0, 4),
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(box=(2, 0.1)), density=1.0),
        )

        self.bullet = self.world.CreateDynamicBody(
            position=(self._x, 10),
            bullet=True,
            fixtures=b2FixtureDef(shape=b2PolygonShape(
                box=(0.25, 0.25)), density=100.0),
            linearVelocity=(0, -50)
        )

    def Launch(self):
        self.body.transform = [(0, 4), 0]
        self.body.linearVelocity = (0, 0)
        self.body.angularVelocity = 0

        self.x = b2Random()
        self.bullet.transform = [(self.x, 10), 0]
        self.bullet.linearVelocity = (0, -50)
        self.bullet.angularVelocity = 0

    def Step(self, settings):
        super(Bullet, self).Step(settings)

        if (self.stepCount % 60) == 0:
            self.Launch()

if __name__ == "__main__":
    main(Bullet)
