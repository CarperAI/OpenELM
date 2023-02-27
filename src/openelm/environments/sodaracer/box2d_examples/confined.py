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

from random import random

from .framework import (Framework, Keys, main)
from Box2D import (b2CircleShape, b2EdgeShape, b2FixtureDef)


class Confined (Framework):
    name = "Confined space"
    description = "Press c to create a circle"

    def __init__(self):
        super(Confined, self).__init__()

        # The ground
        ground = self.world.CreateStaticBody(
            shapes=[b2EdgeShape(vertices=[(-10, 0), (10, 0)]),
                    b2EdgeShape(vertices=[(-10, 0), (-10, 20)]),
                    b2EdgeShape(vertices=[(10, 0), (10, 20)]),
                    b2EdgeShape(vertices=[(-10, 20), (10, 20)]),
                    ])

        # The bodies
        self.radius = radius = 0.5
        columnCount = 5
        rowCount = 5

        for j in range(columnCount):
            for i in range(rowCount):
                self.CreateCircle((-10 + (2.1 * j + 1 + 0.01 * i) * radius,
                                  (2 * i + 1) * radius))

        self.world.gravity = (0, 0)

    def CreateCircle(self, pos):
        fixture = b2FixtureDef(shape=b2CircleShape(radius=self.radius,
                                                   pos=(0, 0)),
                               density=1, friction=0.1)

        self.world.CreateDynamicBody(
            position=pos,
            fixtures=fixture
        )

    def Keyboard(self, key):
        if key == Keys.K_c:
            self.CreateCircle((2.0 * random() - 1.0,
                               self.radius * (1.0 + random())))

if __name__ == "__main__":
    main(Confined)
