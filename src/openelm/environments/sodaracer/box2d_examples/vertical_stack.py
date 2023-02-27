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
from Box2D import (b2CircleShape, b2EdgeShape, b2FixtureDef, b2PolygonShape)


class VerticalStack (Framework):
    name = "Vertical Stack"
    description = ("Tests the stability of stacking circles and boxes\n"
                   "Press B to launch a horizontal bullet")
    bullet = None

    def __init__(self):
        super(VerticalStack, self).__init__()

        columns = 5
        rows = 16

        ground = self.world.CreateStaticBody(
            shapes=[
                b2EdgeShape(vertices=[(-40, 0), (40, 0)]),
                b2EdgeShape(vertices=[(20, 0), (20, 20)]),
            ]
        )

        box = b2FixtureDef(
            shape=b2PolygonShape(box=(0.5, 0.5)),
            density=1,
            friction=0.3)
        circle = b2FixtureDef(
            shape=b2CircleShape(radius=0.5),
            density=1,
            friction=0.3)

        box_start = -10
        box_space = 2.5
        circle_start = 8
        circle_space = 2.5
        for j in range(columns):
            for i in range(rows):
                self.world.CreateDynamicBody(
                    fixtures=box,
                    position=(box_start + box_space * j, 0.752 + 1.54 * i)
                )
                self.world.CreateDynamicBody(
                    fixtures=circle,
                    position=(circle_start + circle_space *
                              j, 0.752 + 1.54 * i)
                )

    def Step(self, settings):
        super(VerticalStack, self).Step(settings)

    def Keyboard(self, key):
        if key == Keys.K_b:
            if self.bullet:
                self.world.DestroyBody(self.bullet)
                self.bullet = None
            circle = b2FixtureDef(
                shape=b2CircleShape(radius=0.25),
                density=20,
                restitution=0.05)
            self.bullet = self.world.CreateDynamicBody(
                position=(-31, 5),
                bullet=True,
                fixtures=circle,
                linearVelocity=(400, 0),
            )

if __name__ == "__main__":
    main(VerticalStack)
