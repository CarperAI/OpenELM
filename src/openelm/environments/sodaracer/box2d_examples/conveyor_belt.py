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
from Box2D import (b2EdgeShape, b2FixtureDef, b2PolygonShape)


class ConveyorBelt (Framework):
    name = "ConveyorBelt"

    def __init__(self):
        Framework.__init__(self)

        self.using_contacts = True
        ground = self.world.CreateStaticBody(
            shapes=[b2EdgeShape(vertices=[(-20, 0), (20, 0)])],
        )

        # Platform
        self.platform = self.world.CreateStaticBody(
            position=(-5, 5),
            allowSleep=False,
            fixtures=b2FixtureDef(friction=0.8,
                                  shape=b2PolygonShape(box=(10.0, 5.0)),),
        )

        self.platform_fixture = self.platform.fixtures[0]

        # Boxes
        for i in range(5):
            self.platform = self.world.CreateDynamicBody(
                position=(-10.0 + 2.0 * i, 7.0),
                fixtures=b2FixtureDef(density=20.0,
                                      shape=b2PolygonShape(box=(0.5, 0.5)),),
            )

    def PreSolve(self, contact, old_manifold):
        Framework.PreSolve(self, contact, old_manifold)

        fixture_a, fixture_b = contact.fixtureA, contact.fixtureB

        if fixture_a == self.platform_fixture:
            contact.tangentSpeed = 5.0
        elif fixture_b == self.platform_fixture:
            contact.tangentSpeed = -5.0

if __name__ == "__main__":
    main(ConveyorBelt)
