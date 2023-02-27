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
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape)


class Pulley (Framework):
    name = "Pulley"

    def __init__(self):
        super(Pulley, self).__init__()
        y, L, a, b = 16.0, 12.0, 1.0, 2.0
        # The ground
        ground = self.world.CreateStaticBody(
            shapes=[edgeShape(vertices=[(-40, 0), (40, 0)]),
                    circleShape(radius=2, pos=(-10.0, y + b + L)),
                    circleShape(radius=2, pos=(10.0, y + b + L))]
        )

        bodyA = self.world.CreateDynamicBody(
            position=(-10, y),
            fixtures=fixtureDef(shape=polygonShape(box=(a, b)), density=5.0),
        )
        bodyB = self.world.CreateDynamicBody(
            position=(10, y),
            fixtures=fixtureDef(shape=polygonShape(box=(a, b)), density=5.0),
        )

        self.pulley = self.world.CreatePulleyJoint(
            bodyA=bodyA,
            bodyB=bodyB,
            anchorA=(-10.0, y + b),
            anchorB=(10.0, y + b),
            groundAnchorA=(-10.0, y + b + L),
            groundAnchorB=(10.0, y + b + L),
            ratio=1.5,
        )

    def Step(self, settings):
        super(Pulley, self).Step(settings)

        ratio = self.pulley.ratio
        L = self.pulley.length1 + self.pulley.length2 * ratio
        self.Print('L1 + %4.2f * L2 = %4.2f' % (ratio, L))

if __name__ == "__main__":
    main(Pulley)
