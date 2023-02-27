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
from Box2D import (b2CircleShape, b2EdgeShape, b2FixtureDef)


class Restitution (Framework):
    name = "Restitution example"
    description = "Note the difference in bounce height of the circles"

    def __init__(self):
        super(Restitution, self).__init__()

        # The ground
        ground = self.world.CreateStaticBody(
            shapes=b2EdgeShape(vertices=[(-20, 0), (20, 0)])
        )

        radius = 1.0
        density = 1.0
        # The bodies
        for i, restitution in enumerate([0.0, 0.1, 0.3, 0.5, 0.75, 0.9, 1.0]):
            self.world.CreateDynamicBody(
                position=(-10 + 3.0 * i, 20),
                fixtures=b2FixtureDef(
                    shape=b2CircleShape(radius=radius),
                    density=density, restitution=restitution)
            )

if __name__ == "__main__":
    main(Restitution)
