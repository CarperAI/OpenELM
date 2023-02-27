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
from Box2D import (b2EdgeShape, b2PolygonShape)


class EdgeTest (Framework):
    name = "EdgeTest"
    description = "Utilizes b2EdgeShape"

    def __init__(self):
        super(EdgeTest, self).__init__()

        v1 = (-10.0, 0.0)
        v2 = (-7.0, -1.0)
        v3 = (-4.0, 0.0)
        v4 = (0.0, 0.0)
        v5 = (4.0, 0.0)
        v6 = (7.0, 1.0)
        v7 = (10.0, 0.0)

        shapes = [b2EdgeShape(vertices=[None, v1, v2, v3]),
                  b2EdgeShape(vertices=[v1, v2, v3, v4]),
                  b2EdgeShape(vertices=[v2, v3, v4, v5]),
                  b2EdgeShape(vertices=[v3, v4, v5, v6]),
                  b2EdgeShape(vertices=[v4, v5, v6, v7]),
                  b2EdgeShape(vertices=[v5, v6, v7])
                  ]
        ground = self.world.CreateStaticBody(shapes=shapes)

        box = self.world.CreateDynamicBody(
            position=(0.5, 0.6),
            allowSleep=False,
            shapes=b2PolygonShape(box=(0.5, 0.5))
        )

if __name__ == "__main__":
    main(EdgeTest)
