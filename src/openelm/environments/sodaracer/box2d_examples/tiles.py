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

from math import ceil, log

from .framework import (Framework, main)
from Box2D import (b2FixtureDef, b2PolygonShape, b2Vec2)


class Tiles (Framework):
    name = "Tiles"
    description = ('This stress tests the dynamic tree broad-phase. This also'
                   'shows that tile based collision\nis _not_ smooth due to '
                   'Box2D not knowing about adjacency.')

    def __init__(self):
        super(Tiles, self).__init__()

        a = 0.5

        def ground_positions():
            N = 200
            M = 10
            position = b2Vec2(0, 0)
            for i in range(M):
                position.x = -N * a
                for j in range(N):
                    yield position
                    position.x += 2.0 * a
                position.y -= 2.0 * a

        ground = self.world.CreateStaticBody(
            position=(0, -a),
            shapes=[b2PolygonShape(box=(a, a, position, 0))
                    for position in ground_positions()]
        )

        count = 20

        def dynamic_positions():
            x = b2Vec2(-7.0, 0.75)
            deltaX = (0.5625, 1.25)
            deltaY = (1.125, 0.0)
            for i in range(count):
                y = x.copy()
                for j in range(i, count):
                    yield y
                    y += deltaY
                x += deltaX

        for pos in dynamic_positions():
            self.world.CreateDynamicBody(
                position=pos,
                fixtures=b2FixtureDef(
                    shape=b2PolygonShape(box=(a, a)), density=5)
            )

    def Step(self, settings):
        super(Tiles, self).Step(settings)
        cm = self.world.contactManager
        height = cm.broadPhase.treeHeight
        leafCount = cm.broadPhase.proxyCount
        minNodeCount = 2 * leafCount - 1
        minHeight = ceil(log(float(minNodeCount)) / log(2))
        self.Print('Dynamic tree height=%d, min=%d' % (height, minHeight))

if __name__ == "__main__":
    main(Tiles)
