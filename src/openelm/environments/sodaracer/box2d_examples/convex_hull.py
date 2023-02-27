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
from Box2D import (b2Clamp, b2Color, b2PolygonShape, b2Random,
                   b2_maxPolygonVertices)


class ConvexHull (Framework):
    name = "ConvexHull"
    description = ('Press g to generate a new random convex hull, a to switch '
                   'to automatic mode')

    def __init__(self):
        Framework.__init__(self)

        self.auto = False

        self.generate()

    def generate(self):
        lower = (-8, -8)
        upper = (8, 8)

        self.verts = verts = []
        for i in range(b2_maxPolygonVertices):
            x = 10.0 * b2Random(0.0, 10.0)
            y = 10.0 * b2Random(0.0, 10.0)

            # Clamp onto a square to help create collinearities.
            # This will stress the convex hull algorithm.
            verts.append(b2Clamp((x, y), lower, upper))

    def Keyboard(self, key):
        if key == Keys.K_a:
            self.auto = not self.auto
        elif key == Keys.K_g:
            self.generate()

    def Step(self, settings):
        Framework.Step(self, settings)

        renderer = self.renderer

        try:
            poly = b2PolygonShape(vertices=self.verts)
        except AssertionError as ex:
            self.Print('b2PolygonShape failed: %s' % ex)
        else:
            self.Print('Valid: %s' % poly.valid)

        renderer.DrawPolygon([renderer.to_screen(v)
                              for v in self.verts], b2Color(0.9, 0.9, 0.9))
        for i, v in enumerate(self.verts):
            renderer.DrawPoint(renderer.to_screen(v), 2.0,
                               b2Color(0.9, 0.5, 0.5))

            x, y = renderer.to_screen(v)
            self.DrawStringAt(x + 0.05, y + 0.05, '%d' % i)

        if self.auto:
            self.generate()

if __name__ == "__main__":
    main(ConvexHull)
