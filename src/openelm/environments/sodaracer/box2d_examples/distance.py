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
from Box2D import (b2Color, b2Distance, b2PolygonShape, b2Transform, b2Vec2,
                   b2_pi)


class Distance (Framework):
    name = "Distance"
    description = ("Use WASD to move and QE to rotate the small rectangle.\n"
                   "The distance between the marked points is shown.")
    point_a_color = b2Color(1, 0, 0)
    point_b_color = b2Color(1, 1, 0)
    poly_color = b2Color(0.9, 0.9, 0.9)

    def __init__(self):
        super(Distance, self).__init__()
        # Transform A -- a simple translation/offset of (0,-0.2)
        self.transformA = b2Transform()
        self.transformA.SetIdentity()
        self.transformA.position = (0, -0.2)

        # Transform B -- a translation and a rotation
        self.transformB = b2Transform()
        self.positionB = b2Vec2(12.017401, 0.13678508)
        self.angleB = -0.0109265
        self.transformB.Set(self.positionB, self.angleB)

        # The two shapes, transformed by the respective transform[A,B]
        self.polygonA = b2PolygonShape(box=(10, 0.2))
        self.polygonB = b2PolygonShape(box=(2, 0.1))

    def Step(self, settings):
        super(Distance, self).Step(settings)

        # Calculate the distance between the two shapes with the specified
        # transforms
        dist_result = b2Distance(shapeA=self.polygonA, shapeB=self.polygonB,
                                 transformA=self.transformA,
                                 transformB=self.transformB)

        pointA, pointB, distance, iterations = dist_result

        self.Print('Distance = %g' % distance)
        self.Print('Iterations = %d' % iterations)

        # Manually transform the vertices and draw the shapes
        for shape, transform in [(self.polygonA, self.transformA), (self.polygonB, self.transformB)]:
            new_verts = [self.renderer.to_screen(
                transform * v) for v in shape.vertices]
            self.renderer.DrawPolygon(new_verts, self.poly_color)

        self.renderer.DrawPoint(self.renderer.to_screen(pointA), 4,
                                self.point_a_color)
        self.renderer.DrawPoint(self.renderer.to_screen(pointB), 4,
                                self.point_b_color)

    def Keyboard(self, key):
        if key == Keys.K_a:
            self.positionB -= (0.1, 0)
        elif key == Keys.K_d:
            self.positionB += (0.1, 0)
        elif key == Keys.K_w:
            self.positionB += (0, 0.1)
        elif key == Keys.K_s:
            self.positionB -= (0, 0.1)
        elif key == Keys.K_q:
            self.angleB += 0.1 * b2_pi
        elif key == Keys.K_e:
            self.angleB -= 0.1 * b2_pi
        self.transformB.Set(self.positionB, self.angleB)

if __name__ == "__main__":
    main(Distance)
