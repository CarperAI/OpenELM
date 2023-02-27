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
from Box2D import (b2Color, b2Globals, b2PolygonShape, b2Sweep, b2TimeOfImpact)


class TimeOfImpact (Framework):
    name = "Time of Impact"
    description = ("See the source code for more information."
                   "No additional controls.")

    def __init__(self):
        super(TimeOfImpact, self).__init__()

        # The two shapes to check for time of impact
        self.shapeA = b2PolygonShape(box=(0.2, 1, (0.5, 1), 0))
        self.shapeB = b2PolygonShape(box=(2, 0.1))

    def Step(self, settings):
        super(TimeOfImpact, self).Step(settings)

        # b2Sweep describes the motion of a body/shape for TOI computation.
        # Shapes are defined with respect to the body origin, which may no
        # coincide with the center of mass. However, to support dynamics we
        # must interpolate the center of mass position.
        sweepA = b2Sweep(c0=(0, 0), c=(0, 0),
                         a=0, a0=0,
                         localCenter=(0, 0))

        # The parameters of the sweep are defined as follows:
        # localCenter - local center of mass position
        # c0, c       - center world positions
        # a0, a       - world angles
        # t0          - time interval = [t0,1], where t0 is in [0,1]

        sweepB = b2Sweep(c0=(-0.20382018, 2.1368704),
                         a0=-3.1664171,
                         c=(-0.26699525, 2.3552670),
                         a=-3.3926492,
                         localCenter=(0, 0))

        type_, time_of_impact = b2TimeOfImpact(shapeA=self.shapeA,
                                               shapeB=self.shapeB,
                                               sweepA=sweepA, sweepB=sweepB,
                                               tMax=1.0)

        # Alternative pybox2d syntax (no kwargs):
        #  type_, t = b2TimeOfImpact(self.shapeA, self.shapeB, sweepA, sweepB, 1.0)
        #
        # And even uglier:
        #  input=b2TOIInput(proxyA=b2DistanceProxy(shape=self.shapeA), proxyB=b2DistanceProxy(shape=self.shapeB), sweepA=sweepA, sweepB=sweepB, tMax=1.0)
        #  type_, t = b2TimeOfImpact(input)

        self.Print("TOI = %g" % time_of_impact)
        self.Print("max toi iters = %d, max root iters = %d" %
                   (b2Globals.b2_toiMaxIters, b2Globals.b2_toiMaxRootIters))

        # Draw the shapes at their current position (t=0)
        # shapeA (the vertical polygon)
        transform = sweepA.GetTransform(0)
        self.renderer.DrawPolygon([self.renderer.to_screen(transform * v)
                                   for v in self.shapeA.vertices],
                                  b2Color(0.9, 0.9, 0.9))

        # shapeB (the horizontal polygon)
        transform = sweepB.GetTransform(0)
        self.renderer.DrawPolygon([self.renderer.to_screen(transform * v)
                                   for v in self.shapeB.vertices],
                                  b2Color(0.5, 0.9, 0.5))

        # localPoint=(2, -0.1)
        # rB = transform * localPoint - sweepB.c0
        # wB = sweepB.a - sweepB.a0
        # vB = sweepB.c - sweepB.c0
        # v = vB + b2Cross(wB, rB)

        # Now, draw shapeB in a different color when they would collide (i.e.,
        # at t=time of impact) This shows that the polygon would rotate upon
        # collision
        transform = sweepB.GetTransform(time_of_impact)
        self.renderer.DrawPolygon([self.renderer.to_screen(transform * v)
                                   for v in self.shapeB.vertices],
                                  b2Color(0.5, 0.7, 0.9))

        # And finally, draw shapeB at t=1.0, where it would be if it did not
        # collide with shapeA In this case, time_of_impact = 1.0, so these
        # become the same polygon.
        transform = sweepB.GetTransform(1.0)
        self.renderer.DrawPolygon([self.renderer.to_screen(transform * v)
                                   for v in self.shapeB.vertices],
                                  b2Color(0.9, 0.5, 0.5))

if __name__ == "__main__":
    main(TimeOfImpact)
