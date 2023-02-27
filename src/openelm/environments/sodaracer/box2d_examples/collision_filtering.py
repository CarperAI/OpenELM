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
from Box2D import (b2CircleShape, b2EdgeShape, b2Filter, b2FixtureDef,
                   b2PolygonShape, b2Vec2)


class CollisionFiltering (Framework):
    name = "Collision Filtering"
    description = "See which shapes collide with each other."
    # This is a test of collision filtering.
    # There is a triangle, a box, and a circle.
    # There are 6 shapes. 3 large and 3 small.
    # The 3 small ones always collide.
    # The 3 large ones never collide.
    # The boxes don't collide with triangles (except if both are small).
    # The box connected to the large triangle has no filter settings,
    # so it collides with everything.

    def __init__(self):
        super(CollisionFiltering, self).__init__()
        # Ground body
        world = self.world
        ground = world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-40, 0), (40, 0)])
        )

        # Define the groups that fixtures can fall into
        # Note that negative groups never collide with other negative ones.
        smallGroup = 1
        largeGroup = -1

        # And the categories
        # Note that these are bit-locations, and as such are written in
        # hexadecimal.
        # defaultCategory = 0x0001
        triangleCategory = 0x0002
        boxCategory = 0x0004
        circleCategory = 0x0008

        # And the masks that define which can hit one another
        # A mask of 0xFFFF means that it will collide with everything else in
        # its group. The box mask below uses an exclusive OR (XOR) which in
        # effect toggles the triangleCategory bit, making boxMask = 0xFFFD.
        # Such a mask means that boxes never collide with triangles.  (if
        # you're still confused, see the implementation details below)

        triangleMask = 0xFFFF
        boxMask = 0xFFFF ^ triangleCategory
        circleMask = 0xFFFF

        # The actual implementation determining whether or not two objects
        # collide is defined in the C++ source code, but it can be overridden
        # in Python (with b2ContactFilter).
        # The default behavior goes like this:
        #   if (filterA.groupIndex == filterB.groupIndex and filterA.groupIndex != 0):
        #       collide if filterA.groupIndex is greater than zero (negative groups never collide)
        #   else:
        #       collide if (filterA.maskBits & filterB.categoryBits) != 0 and (filterA.categoryBits & filterB.maskBits) != 0
        #
        # So, if they have the same group index (and that index isn't the
        # default 0), then they collide if the group index is > 0 (since
        # negative groups never collide)
        # (Note that a body with the default filter settings will always
        # collide with everything else.)
        # If their group indices differ, then only if their bitwise-ANDed
        # category and mask bits match up do they collide.
        #
        # For more help, some basics of bit masks might help:
        # -> http://en.wikipedia.org/wiki/Mask_%28computing%29

        # Small triangle
        triangle = b2FixtureDef(
            shape=b2PolygonShape(vertices=[(-1, 0), (1, 0), (0, 2)]),
            density=1,
            filter=b2Filter(
                groupIndex=smallGroup,
                categoryBits=triangleCategory,
                maskBits=triangleMask,
            )
        )

        world.CreateDynamicBody(
            position=(-5, 2),
            fixtures=triangle,
        )

        # Large triangle (recycle definitions)
        triangle.shape.vertices = [
            2.0 * b2Vec2(v) for v in triangle.shape.vertices]
        triangle.filter.groupIndex = largeGroup

        trianglebody = world.CreateDynamicBody(
            position=(-5, 6),
            fixtures=triangle,
            fixedRotation=True,  # <--
        )
        # note that the large triangle will not rotate

        # Small box
        box = b2FixtureDef(
            shape=b2PolygonShape(box=(1, 0.5)),
            density=1,
            restitution=0.1,
            filter = b2Filter(
                groupIndex=smallGroup,
                categoryBits=boxCategory,
                maskBits=boxMask,
            )
        )

        world.CreateDynamicBody(
            position=(0, 2),
            fixtures=box,
        )

        # Large box
        box.shape.box = (2, 1)
        box.filter.groupIndex = largeGroup
        world.CreateDynamicBody(
            position=(0, 6),
            fixtures=box,
        )

        # Small circle
        circle = b2FixtureDef(
            shape=b2CircleShape(radius=1),
            density=1,
            filter=b2Filter(
                groupIndex=smallGroup,
                categoryBits=circleCategory,
                maskBits=circleMask,
            )
        )

        world.CreateDynamicBody(
            position=(5, 2),
            fixtures=circle,
        )

        # Large circle
        circle.shape.radius *= 2
        circle.filter.groupIndex = largeGroup
        world.CreateDynamicBody(
            position=(5, 6),
            fixtures=circle,
        )

        # Create a joint for fun on the big triangle
        # Note that it does not inherit or have anything to do with the
        # filter settings of the attached triangle.
        box = b2FixtureDef(shape=b2PolygonShape(box=(0.5, 1)), density=1)

        testbody = world.CreateDynamicBody(
            position=(-5, 10),
            fixtures=box,
        )
        world.CreatePrismaticJoint(
            bodyA=trianglebody,
            bodyB=testbody,
            enableLimit=True,
            localAnchorA=(0, 4),
            localAnchorB=(0, 0),
            localAxisA=(0, 1),
            lowerTranslation=-1,
            upperTranslation=1,
        )


if __name__ == "__main__":
    main(CollisionFiltering)
