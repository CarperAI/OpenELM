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
from Box2D import (b2CircleShape, b2EdgeShape, b2FixtureDef, b2PolygonShape,
                   b2_linearSlop)


class OneSidedPlatform (Framework):
    name = "One-sided Platform"
    description = ("Try to move the shape with the mouse through the platform "
                   "from all directions")

    def __init__(self):
        super(OneSidedPlatform, self).__init__()

        # The ground
        ground = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-20, 0), (20, 0)])
        )

        # The platform
        half_height = 0.5
        ypos = 10
        body = self.world.CreateBody(
            position=(0, ypos),
            shapes=b2PolygonShape(box=(3, half_height))
        )
        self.platform = body.fixtures[0]

        # The circular character
        self.character_radius = 0.5
        body = self.world.CreateDynamicBody(
            position=(0, 12),
            fixtures=b2FixtureDef(shape=b2CircleShape(
                radius=self.character_radius), density=1.0),
        )

        self.character = body.fixtures[0]
        body.linearVelocity = (0, -50)

        self.bottom = ypos - half_height  # The bottom of the platform
        self.top = ypos + half_height    # The top of the platform
        self.state = 'unknown'

    def PreSolve(self, contact, oldManifold):
        super(OneSidedPlatform, self).PreSolve(contact, oldManifold)

        #  Make sure we're dealing with the platform and the character
        if (contact.fixtureA != self.platform and
                contact.fixtureA != self.character):
            return
        if (contact.fixtureB != self.platform and
                contact.fixtureB != self.character):
            return

        # If below the top of the platform, disable the collision response
        pos = (self.character.body.position.y - self.character_radius +
               3.0 * b2_linearSlop)
        if pos < self.top:
            contact.enabled = False

if __name__ == "__main__":
    main(OneSidedPlatform)
