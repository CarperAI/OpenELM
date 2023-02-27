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

from math import sqrt

from .framework import (Framework, Keys, main)
from Box2D import (b2CircleShape, b2FixtureDef, b2PolygonShape, b2Random,
                   b2Vec2, b2_epsilon)

# ***** NOTE *****
# ***** NOTE *****
# This example does not appear to be working currently...
# It was ported from the JBox2D (Java) version
# ***** NOTE *****
# ***** NOTE *****


class Liquid (Framework):
    name = "Liquid Test"
    description = ''
    bullet = None

    num_particles = 1000
    total_mass = 10.0

    fluid_minx = -11.0
    fluid_maxx = 5.0
    fluid_miny = -10.0
    fluid_maxy = 10.0

    hash_width = 40
    hash_height = 40

    rad = 0.6
    visc = 0.004

    def __init__(self):
        super(Liquid, self).__init__()

        self.per_particle_mass = self.total_mass / self.num_particles

        ground = self.world.CreateStaticBody(
            shapes=[
                b2PolygonShape(box=[5.0, 0.5]),
                b2PolygonShape(box=[1.0, 0.2, (0, 4), -0.2]),
                b2PolygonShape(box=[1.5, 0.2, (-1.2, 5.2), -1.5]),
                b2PolygonShape(box=[0.5, 50.0, (5, 0), 0.0]),
                b2PolygonShape(box=[0.5, 3.0, (-8, 0), 0.0]),
                b2PolygonShape(box=[2.0, 0.1, (-6, -2.8), 0.1]),
                b2CircleShape(radius=0.5, pos=(-.5, -4)),
            ]
        )

        cx = 0
        cy = 25
        box_width = 2.0
        box_height = 20.0
        self.liquid = []
        for i in range(self.num_particles):
            self.createDroplet((b2Random(cx - box_width * 0.5,
                                         cx + box_width * 0.5),
                                b2Random(cy - box_height * 0.5,
                                         cy + box_height * 0.5)))

        self.createBoxSurfer()

        if hasattr(self, 'settings'):
            self.settings.enableSubStepping = False

    def createBoxSurfer(self):
        self.surfer = self.world.CreateDynamicBody(position=(0, 25))
        self.surfer.CreatePolygonFixture(
            density=1,
            box=(b2Random(0.3, 0.7), b2Random(0.3, 0.7)),
        )

    def createDroplet(self, position):
        body = self.world.CreateDynamicBody(
            position=position,
            fixedRotation=True,
            allowSleep=False,
        )
        body.CreateCircleFixture(
            groupIndex=-10,
            radius=0.05,
            restitution=0.4,
            friction=0,
        )
        body.mass = self.per_particle_mass
        self.liquid.append(body)

    def applyLiquidConstraint(self, dt):
        # (original comments left untouched)
        # Unfortunately, this simulation method is not actually scale
        # invariant, and it breaks down for rad < ~3 or so.  So we need
        # to scale everything to an ideal rad and then scale it back after.

        idealRad = 50
        idealRad2 = idealRad ** 2
        multiplier = idealRad / self.rad
        info = dict([(drop, (drop.position, multiplier * drop.position,
                             multiplier * drop.linearVelocity))
                     for drop in self.liquid])
        change = dict([(drop, b2Vec2(0, 0)) for drop in self.liquid])
        dx = self.fluid_maxx - self.fluid_minx
        dy = self.fluid_maxy - self.fluid_miny
        range_ = (-1, 0, 1)
        hash_width = self.hash_width
        hash_height = self.hash_height
        max_len = 9.9e9
        visc = self.visc
        hash = self.hash
        neighbors = set()
        # Populate the neighbor list from the 9 nearest cells
        for drop, ((worldx, worldy), (mx, my), (mvx, mvy)) in list(info.items()):
            hx = int((worldx / dx) * hash_width)
            hy = int((worldy / dy) * hash_height)
            neighbors.clear()
            for nx in range_:
                xc = hx + nx
                if not (0 <= xc < hash_width):
                    continue

                for ny in range_:
                    yc = hy + ny
                    if yc in hash[xc]:
                        for neighbor in hash[xc][yc]:
                            neighbors.add(neighbor)

            if drop in neighbors:
                neighbors.remove(drop)

            # Particle pressure calculated by particle proximity
            # Pressures = 0 iff all particles within range are idealRad
            # distance away
            lengths = []
            p = 0
            pnear = 0
            for neighbor in neighbors:
                nx, ny = info[neighbor][1]
                vx, vy = nx - mx, ny - my
                if -idealRad < vx < idealRad and -idealRad < vy < idealRad:
                    len_sqr = vx ** 2 + vy ** 2
                    if len_sqr < idealRad2:
                        len_ = sqrt(len_sqr)
                        if len_ < b2_epsilon:
                            len_ = idealRad - 0.01
                        lengths.append(len_)
                        oneminusq = 1.0 - (len_ / idealRad)
                        sq = oneminusq ** 2
                        p += sq
                        pnear += sq * oneminusq
                else:
                    lengths.append(max_len)

            # Now actually apply the forces
            pressure = (p - 5) / 2.0  # normal pressure term
            presnear = pnear / 2.0  # near particles term
            changex, changey = 0, 0
            for len_, neighbor in zip(lengths, neighbors):
                (nx, ny), (nvx, nvy) = info[neighbor][1:3]
                vx, vy = nx - mx, ny - my
                if -idealRad < vx < idealRad and -idealRad < vy < idealRad:
                    if len_ < idealRad:
                        oneminusq = 1.0 - (len_ / idealRad)
                        factor = oneminusq * \
                            (pressure + presnear * oneminusq) / (2.0 * len_)
                        dx_, dy_ = vx * factor, vy * factor
                        relvx, relvy = nvx - mvx, nvy - mvy

                        factor = visc * oneminusq * dt
                        dx_ -= relvx * factor
                        dy_ -= relvy * factor
                        change[neighbor] += (dx_, dy_)
                        changex -= dx_
                        changey -= dy_

            change[drop] += (changex, changey)

        for drop, (dx_, dy_) in list(change.items()):
            if dx_ != 0 or dy_ != 0:
                drop.position += (dx_ / multiplier, dy_ / multiplier)
                drop.linearVelocity += (dx_ / (multiplier * dt),
                                        dy_ / (multiplier * dt))

    def hashLocations(self):
        hash_width = self.hash_width
        hash_height = self.hash_height

        self.hash = hash = dict([(i, {}) for i in range(hash_width)])
        info = [(drop, drop.position) for drop in self.liquid]

        dx = self.fluid_maxx - self.fluid_minx
        dy = self.fluid_maxy - self.fluid_miny
        xs, ys = set(), set()
        for drop, (worldx, worldy) in info:
            hx = int((worldx / dx) * hash_width)
            hy = int((worldy / dy) * hash_height)
            xs.add(hx)
            ys.add(hy)
            if 0 <= hx < hash_width and 0 <= hy < hash_height:
                x = hash[hx]
                if hy not in x:
                    x[hy] = [drop]
                else:
                    x[hy].append(drop)

    def dampenLiquid(self):
        for drop in self.liquid:
            drop.linearVelocity *= 0.995

    def checkBounds(self):
        self.hash = None

        to_remove = [
            drop for drop in self.liquid if drop.position.y < self.fluid_miny]
        for drop in to_remove:
            self.liquid.remove(drop)
            self.world.DestroyBody(drop)

            self.createDroplet(
                (0.0 + b2Random(-0.6, 0.6), 15.0 + b2Random(-2.3, 2.0)))

        if self.surfer.position.y < -15:
            self.world.DestroyBody(self.surfer)
            self.createBoxSurfer()

    def Step(self, settings):
        super(Liquid, self).Step(settings)

        dt = 1.0 / settings.hz
        self.hashLocations()
        self.applyLiquidConstraint(dt)
        self.dampenLiquid()
        self.checkBounds()

    def Keyboard(self, key):
        if key == Keys.K_b:
            if self.bullet:
                self.world.DestroyBody(self.bullet)
                self.bullet = None
            circle = b2FixtureDef(
                shape=b2CircleShape(radius=0.25),
                density=20,
                restitution=0.05)
            self.bullet = self.world.CreateDynamicBody(
                position=(-31, 5),
                bullet=True,
                fixtures=circle,
                linearVelocity=(400, 0),
            )

if __name__ == "__main__":
    main(Liquid)
