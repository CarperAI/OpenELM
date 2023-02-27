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

# Contributed by Giorgos Giagas (giorgosg)
# - updated for 2.1 by Ken

from math import sin, cos, pi

from .framework import (Framework, Keys, main)
from Box2D import (b2CircleShape, b2FixtureDef)


def create_blob(world, center, radius, circle_radius=0.5, shape_num=24,
                angularDamping=0.5, linearDamping=0.5, friction=0.5,
                density=5.0, **kwargs):

    def get_pos(angle):
        return (cos(angle * pi / 180.0) * radius + center[0],
                sin(angle * pi / 180.0) * radius + center[1])

    circle = b2CircleShape(radius=circle_radius)
    fixture = b2FixtureDef(shape=circle, friction=friction, density=density,
                           restitution=0.0)

    bodies = [world.CreateDynamicBody(position=get_pos(
        i), fixtures=fixture) for i in range(0, 360, int(360 / shape_num))]
    joints = []

    prev_body = bodies[-1]
    for body in bodies:
        joint = world.CreateDistanceJoint(
            bodyA=prev_body,
            bodyB=body,
            anchorA=prev_body.position,
            anchorB=body.position,
            dampingRatio=10.0)

        joints.append(joint)
        prev_body = body

    return bodies, joints


def add_spring_force(bodyA, localA, bodyB, localB, force_k, friction,
                     desiredDist):
    worldA = bodyA.GetWorldPoint(localA)
    worldB = bodyB.GetWorldPoint(localB)
    diff = worldB - worldA

    # Find velocities of attach points
    velA = bodyA.linearVelocity - \
        bodyA.GetWorldVector(localA).cross(bodyA.angularVelocity)
    velB = bodyB.linearVelocity - \
        bodyB.GetWorldVector(localB).cross(bodyB.angularVelocity)

    vdiff = velB - velA
    dx = diff.Normalize()  # Normalizes diff and puts length into dx
    vrel = vdiff.x * diff.x + vdiff.y * diff.y
    forceMag = -force_k * (dx - desiredDist) - friction * vrel

    bodyB.ApplyForce(diff * forceMag, bodyA.GetWorldPoint(localA), True)
    bodyA.ApplyForce(diff * -forceMag, bodyB.GetWorldPoint(localB), True)


def blob_step(world, blob_bodies, radius, upward_force, move=0,
              spring_friction=5.0):
    body_count = len(blob_bodies)
    bodies1, bodies2 = (blob_bodies[:body_count // 2],
                        blob_bodies[body_count // 2:])
    for body1, body2 in zip(bodies1, bodies2):
        add_spring_force(body1, (0, 0), body2, (0, 0),
                         upward_force, spring_friction, radius * 2)
    if move:
        top_body = [(body.position.y, body) for body in blob_bodies]
        top_body.sort(key=lambda val: val[0])
        top_body = top_body[-1][1]
        top_body.ApplyForce((move, 0), top_body.position, True)


class GishTribute (Framework):
    name = "Tribute to Gish"
    description = 'Keys: Left (a), Right (d), Jump (w), Stop (s)'
    move = 0
    jump = 100

    def __init__(self):
        super(GishTribute, self).__init__()

        # The ground
        ground = self.world.CreateStaticBody()
        ground.CreateEdgeFixture(vertices=[(-50, 0), (50, 0)], friction=0.2)
        ground.CreateEdgeFixture(vertices=[(-50, 0), (-50, 10)], friction=0.2)
        ground.CreateEdgeFixture(vertices=[(50, 0), (50, 10)], friction=0.2)

        for i in range(2, 18, 2):
            body = self.world.CreateDynamicBody(position=(-10.1, i))
            body.CreatePolygonFixture(box=(3.0, 1.0), density=3.0)

        self.blob_radius = 2
        self.bodies, self.joints = create_blob(
            self.world, (-10, 50), self.blob_radius, circle_radius=0.5)

    def Keyboard(self, key):
        if key == Keys.K_w:
            self.jump = 10000
        elif key == Keys.K_a:
            self.move = -500
        elif key == Keys.K_d:
            self.move = 500
        elif key == Keys.K_s:
            self.move = 0
            self.jump = 100

    def Step(self, settings):
        Framework.Step(self, settings)

        blob_step(self.world, self.bodies,
                  self.blob_radius, self.jump, self.move)
        self.jump = 100

if __name__ == "__main__":
    main(GishTribute)
