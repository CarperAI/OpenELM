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
from Box2D import (b2CircleShape, b2EdgeShape, b2FixtureDef, b2PolygonShape,
                   b2Vec2, b2_pi)

# Original inspired by a contribution by roman_m
# Dimensions scooped from APE (http://www.cove.org/ape/index.htm)


class TheoJansen (Framework):
    name = "Theo Jansen"
    description = "Keys: left = a, brake = s, right = d, toggle motor = m"
    motorSpeed = 2
    motorOn = True
    offset = (0, 8)

    def __init__(self):
        super(TheoJansen, self).__init__()

        #
        ball_count = 40
        pivot = b2Vec2(0, 0.8)

        # The ground
        ground = self.world.CreateStaticBody(
            shapes=[
                b2EdgeShape(vertices=[(-50, 0), (50, 0)]),
                b2EdgeShape(vertices=[(-50, 0), (-50, 10)]),
                b2EdgeShape(vertices=[(50, 0), (50, 10)]),
            ]
        )

        box = b2FixtureDef(
            shape=b2PolygonShape(box=(0.5, 0.5)),
            density=1,
            friction=0.3)
        circle = b2FixtureDef(
            shape=b2CircleShape(radius=0.25),
            density=1)

        # Create the balls on the ground
        for i in range(ball_count):
            self.world.CreateDynamicBody(
                fixtures=circle,
                position=(-40 + 2.0 * i, 0.5),
            )

        # The chassis
        chassis_fixture = b2FixtureDef(
            shape=b2PolygonShape(box=(2.5, 1)),
            density=1,
            friction=0.3,
            groupIndex=-1)

        self.chassis = self.world.CreateDynamicBody(
            fixtures=chassis_fixture,
            position=pivot + self.offset)

        # Chassis wheel
        wheel_fixture = b2FixtureDef(
            shape=b2CircleShape(radius=1.6),
            density=1,
            friction=0.3,
            groupIndex=-1)

        self.wheel = self.world.CreateDynamicBody(
            fixtures=wheel_fixture,
            position=pivot + self.offset)

        # Add a joint between the chassis wheel and the chassis itself
        self.motorJoint = self.world.CreateRevoluteJoint(
            bodyA=self.wheel,
            bodyB=self.chassis,
            anchor=pivot + self.offset,
            collideConnected=False,
            motorSpeed=self.motorSpeed,
            maxMotorTorque=400,
            enableMotor=self.motorOn)

        wheelAnchor = pivot + (0, -0.8)
        self.CreateLeg(-1, wheelAnchor)
        self.CreateLeg(1, wheelAnchor)

        self.wheel.transform = (self.wheel.position, 120.0 * b2_pi / 180)
        self.CreateLeg(-1, wheelAnchor)
        self.CreateLeg(1, wheelAnchor)

        self.wheel.transform = (self.wheel.position, -120.0 * b2_pi / 180)
        self.CreateLeg(-1, wheelAnchor)
        self.CreateLeg(1, wheelAnchor)

    def CreateLeg(self, s, wheelAnchor):
        p1, p2 = b2Vec2(5.4 * s, -6.1), b2Vec2(7.2 * s, -1.2)
        p3, p4 = b2Vec2(4.3 * s, -1.9), b2Vec2(3.1 * s, 0.8)
        p5, p6 = b2Vec2(6.0 * s, 1.5), b2Vec2(2.5 * s, 3.7)

        # Use a simple system to create mirrored vertices
        if s > 0:
            poly1 = b2PolygonShape(vertices=(p1, p2, p3))
            poly2 = b2PolygonShape(vertices=((0, 0), p5 - p4, p6 - p4))
        else:
            poly1 = b2PolygonShape(vertices=(p1, p3, p2))
            poly2 = b2PolygonShape(vertices=((0, 0), p6 - p4, p5 - p4))

        body1 = self.world.CreateDynamicBody(
            position=self.offset,
            angularDamping=10,
            fixtures=b2FixtureDef(
                shape=poly1,
                groupIndex=-1,
                density=1),
        )

        body2 = self.world.CreateDynamicBody(
            position=p4 + self.offset,
            angularDamping=10,
            fixtures=b2FixtureDef(
                shape=poly2,
                groupIndex=-1,
                density=1),
        )

        # Using a soft distance constraint can reduce some jitter.
        # It also makes the structure seem a bit more fluid by
        # acting like a suspension system.
        # Now, join all of the bodies together with distance joints,
        # and one single revolute joint on the chassis
        self.world.CreateDistanceJoint(
            dampingRatio=0.5,
            frequencyHz=10,
            bodyA=body1, bodyB=body2,
            anchorA=p2 + self.offset,
            anchorB=p5 + self.offset,
        )

        self.world.CreateDistanceJoint(
            dampingRatio=0.5,
            frequencyHz=10,
            bodyA=body1, bodyB=body2,
            anchorA=p3 + self.offset,
            anchorB=p4 + self.offset,
        )

        self.world.CreateDistanceJoint(
            dampingRatio=0.5,
            frequencyHz=10,
            bodyA=body1, bodyB=self.wheel,
            anchorA=p3 + self.offset,
            anchorB=wheelAnchor + self.offset,
        )

        self.world.CreateDistanceJoint(
            dampingRatio=0.5,
            frequencyHz=10,
            bodyA=body2, bodyB=self.wheel,
            anchorA=p6 + self.offset,
            anchorB=wheelAnchor + self.offset,
        )

        self.world.CreateRevoluteJoint(
            bodyA=body2,
            bodyB=self.chassis,
            anchor=p4 + self.offset,
        )

    def Keyboard(self, key):
        if key == Keys.K_a:
            self.motorJoint.motorSpeed = -self.motorSpeed
        elif key == Keys.K_d:
            self.motorJoint.motorSpeed = self.motorSpeed
        elif key == Keys.K_s:
            self.motorJoint.motorSpeed = 0
        elif key == Keys.K_m:
            self.motorJoint.motorEnabled = not self.motorJoint.motorEnabled

if __name__ == "__main__":
    main(TheoJansen)
