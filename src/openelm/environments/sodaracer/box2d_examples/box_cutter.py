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

# Original C++ version by Daid
#  http://www.box2d.org/forum/viewtopic.php?f=3&t=1473
# - Written for pybox2d 2.1 by Ken
import sys

from .framework import (Framework, Keys, main)
from Box2D import (b2AssertException, b2Color, b2EdgeShape, b2FixtureDef,
                   b2PolygonShape, b2RayCastCallback, b2Vec2)


LASER_HALF_WIDTH = 2
LASER_SPLIT_SIZE = 0.1
LASER_SPLIT_TAG = 'can_cut'


def _polygon_split(fixture, p1, p2, split_size):
    polygon = fixture.shape
    body = fixture.body
    # transform = body.transform

    local_entry = body.GetLocalPoint(p1)
    local_exit = body.GetLocalPoint(p2)
    entry_vector = local_exit - local_entry
    entry_normal = entry_vector.cross(1.0)
    last_verts = None
    new_vertices = [[], []]
    cut_added = [-1, -1]
    for vertex in polygon.vertices:
        # Find out if this vertex is on the new or old shape
        if entry_normal.dot(b2Vec2(vertex) - local_entry) > 0.0:
            verts = new_vertices[0]
        else:
            verts = new_vertices[1]

        if last_verts != verts:
            # if we switch from one shape to the other, add the cut vertices
            if last_verts == new_vertices[0]:
                if cut_added[0] != -1:
                    return []
                cut_added[0] = len(last_verts)
                last_verts.append(b2Vec2(local_exit))
                last_verts.append(b2Vec2(local_entry))
            elif last_verts == new_vertices[1]:
                if cut_added[1] != -1:
                    return []
                cut_added[1] = len(last_verts)
                last_verts.append(b2Vec2(local_entry))
                last_verts.append(b2Vec2(local_exit))

        verts.append(b2Vec2(vertex))
        last_verts = verts

    # Add the cut if not added yet
    if cut_added[0] < 0:
        cut_added[0] = len(new_vertices[0])
        new_vertices[0].append(b2Vec2(local_exit))
        new_vertices[0].append(b2Vec2(local_entry))
    if cut_added[1] < 0:
        cut_added[1] = len(new_vertices[1])
        new_vertices[1].append(b2Vec2(local_entry))
        new_vertices[1].append(b2Vec2(local_exit))

    # Cut based on the split size
    for added, verts in zip(cut_added, new_vertices):
        if added > 0:
            offset = verts[added - 1] - verts[added]
        else:
            offset = verts[-1] - verts[0]
        offset.Normalize()
        verts[added] += split_size * offset

        if added < len(verts) - 2:
            offset = verts[added + 2] - verts[added + 1]
        else:
            offset = verts[0] - verts[len(verts) - 1]
        offset.Normalize()
        verts[added + 1] += split_size * offset

    # Ensure the new shapes aren't too small
    for verts in new_vertices:
        for i, v1 in enumerate(verts):
            for j, v2 in enumerate(verts):
                if i != j and (v1 - v2).length < 0.1:
                    # print('Failed to split: too small')
                    return []

    try:
        return [b2PolygonShape(vertices=verts) for verts in new_vertices]
    except b2AssertException:
        return []
    except ValueError:
        return []


def _laser_cut(world, laser_body, length=30.0, laser_half_width=2, **kwargs):
    p1, p2 = get_laser_line(laser_body, length, laser_half_width)

    callback = laser_callback()
    world.RayCast(callback, p1, p2)
    if not callback.hit:
        return []
    hits_forward = callback.hits

    callback = laser_callback()
    world.RayCast(callback, p2, p1)
    if not callback.hit:
        return []

    hits_reverse = callback.hits

    if len(hits_forward) != len(hits_reverse):
        return []

    ret = []
    for (fixture1, point1), (fixture2, point2) in zip(hits_forward, hits_reverse):
        # renderer.DrawPoint(renderer.to_screen(point1), 2, b2Color(1,0,0))
        # renderer.DrawPoint(renderer.to_screen(point2), 2, b2Color(0,1,0))
        # renderer.DrawSegment(renderer.to_screen(point1), renderer.to_screen(point2), b2Color(0,1,1))
        if fixture1 != fixture2:
            continue

        new_polygons = _polygon_split(
            fixture1, point1, point2, LASER_SPLIT_SIZE)
        if new_polygons:
            ret.append((fixture1, new_polygons))

    return ret


def laser_cut(world, laser_body, length=30.0, laser_half_width=2, **kwargs):
    cut_fixtures = _laser_cut(
        world, laser_body, laser_half_width=LASER_HALF_WIDTH)
    remove_bodies = []
    for fixture, new_shapes in cut_fixtures:
        body = fixture.body
        if body in remove_bodies:
            continue

        new_body = world.CreateDynamicBody(
            userData=LASER_SPLIT_TAG,
            position=body.position,
            angle=body.angle,
            linearVelocity=body.linearVelocity,
            angularVelocity=body.angularVelocity,
        )

        try:
            new_body.CreateFixture(
                friction=fixture.friction,
                restitution=fixture.restitution,
                density=fixture.density,
                shape=new_shapes[1],
            )
        except AssertionError:
            print('New body fixture failed: %s' % sys.exc_info()[1])
            remove_bodies.append(new_body)

        try:
            body.CreateFixture(
                friction=fixture.friction,
                restitution=fixture.restitution,
                density=fixture.density,
                shape=new_shapes[0],
            )

            body.DestroyFixture(fixture)
        except AssertionError:
            print('New fixture/destroy failed: %s' % sys.exc_info()[1])
            remove_bodies.append(body)

    for body in remove_bodies:
        world.DestroyBody(body)


def get_laser_line(laser_body, length, laser_half_width):
    laser_start = (laser_half_width - 0.1, 0.0)
    laser_dir = (length, 0.0)
    p1 = laser_body.GetWorldPoint(laser_start)
    p2 = p1 + laser_body.GetWorldVector(laser_dir)
    return (p1, p2)


def laser_display(renderer, laser_body, length=30.0, laser_color=(1, 0, 0), laser_half_width=2, **kwargs):
    if not renderer:
        return

    p1, p2 = get_laser_line(laser_body, length, laser_half_width)
    renderer.DrawSegment(renderer.to_screen(
        p1), renderer.to_screen(p2), b2Color(*laser_color))


class laser_callback(b2RayCastCallback):
    """This raycast collects multiple hits."""

    def __init__(self, **kwargs):
        b2RayCastCallback.__init__(self, **kwargs)
        self.hit = False
        self.hits = []

    def ReportFixture(self, fixture, point, normal, fraction):
        self.hit = True

        if fixture.body.userData == LASER_SPLIT_TAG:
            self.hits.append((fixture, point))

        self.last_fixture = fixture
        self.last_point = point
        return 1.0


class BoxCutter(Framework):
    name = "Box Cutter"
    description = 'Press (c) to cut'
    move = 0
    jump = 100

    def __init__(self):
        super(BoxCutter, self).__init__()
        # The ground
        self.ground = self.world.CreateStaticBody(
            userData='ground',
            shapes=[
                b2EdgeShape(vertices=[(-50, 0), (50, 0)]),
                b2EdgeShape(vertices=[(-50, 0), (-50, 10)]),
                b2EdgeShape(vertices=[(50, 0), (50, 10)]),
            ]
        )

        self.laser_body = self.world.CreateDynamicBody(
            userData='laser',
            position=(0, 2),
            fixtures=b2FixtureDef(
                density=4.0,
                shape=b2PolygonShape(box=(LASER_HALF_WIDTH, 1))
            )
        )

        for i in range(2):
            self.world.CreateDynamicBody(
                userData=LASER_SPLIT_TAG,
                position=(3.0 + i * 6, 8),
                fixtures=b2FixtureDef(
                    density=5.0,
                    shape=b2PolygonShape(box=(3, 3))
                )
            )

    def Keyboard(self, key):
        if key == Keys.K_c:
            laser_cut(self.world, self.laser_body,
                      laser_half_width=LASER_HALF_WIDTH)

    def Step(self, settings):
        Framework.Step(self, settings)

        laser_display(self.renderer, self.laser_body,
                      laser_half_width=LASER_HALF_WIDTH)

if __name__ == "__main__":
    main(BoxCutter)
