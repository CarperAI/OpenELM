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

"""
A simple, minimal Pygame-based backend.
It will only draw and support very basic keyboard input (ESC to quit).

There are no main dependencies other than the actual test you are running.
Note that this only relies on framework.py for the loading of this backend,
and holding the Keys class. If you write a test that depends only on this
backend, you can remove references to that file here and import this module
directly in your test.

To use this backend, try:
 % python -m examples.web --backend simple

NOTE: Examples with Step() re-implemented are not yet supported, as I wanted
to do away with the Settings class. This means the following will definitely
not work: Breakable, Liquid, Raycast, TimeOfImpact, ... (incomplete)
"""

import pygame
from pygame.locals import (QUIT, KEYDOWN, KEYUP)
import Box2D
from Box2D.b2 import (world, staticBody, dynamicBody, kinematicBody,
                      polygonShape, circleShape, edgeShape, loopShape)

TARGET_FPS = 60
PPM = 10.0
TIMESTEP = 1.0 / TARGET_FPS
VEL_ITERS, POS_ITERS = 10, 10
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
SCREEN_OFFSETX, SCREEN_OFFSETY = SCREEN_WIDTH * 1.0 / 2.0, SCREEN_HEIGHT * 2.0 / 3
colors = {
    staticBody: (255, 255, 255, 255),
    dynamicBody: (127, 127, 127, 255),
    kinematicBody: (127, 127, 230, 255),
}


def fix_vertices(vertices):
    return [(int(SCREEN_OFFSETX + v[0]), int(SCREEN_OFFSETY - v[1])) for v in vertices]


def _draw_polygon(polygon, screen, body, fixture):
    transform = body.transform
    vertices = fix_vertices([transform * v * PPM for v in polygon.vertices])
    pygame.draw.polygon(
        screen, [c / 2.0 for c in colors[body.type]], vertices, 0)
    pygame.draw.polygon(screen, colors[body.type], vertices, 1)
polygonShape.draw = _draw_polygon


def _draw_circle(circle, screen, body, fixture):
    position = fix_vertices([body.transform * circle.pos * PPM])[0]
    pygame.draw.circle(screen, colors[body.type],
                       position, int(circle.radius * PPM))
circleShape.draw = _draw_circle


def _draw_edge(edge, screen, body, fixture):
    vertices = fix_vertices(
        [body.transform * edge.vertex1 * PPM, body.transform * edge.vertex2 * PPM])
    pygame.draw.line(screen, colors[body.type], vertices[0], vertices[1])
edgeShape.draw = _draw_edge


def _draw_loop(loop, screen, body, fixture):
    transform = body.transform
    vertices = fix_vertices([transform * v * PPM for v in loop.vertices])
    v1 = vertices[-1]
    for v2 in vertices:
        pygame.draw.line(screen, colors[body.type], v1, v2)
        v1 = v2
loopShape.draw = _draw_loop


def draw_world(screen, world):
    # Draw the world
    for body in world.bodies:
        for fixture in body.fixtures:
            fixture.shape.draw(screen, body, fixture)


class Keys(object):
    pass

# The following import is only needed to do the initial loading and
# overwrite the Keys class.
import framework
# Set up the keys (needed as the normal framework abstracts them between
# backends)
keys = [s for s in dir(pygame.locals) if s.startswith('K_')]
for key in keys:
    value = getattr(pygame.locals, key)
    setattr(Keys, key, value)
framework.Keys = Keys


class SimpleFramework(object):
    name = 'None'
    description = ''

    def __init__(self):
        self.world = world()

        print('Initializing pygame framework...')
        # Pygame Initialization
        pygame.init()
        caption = "Python Box2D Testbed - Simple backend - " + self.name
        pygame.display.set_caption(caption)

        # Screen and debug draw
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.font = pygame.font.Font(None, 15)

        self.groundbody = self.world.CreateBody()

    def run(self):
        """
        Main loop.

        Updates the world and then the screen.
        """

        running = True
        clock = pygame.time.Clock()
        while running:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == Keys.K_ESCAPE):
                    running = False
                elif event.type == KEYDOWN:
                    self.Keyboard(event.key)
                elif event.type == KEYUP:
                    self.KeyboardUp(event.key)

            self.screen.fill((0, 0, 0))

            self.textLine = 15

            # Step the world
            self.world.Step(TIMESTEP, VEL_ITERS, POS_ITERS)
            self.world.ClearForces()

            draw_world(self.screen, self.world)

            # Draw the name of the test running
            self.Print(self.name, (127, 127, 255))

            if self.description:
                # Draw the name of the test running
                for s in self.description.split('\n'):
                    self.Print(s, (127, 255, 127))

            pygame.display.flip()
            clock.tick(TARGET_FPS)
            self.fps = clock.get_fps()

        self.world.contactListener = None
        self.world.destructionListener = None
        self.world.renderer = None

    def Print(self, str, color=(229, 153, 153, 255)):
        """
        Draw some text at the top status lines
        and advance to the next line.
        """
        self.screen.blit(self.font.render(
            str, True, color), (5, self.textLine))
        self.textLine += 15

    def Keyboard(self, key):
        """
        Callback indicating 'key' has been pressed down.
        The keys are mapped after pygame's style.

         from .framework import Keys
         if key == Keys.K_z:
             ...
        """
        pass

    def KeyboardUp(self, key):
        """
        Callback indicating 'key' has been released.
        See Keyboard() for key information
        """
        pass
