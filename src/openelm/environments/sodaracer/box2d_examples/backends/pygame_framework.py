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
Global Keys:
    F1     - toggle menu (can greatly improve fps)
    Space  - shoot projectile
    Z/X    - zoom
    Escape - quit

Other keys can be set by the individual test.

Mouse:
    Left click  - select/drag body (creates mouse joint)
    Right click - pan
    Shift+Left  - drag to create a directed projectile
    Scroll      - zoom

"""

from __future__ import (print_function, absolute_import, division)
import sys
import warnings

try:
    import pygame_sdl2
except ImportError:
    if sys.platform in ('darwin', ):
        warnings.warn('OSX has major issues with pygame/SDL 1.2 when used '
                      'inside a virtualenv. If this affects you, try '
                      'installing the updated pygame_sdl2 library.')
else:
    # pygame_sdl2 is backward-compatible with pygame:
    pygame_sdl2.import_as_pygame()

import pygame
from pygame.locals import (QUIT, KEYDOWN, KEYUP, MOUSEBUTTONDOWN,
                           MOUSEBUTTONUP, MOUSEMOTION, KMOD_LSHIFT)

from ..framework import (FrameworkBase, Keys)
from ..settings import fwSettings
from Box2D import (b2DrawExtended, b2Vec2)

try:
    from .pygame_gui import (fwGUI, gui)
    GUIEnabled = True
except Exception as ex:
    print('Unable to load PGU; menu disabled.')
    print('(%s) %s' % (ex.__class__.__name__, ex))
    GUIEnabled = False


class PygameDraw(b2DrawExtended):
    """
    This debug draw class accepts callbacks from Box2D (which specifies what to
    draw) and handles all of the rendering.

    If you are writing your own game, you likely will not want to use debug
    drawing.  Debug drawing, as its name implies, is for debugging.
    """
    surface = None
    axisScale = 10.0

    def __init__(self, test=None, **kwargs):
        b2DrawExtended.__init__(self, **kwargs)
        self.flipX = False
        self.flipY = True
        self.convertVertices = True
        self.test = test

    def StartDraw(self):
        self.zoom = self.test.viewZoom
        self.center = self.test.viewCenter
        self.offset = self.test.viewOffset
        self.screenSize = self.test.screenSize

    def EndDraw(self):
        pass

    def DrawPoint(self, p, size, color):
        """
        Draw a single point at point p given a pixel size and color.
        """
        self.DrawCircle(p, size / self.zoom, color, drawwidth=0)

    def DrawAABB(self, aabb, color):
        """
        Draw a wireframe around the AABB with the given color.
        """
        points = [(aabb.lowerBound.x, aabb.lowerBound.y),
                  (aabb.upperBound.x, aabb.lowerBound.y),
                  (aabb.upperBound.x, aabb.upperBound.y),
                  (aabb.lowerBound.x, aabb.upperBound.y)]

        pygame.draw.aalines(self.surface, color, True, points)

    def DrawSegment(self, p1, p2, color):
        """
        Draw the line segment from p1-p2 with the specified color.
        """
        pygame.draw.aaline(self.surface, color.bytes, p1, p2)

    def DrawTransform(self, xf):
        """
        Draw the transform xf on the screen
        """
        p1 = xf.position
        p2 = self.to_screen(p1 + self.axisScale * xf.R.x_axis)
        p3 = self.to_screen(p1 + self.axisScale * xf.R.y_axis)
        p1 = self.to_screen(p1)
        pygame.draw.aaline(self.surface, (255, 0, 0), p1, p2)
        pygame.draw.aaline(self.surface, (0, 255, 0), p1, p3)

    def DrawCircle(self, center, radius, color, drawwidth=1):
        """
        Draw a wireframe circle given the center, radius, axis of orientation
        and color.
        """
        radius *= self.zoom
        if radius < 1:
            radius = 1
        else:
            radius = int(radius)

        pygame.draw.circle(self.surface, color.bytes,
                           center, radius, drawwidth)

    def DrawSolidCircle(self, center, radius, axis, color):
        """
        Draw a solid circle given the center, radius, axis of orientation and
        color.
        """
        radius *= self.zoom
        if radius < 1:
            radius = 1
        else:
            radius = int(radius)

        pygame.draw.circle(self.surface, (color / 2).bytes + [127],
                           center, radius, 0)
        pygame.draw.circle(self.surface, color.bytes, center, radius, 1)
        pygame.draw.aaline(self.surface, (255, 0, 0), center,
                           (center[0] - radius * axis[0],
                            center[1] + radius * axis[1]))

    def DrawPolygon(self, vertices, color):
        """
        Draw a wireframe polygon given the screen vertices with the specified color.
        """
        if not vertices:
            return

        if len(vertices) == 2:
            pygame.draw.aaline(self.surface, color.bytes,
                               vertices[0], vertices)
        else:
            pygame.draw.polygon(self.surface, color.bytes, vertices, 1)

    def DrawSolidPolygon(self, vertices, color):
        """
        Draw a filled polygon given the screen vertices with the specified color.
        """
        if not vertices:
            return

        if len(vertices) == 2:
            pygame.draw.aaline(self.surface, color.bytes,
                               vertices[0], vertices[1])
        else:
            pygame.draw.polygon(
                self.surface, (color / 2).bytes + [127], vertices, 0)
            pygame.draw.polygon(self.surface, color.bytes, vertices, 1)

    # the to_screen conversions are done in C with b2DrawExtended, leading to
    # an increase in fps.
    # You can also use the base b2Draw and implement these yourself, as the
    # b2DrawExtended is implemented:
    # def to_screen(self, point):
    #     """
    #     Convert from world to screen coordinates.
    #     In the class instance, we store a zoom factor, an offset indicating where
    #     the view extents start at, and the screen size (in pixels).
    #     """
    #     x=(point.x * self.zoom)-self.offset.x
    #     if self.flipX:
    #         x = self.screenSize.x - x
    #     y=(point.y * self.zoom)-self.offset.y
    #     if self.flipY:
    #         y = self.screenSize.y-y
    #     return (x, y)


class PygameFramework(FrameworkBase):
    TEXTLINE_START = 30

    def setup_keys(self):
        keys = [s for s in dir(pygame.locals) if s.startswith('K_')]
        for key in keys:
            value = getattr(pygame.locals, key)
            setattr(Keys, key, value)

    def __reset(self):
        # Screen/rendering-related
        self._viewZoom = 10.0
        self._viewCenter = None
        self._viewOffset = None
        self.screenSize = None
        self.rMouseDown = False
        self.textLine = 30
        self.font = None
        self.fps = 0

        # GUI-related (PGU)
        self.gui_app = None
        self.gui_table = None
        self.setup_keys()

    def __init__(self):
        super(PygameFramework, self).__init__()

        self.__reset()
        if fwSettings.onlyInit:  # testing mode doesn't initialize pygame
            return

        print('Initializing pygame framework...')
        # Pygame Initialization
        pygame.init()
        caption = "Python Box2D Testbed - " + self.name
        pygame.display.set_caption(caption)

        # Screen and debug draw
        self.screen = pygame.display.set_mode((640, 480))
        self.screenSize = b2Vec2(*self.screen.get_size())

        self.renderer = PygameDraw(surface=self.screen, test=self)
        self.world.renderer = self.renderer

        try:
            self.font = pygame.font.Font(None, 15)
        except IOError:
            try:
                self.font = pygame.font.Font("freesansbold.ttf", 15)
            except IOError:
                print("Unable to load default font or 'freesansbold.ttf'")
                print("Disabling text drawing.")
                self.Print = lambda *args: 0
                self.DrawStringAt = lambda *args: 0

        # GUI Initialization
        if GUIEnabled:
            self.gui_app = gui.App()
            self.gui_table = fwGUI(self.settings)
            container = gui.Container(align=1, valign=-1)
            container.add(self.gui_table, 0, 0)
            self.gui_app.init(container)

        self.viewCenter = (0, 20.0)
        self.groundbody = self.world.CreateBody()

    def setCenter(self, value):
        """
        Updates the view offset based on the center of the screen.

        Tells the debug draw to update its values also.
        """
        self._viewCenter = b2Vec2(*value)
        self._viewCenter *= self._viewZoom
        self._viewOffset = self._viewCenter - self.screenSize / 2

    def setZoom(self, zoom):
        self._viewZoom = zoom

    viewZoom = property(lambda self: self._viewZoom, setZoom,
                        doc='Zoom factor for the display')
    viewCenter = property(lambda self: self._viewCenter / self._viewZoom, setCenter,
                          doc='Screen center in camera coordinates')
    viewOffset = property(lambda self: self._viewOffset,
                          doc='The offset of the top-left corner of the screen')

    def checkEvents(self):
        """
        Check for pygame events (mainly keyboard/mouse events).
        Passes the events onto the GUI also.
        """
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == Keys.K_ESCAPE):
                return False
            elif event.type == KEYDOWN:
                self._Keyboard_Event(event.key, down=True)
            elif event.type == KEYUP:
                self._Keyboard_Event(event.key, down=False)
            elif event.type == MOUSEBUTTONDOWN:
                p = self.ConvertScreenToWorld(*event.pos)
                if event.button == 1:  # left
                    mods = pygame.key.get_mods()
                    if mods & KMOD_LSHIFT:
                        self.ShiftMouseDown(p)
                    else:
                        self.MouseDown(p)
                elif event.button == 2:  # middle
                    pass
                elif event.button == 3:  # right
                    self.rMouseDown = True
                elif event.button == 4:
                    self.viewZoom *= 1.1
                elif event.button == 5:
                    self.viewZoom /= 1.1
            elif event.type == MOUSEBUTTONUP:
                p = self.ConvertScreenToWorld(*event.pos)
                if event.button == 3:  # right
                    self.rMouseDown = False
                else:
                    self.MouseUp(p)
            elif event.type == MOUSEMOTION:
                p = self.ConvertScreenToWorld(*event.pos)

                self.MouseMove(p)

                if self.rMouseDown:
                    self.viewCenter -= (event.rel[0] /
                                        5.0, -event.rel[1] / 5.0)

            if GUIEnabled:
                self.gui_app.event(event)  # Pass the event to the GUI

        return True

    def run(self):
        """
        Main loop.

        Continues to run while checkEvents indicates the user has
        requested to quit.

        Updates the screen and tells the GUI to paint itself.
        """

        # If any of the test constructors update the settings, reflect
        # those changes on the GUI before running
        if GUIEnabled:
            self.gui_table.updateGUI(self.settings)

        running = True
        clock = pygame.time.Clock()
        while running:
            running = self.checkEvents()
            self.screen.fill((0, 0, 0))

            # Check keys that should be checked every loop (not only on initial
            # keydown)
            self.CheckKeys()

            # Run the simulation loop
            self.SimulationLoop()

            if GUIEnabled and self.settings.drawMenu:
                self.gui_app.paint(self.screen)

            pygame.display.flip()
            clock.tick(self.settings.hz)
            self.fps = clock.get_fps()

        self.world.contactListener = None
        self.world.destructionListener = None
        self.world.renderer = None

    def _Keyboard_Event(self, key, down=True):
        """
        Internal keyboard event, don't override this.

        Checks for the initial keydown of the basic testbed keys. Passes the unused
        ones onto the test via the Keyboard() function.
        """
        if down:
            if key == Keys.K_z:       # Zoom in
                self.viewZoom = min(1.1 * self.viewZoom, 50.0)
            elif key == Keys.K_x:     # Zoom out
                self.viewZoom = max(0.9 * self.viewZoom, 0.02)
            elif key == Keys.K_SPACE:  # Launch a bomb
                self.LaunchRandomBomb()
            elif key == Keys.K_F1:    # Toggle drawing the menu
                self.settings.drawMenu = not self.settings.drawMenu
            elif key == Keys.K_F2:    # Do a single step
                self.settings.singleStep = True
                if GUIEnabled:
                    self.gui_table.updateGUI(self.settings)
            else:              # Inform the test of the key press
                self.Keyboard(key)
        else:
            self.KeyboardUp(key)

    def CheckKeys(self):
        """
        Check the keys that are evaluated on every main loop iteration.
        I.e., they aren't just evaluated when first pressed down
        """

        pygame.event.pump()
        self.keys = keys = pygame.key.get_pressed()
        if keys[Keys.K_LEFT]:
            self.viewCenter -= (0.5, 0)
        elif keys[Keys.K_RIGHT]:
            self.viewCenter += (0.5, 0)

        if keys[Keys.K_UP]:
            self.viewCenter += (0, 0.5)
        elif keys[Keys.K_DOWN]:
            self.viewCenter -= (0, 0.5)

        if keys[Keys.K_HOME]:
            self.viewZoom = 1.0
            self.viewCenter = (0.0, 20.0)

    def Step(self, settings):
        if GUIEnabled:
            # Update the settings based on the GUI
            self.gui_table.updateSettings(self.settings)

        super(PygameFramework, self).Step(settings)

        if GUIEnabled:
            # In case during the step the settings changed, update the GUI reflecting
            # those settings.
            self.gui_table.updateGUI(self.settings)

    def ConvertScreenToWorld(self, x, y):
        return b2Vec2((x + self.viewOffset.x) / self.viewZoom,
                      ((self.screenSize.y - y + self.viewOffset.y) / self.viewZoom))

    def DrawStringAt(self, x, y, str, color=(229, 153, 153, 255)):
        """
        Draw some text, str, at screen coordinates (x, y).
        """
        self.screen.blit(self.font.render(str, True, color), (x, y))

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

         from framework import Keys
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
