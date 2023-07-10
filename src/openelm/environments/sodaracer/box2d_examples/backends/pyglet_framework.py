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
    Space  - shoot projectile
    Z/X    - zoom
    Escape - quit

Other keys can be set by the individual test.

Mouse:
    Left click  - select/drag body (creates mouse joint)
    Right click - pan
    Shift+Left  - drag to create a directional projectile
    Scroll      - zoom

You can easily add your own tests based on test_empty.
"""
import string
import math

import pyglet
from pyglet import gl

from Box2D import (b2Vec2, b2Draw)
from ..framework import (FrameworkBase, Keys)
from ..settings import fwSettings


class grBlended (pyglet.graphics.Group):
    """
    This pyglet rendering group enables blending.
    """

    def set_state(self):
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def unset_state(self):
        gl.glDisable(gl.GL_BLEND)


class grPointSize (pyglet.graphics.Group):
    """
    This pyglet rendering group sets a specific point size.
    """

    def __init__(self, size=4.0):
        super(grPointSize, self).__init__()
        self.size = size

    def set_state(self):
        gl.glPointSize(self.size)

    def unset_state(self):
        gl.glPointSize(1.0)


class grText(pyglet.graphics.Group):
    """
    This pyglet rendering group sets the proper projection for
    displaying text when used.
    """
    window = None

    def __init__(self, window=None):
        super(grText, self).__init__()
        self.window = window

    def set_state(self):
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        gl.gluOrtho2D(0, self.window.width, 0, self.window.height)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()
        gl.glLoadIdentity()

    def unset_state(self):
        gl.glPopMatrix()
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPopMatrix()
        gl.glMatrixMode(gl.GL_MODELVIEW)


class PygletDraw(b2Draw):
    """
    This debug draw class accepts callbacks from Box2D (which specifies what to draw)
    and handles all of the rendering.

    If you are writing your own game, you likely will not want to use debug drawing.
    Debug drawing, as its name implies, is for debugging.
    """
    blended = grBlended()
    circle_segments = 16
    surface = None
    circle_cache_tf = {}  # triangle fan (inside)
    circle_cache_ll = {}  # line loop (border)

    def __init__(self, test):
        super(PygletDraw, self).__init__()
        self.test = test

    def StartDraw(self):
        pass

    def EndDraw(self):
        pass

    def triangle_fan(self, vertices):
        """
        in: vertices arranged for gl_triangle_fan ((x,y),(x,y)...)
        out: vertices arranged for gl_triangles (x,y,x,y,x,y...)
        """
        out = []
        for i in range(1, len(vertices) - 1):
            # 0,1,2   0,2,3  0,3,4 ..
            out.extend(vertices[0])
            out.extend(vertices[i])
            out.extend(vertices[i + 1])
        return len(out) // 2, out

    def line_loop(self, vertices):
        """
        in: vertices arranged for gl_line_loop ((x,y),(x,y)...)
        out: vertices arranged for gl_lines (x,y,x,y,x,y...)
        """
        out = []
        for i in range(len(vertices) - 1):
            # 0,1  1,2  2,3 ... len-1,len  len,0
            out.extend(vertices[i])
            out.extend(vertices[i + 1])

        out.extend(vertices[len(vertices) - 1])
        out.extend(vertices[0])

        return len(out) // 2, out

    def _getLLCircleVertices(self, radius, points):
        """
        Get the line loop-style vertices for a given circle.
        Drawn as lines.

        "Line Loop" is used as that's how the C++ code draws the
        vertices, with lines going around the circumference of the
        circle (GL_LINE_LOOP).

        This returns 'points' amount of lines approximating the
        border of a circle.

        (x1, y1, x2, y2, x3, y3, ...)
        """
        ret = []
        step = 2 * math.pi / points
        n = 0
        for i in range(points):
            ret.append((math.cos(n) * radius, math.sin(n) * radius))
            n += step
            ret.append((math.cos(n) * radius, math.sin(n) * radius))
        return ret

    def _getTFCircleVertices(self, radius, points):
        """
        Get the triangle fan-style vertices for a given circle.
        Drawn as triangles.

        "Triangle Fan" is used as that's how the C++ code draws the
        vertices, with triangles originating at the center of the
        circle, extending around to approximate a filled circle
        (GL_TRIANGLE_FAN).

        This returns 'points' amount of lines approximating the
        circle.

        (a1, b1, c1, a2, b2, c2, ...)
        """
        ret = []
        step = 2 * math.pi / points
        n = 0
        for i in range(points):
            ret.append((0.0, 0.0))
            ret.append((math.cos(n) * radius, math.sin(n) * radius))
            n += step
            ret.append((math.cos(n) * radius, math.sin(n) * radius))
        return ret

    def getCircleVertices(self, center, radius, points):
        """
        Returns the triangles that approximate the circle and
        the lines that border the circles edges, given
        (center, radius, points).

        Caches the calculated LL/TF vertices, but recalculates
        based on the center passed in.

        TODO: Currently, there's only one point amount,
        so the circle cache ignores it when storing. Could cause
        some confusion if you're using multiple point counts as
        only the first stored point-count for that radius will
        show up.
        TODO: What does the previous TODO mean?

        Returns: (tf_vertices, ll_vertices)
        """
        if radius not in self.circle_cache_tf:
            self.circle_cache_tf[
                radius] = self._getTFCircleVertices(radius, points)
            self.circle_cache_ll[
                radius] = self._getLLCircleVertices(radius, points)

        ret_tf, ret_ll = [], []

        for x, y in self.circle_cache_tf[radius]:
            ret_tf.extend((x + center[0], y + center[1]))
        for x, y in self.circle_cache_ll[radius]:
            ret_ll.extend((x + center[0], y + center[1]))
        return ret_tf, ret_ll

    def DrawCircle(self, center, radius, color):
        """
        Draw an unfilled circle given center, radius and color.
        """
        unused, ll_vertices = self.getCircleVertices(
            center, radius, self.circle_segments)
        ll_count = len(ll_vertices) // 2

        self.batch.add(ll_count, gl.GL_LINES, None,
                       ('v2f', ll_vertices),
                       ('c4f', [color.r, color.g, color.b, 1.0] * ll_count))

    def DrawSolidCircle(self, center, radius, axis, color):
        """
        Draw an filled circle given center, radius, axis (of orientation) and color.
        """
        tf_vertices, ll_vertices = self.getCircleVertices(
            center, radius, self.circle_segments)
        tf_count, ll_count = len(tf_vertices) // 2, len(ll_vertices) // 2

        self.batch.add(tf_count, gl.GL_TRIANGLES, self.blended,
                       ('v2f', tf_vertices),
                       ('c4f', [0.5 * color.r, 0.5 * color.g, 0.5 * color.b, 0.5] * tf_count))

        self.batch.add(ll_count, gl.GL_LINES, None,
                       ('v2f', ll_vertices),
                       ('c4f', [color.r, color.g, color.b, 1.0] * (ll_count)))

        p = b2Vec2(center) + radius * b2Vec2(axis)
        self.batch.add(2, gl.GL_LINES, None,
                       ('v2f', (center[0], center[1], p[0], p[1])),
                       ('c3f', [1.0, 0.0, 0.0] * 2))

    def DrawPolygon(self, vertices, color):
        """
        Draw a wireframe polygon given the world vertices (tuples) with the specified color.
        """
        if len(vertices) == 2:
            p1, p2 = vertices
            self.batch.add(2, gl.GL_LINES, None,
                           ('v2f', (p1[0], p1[1], p2[0], p2[1])),
                           ('c3f', [color.r, color.g, color.b] * 2))
        else:
            ll_count, ll_vertices = self.line_loop(vertices)

            self.batch.add(ll_count, gl.GL_LINES, None,
                           ('v2f', ll_vertices),
                           ('c4f', [color.r, color.g, color.b, 1.0] * (ll_count)))

    def DrawSolidPolygon(self, vertices, color):
        """
        Draw a filled polygon given the world vertices (tuples) with the specified color.
        """
        if len(vertices) == 2:
            p1, p2 = vertices
            self.batch.add(2, gl.GL_LINES, None,
                           ('v2f', (p1[0], p1[1], p2[0], p2[1])),
                           ('c3f', [color.r, color.g, color.b] * 2))
        else:
            tf_count, tf_vertices = self.triangle_fan(vertices)
            if tf_count == 0:
                return

            self.batch.add(tf_count, gl.GL_TRIANGLES, self.blended,
                           ('v2f', tf_vertices),
                           ('c4f', [0.5 * color.r, 0.5 * color.g, 0.5 * color.b, 0.5] * (tf_count)))

            ll_count, ll_vertices = self.line_loop(vertices)

            self.batch.add(ll_count, gl.GL_LINES, None,
                           ('v2f', ll_vertices),
                           ('c4f', [color.r, color.g, color.b, 1.0] * ll_count))

    def DrawSegment(self, p1, p2, color):
        """
        Draw the line segment from p1-p2 with the specified color.
        """
        self.batch.add(2, gl.GL_LINES, None,
                       ('v2f', (p1[0], p1[1], p2[0], p2[1])),
                       ('c3f', [color.r, color.g, color.b] * 2))

    def DrawXForm(self, xf):
        """
        Draw the transform xf on the screen
        """
        p1 = xf.position
        k_axisScale = 0.4
        p2 = p1 + k_axisScale * xf.R.x_axis
        p3 = p1 + k_axisScale * xf.R.y_axis

        self.batch.add(3, gl.GL_LINES, None,
                       ('v2f', (p1[0], p1[1], p2[0], p2[
                        1], p1[0], p1[1], p3[0], p3[1])),
                       ('c3f', [1.0, 0.0, 0.0] * 2 + [0.0, 1.0, 0.0] * 2))

    def DrawPoint(self, p, size, color):
        """
        Draw a single point at point p given a point size and color.
        """
        self.batch.add(1, gl.GL_POINTS, grPointSize(size),
                       ('v2f', (p[0], p[1])),
                       ('c3f', [color.r, color.g, color.b]))

    def DrawAABB(self, aabb, color):
        """
        Draw a wireframe around the AABB with the given color.
        """
        self.renderer.batch.add(8, gl.GL_LINES, None,
                                ('v2f', (aabb.lowerBound.x, aabb.lowerBound.y,
                                         aabb.upperBound.x, aabb.lowerBound.y,
                                         aabb.upperBound.x, aabb.lowerBound.y,
                                         aabb.upperBound.x, aabb.upperBound.y,
                                         aabb.upperBound.x, aabb.upperBound.y,
                                         aabb.lowerBound.x, aabb.upperBound.y,
                                         aabb.lowerBound.x, aabb.upperBound.y,
                                         aabb.lowerBound.x, aabb.lowerBound.y)),
                                ('c3f', [color.r, color.g, color.b] * 8))

    def to_screen(self, point):
        """
        In here for compatibility with other frameworks.
        """
        return tuple(point)


class PygletWindow(pyglet.window.Window):

    def __init__(self, test):
        super(PygletWindow, self).__init__()
        self.test = test

    def on_close(self):
        """
        Callback: user tried to close the window
        """
        pyglet.clock.unschedule(self.test.SimulationLoop)
        super(PygletWindow, self).on_close()

    def on_show(self):
        """
        Callback: the window was shown.
        """
        self.test.updateProjection()

    def on_key_press(self, key, modifiers):
        self.test._Keyboard_Event(key, down=True)

    def on_key_release(self, key, modifiers):
        self.test._Keyboard_Event(key, down=False)

    def on_mouse_press(self, x, y, button, modifiers):
        p = self.test.ConvertScreenToWorld(x, y)
        self.test.mouseWorld = p
        if button == pyglet.window.mouse.LEFT:
            if modifiers & pyglet.window.key.MOD_SHIFT:
                self.test.ShiftMouseDown(p)
            else:
                self.test.MouseDown(p)
        elif button == pyglet.window.mouse.MIDDLE:
            pass

    def on_mouse_release(self, x, y, button, modifiers):
        """
        Mouse up
        """
        p = self.test.ConvertScreenToWorld(x, y)
        self.test.mouseWorld = p

        if button == pyglet.window.mouse.LEFT:
            self.test.MouseUp(p)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        """
        Mouse scrollwheel used
        """
        if scroll_y < 0:
            self.test.viewZoom *= 1.1
        elif scroll_y > 0:
            self.test.viewZoom /= 1.1

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """
        Mouse moved while clicking
        """
        p = self.test.ConvertScreenToWorld(x, y)
        self.test.mouseWorld = p

        self.test.MouseMove(p)

        if buttons & pyglet.window.mouse.RIGHT:
            self.test.viewCenter -= (float(dx) / 5, float(dy) / 5)


class PygletFramework(FrameworkBase):

    def setup_keys(self):
        key = pyglet.window.key
        self.keys = key.KeyStateHandler()
        # Only basic keys are mapped for now: K_[a-z0-9], K_F[1-12] and
        # K_COMMA.

        if hasattr(string, 'ascii_uppercase'):
            uppercase = string.ascii_uppercase
        else:
            uppercase = string.uppercase

        for letter in uppercase:
            setattr(Keys, 'K_' + letter.lower(), getattr(key, letter))
        for i in range(10):
            setattr(Keys, 'K_%d' % i, getattr(key, '_%d' % i))
        for i in range(1, 13):
            setattr(Keys, 'K_F%d' % i, getattr(key, 'F%d' % i))
        Keys.K_LEFT = key.LEFT
        Keys.K_RIGHT = key.RIGHT
        Keys.K_UP = key.UP
        Keys.K_DOWN = key.DOWN
        Keys.K_HOME = key.HOME
        Keys.K_PAGEUP = key.PAGEUP
        Keys.K_PAGEDOWN = key.PAGEDOWN
        Keys.K_COMMA = key.COMMA

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

        # Window-related
        self.fontname = "Arial"
        self.fontsize = 10
        self.font = None
        self.textGroup = None

        # Screen-related
        self._viewZoom = 1.0
        self._viewCenter = None
        self.screenSize = None
        self.textLine = 30
        self.font = None
        self.fps = 0

        self.setup_keys()

    def __init__(self):
        super(PygletFramework, self).__init__()

        self.__reset()

        if fwSettings.onlyInit:  # testing mode doesn't initialize Pyglet
            return

        print('Initializing Pyglet framework...')
        self.window = PygletWindow(self)

        # Initialize the text display group
        self.textGroup = grText(self.window)

        # Load the font and record the screen dimensions
        self.font = pyglet.font.load(self.fontname, self.fontsize)
        self.screenSize = b2Vec2(self.window.width, self.window.height)

        self.renderer = PygletDraw(self)
        self.renderer.surface = self.window.screen
        self.world.renderer = self.renderer
        self._viewCenter = b2Vec2(0, 10.0)
        self.groundbody = self.world.CreateBody()

    def setCenter(self, value):
        """
        Updates the view offset based on the center of the screen.

        Tells the debug draw to update its values also.
        """
        self._viewCenter = b2Vec2(*value)
        self.updateProjection()

    def setZoom(self, zoom):
        self._viewZoom = zoom
        self.updateProjection()

    viewZoom = property(lambda self: self._viewZoom, setZoom,
                        doc='Zoom factor for the display')
    viewCenter = property(lambda self: self._viewCenter, setCenter,
                          doc='Screen center in camera coordinates')

    def updateProjection(self):
        """
        Recalculates the necessary projection.
        """
        gl.glViewport(0, 0, self.window.width, self.window.height)

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()

        ratio = float(self.window.width) / self.window.height

        extents = b2Vec2(ratio * 25.0, 25.0)
        extents *= self._viewZoom

        lower = self._viewCenter - extents
        upper = self._viewCenter + extents

        # L/R/B/T
        gl.gluOrtho2D(lower.x, upper.x, lower.y, upper.y)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

    def run(self):
        """
        Main loop.
        """
        if self.settings.hz > 0.0:
            pyglet.clock.schedule_interval(
                self.SimulationLoop, 1.0 / self.settings.hz)

        # self.window.push_handlers(pyglet.window.event.WindowEventLogger())
        # TODO: figure out why this is required
        self.window._enable_event_queue = False
        pyglet.app.run()

        self.world.contactListener = None
        self.world.destructionListener = None
        self.world.renderer = None

    def SimulationLoop(self, dt):
        """
        The main simulation loop. Don't override this, override Step instead.
        And be sure to call super(classname, self).Step(settings) at the end
        of your Step function.
        """

        # Check the input and clear the screen
        self.CheckKeys()
        self.window.clear()

        # Update the keyboard status
        self.window.push_handlers(self.keys)

        # Create a new batch for drawing
        self.renderer.batch = pyglet.graphics.Batch()

        # Reset the text position
        self.textLine = 15

        # Draw the title of the test at the top
        self.Print(self.name)

        # Step the physics
        self.Step(self.settings)
        self.renderer.batch.draw()
        self.window.invalid = True

        self.fps = pyglet.clock.get_fps()

    def _Keyboard_Event(self, key, down=True):
        """
        Internal keyboard event, don't override this.

        Checks for the initial keydown of the basic testbed keys. Passes the unused
        ones onto the test via the Keyboard() function.
        """
        if down:
            if key == pyglet.window.key.ESCAPE:
                exit(0)
            elif key == pyglet.window.key.SPACE:
                # Launch a bomb
                self.LaunchRandomBomb()
            elif key == Keys.K_z:
                # Zoom in
                self.viewZoom = min(1.1 * self.viewZoom, 20.0)
            elif key == Keys.K_x:
                # Zoom out
                self.viewZoom = max(0.9 * self.viewZoom, 0.02)
            else:
                # Inform the test of the key press
                self.Keyboard(key)
        else:
            self.KeyboardUp(key)

    def CheckKeys(self):
        """
        Check the keys that are evaluated on every main loop iteration.
        I.e., they aren't just evaluated when first pressed down
        """
        keys = self.keys
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

    # def Step(self, settings):
    #    super(PygletFramework, self).Step(settings)

    def ConvertScreenToWorld(self, x, y):
        """
        Takes screen (x, y) and returns
        world coordinate b2Vec2(x,y).
        """
        u = float(x) / self.window.width
        v = float(y) / self.window.height

        ratio = float(self.window.width) / self.window.height
        extents = b2Vec2(ratio * 25.0, 25.0)
        extents *= self._viewZoom

        lower = self._viewCenter - extents
        upper = self._viewCenter + extents

        p = b2Vec2(
            (1.0 - u) * lower.x + u * upper.x,
            (1.0 - v) * lower.y + v * upper.y)

        return p

    def DrawStringAt(self, x, y, str, color=(229, 153, 153, 255)):
        """
        Draw some text, str, at screen coordinates (x, y).
        """
        pyglet.text.Label(str, font_name=self.fontname,
                          font_size=self.fontsize, x=x, y=self.window.height - y,
                          color=color, batch=self.renderer.batch, group=self.textGroup)

    def Print(self, str, color=(229, 153, 153, 255)):
        """
        Draw some text, str, at screen coordinates (x, y).
        """
        pyglet.text.Label(str, font_name=self.fontname,
                          font_size=self.fontsize, x=5, y=self.window.height -
                          self.textLine, color=color, batch=self.renderer.batch,
                          group=self.textGroup)
        self.textLine += 15

    def Keyboard(self, key):
        """
        Callback indicating 'key' has been pressed down.
        """
        pass

    def KeyboardUp(self, key):
        """
        Callback indicating 'key' has been released.
        See Keyboard() for key information
        """
        pass
