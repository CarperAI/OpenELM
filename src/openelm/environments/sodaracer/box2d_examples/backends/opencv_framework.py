# -*- coding: utf-8 -*-
#
# C++ version Copyright (c) 2006-2007 Erin Catto http://www.box2d.org
# Python version Copyright (c) 2015 John Stowers
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

"""
import string
import time

import cv2
import numpy as np

import framework
from framework import FrameworkBase, fwSettings

from Box2D import b2DrawExtended, b2Vec2

from opencv_draw import cvcolor, cvcoord


class OpencvDraw(framework.b2DrawExtended):
    """
    This debug draw class accepts callbacks from Box2D (which specifies what to
    draw) and handles all of the rendering.

    If you are writing your own game, you likely will not want to use debug
    drawing.  Debug drawing, as its name implies, is for debugging.
    """
    surface = None
    axisScale = 10.0

    def __init__(self, **kwargs):
        b2DrawExtended.__init__(self, **kwargs)
        self.flipX = False
        self.flipY = True
        self.convertVertices = True

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

        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(self.surface, [pts], True, cvcolor(color))

    def DrawSegment(self, p1, p2, color):
        """
        Draw the line segment from p1-p2 with the specified color.
        """
        cv2.line(self.surface, cvcoord(p1), cvcoord(p2), cvcolor(color), 1)

    def DrawTransform(self, xf):
        """
        Draw the transform xf on the screen
        """
        p1 = xf.position
        p2 = self.to_screen(p1 + self.axisScale * xf.R.x_axis)
        p3 = self.to_screen(p1 + self.axisScale * xf.R.y_axis)
        p1 = self.to_screen(p1)
        cv2.line(self.surface, cvcoord(p1), cvcoord(p2), (0, 0, 255), 1)
        cv2.line(self.surface, cvcoord(p1), cvcoord(p3), (0, 255, 0), 1)

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

        cv2.circle(self.surface, cvcoord(center),
                   radius, cvcolor(color), drawwidth)

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

        FILL = False

        cv2.circle(self.surface, cvcoord(center), radius,
                   cvcolor(color), -1 if FILL else 1)

        cv2.line(self.surface, cvcoord(center),
                 cvcoord((center[0] - radius * axis[0],
                          center[1] + radius * axis[1])),
                 (0, 0, 255),
                 1)

    def DrawPolygon(self, vertices, color):
        """
        Draw a wireframe polygon given the screen vertices with the specified
        color.
        """
        if not vertices:
            return

        if len(vertices) == 2:
            cv2.line(self.surface, cvcoord(vertices[0]), cvcoord(
                vertices[1]), cvcolor(color), 1)
        else:
            pts = np.array(vertices, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(self.surface, [pts], True, cvcolor(color))

    def DrawSolidPolygon(self, vertices, color):
        """
        Draw a filled polygon given the screen vertices with the specified color.
        """
        FILL = False

        if not FILL:
            self.DrawPolygon(vertices, color)
            return

        if not vertices:
            return

        if len(vertices) == 2:
            cv2.line(self.surface, cvcoord(vertices[0]), cvcoord(
                vertices[1]), cvcolor(color), 1)
        else:
            pts = np.array(vertices, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(self.surface, [pts], cvcolor(color))

# Only support ascii keys
# The following import is only needed to do the initial loading and
# overwrite the Keys class.


class OpencvKeysType(type):
    def __getattr__(cls, key):
        return getattr(cls, key, None)


class OpencvKeys:
    __metaclass__ = OpencvKeysType


for key in string.ascii_lowercase + string.digits:
    setattr(OpencvKeys, 'K_%c' % key, ord(key))


framework.Keys = OpencvKeys


class OpencvFramework(FrameworkBase):

    def __init__(self, w=640, h=480, resizable=False):
        super(OpencvFramework, self).__init__()

        if fwSettings.onlyInit:  # testing mode doesn't initialize opencv
            return

        self._viewZoom = 10.0
        self._viewCenter = None
        self._viewOffset = None
        self.screenSize = None
        self.fps = 0

        self.screen = np.zeros((h, w, 3), np.uint8)

        if resizable:
            cv2.namedWindow(self.name, getattr(cv2, 'WINDOW_NORMAL', 0))
            cv2.resizeWindow(self.name, w, h)
        else:
            cv2.namedWindow(self.name, getattr(cv2, 'WINDOW_AUTOSIZE', 1))

        cv2.setMouseCallback(self.name, self._on_mouse)

        self._t0 = time.time()

        self.textLine = 30
        self._font_name = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = 0.5
        self._font_thickness = 1

        (_, self._font_h), _ = cv2.getTextSize("X", self._font_name,
                                               self._font_scale,
                                               self._font_thickness)

        self.screenSize = b2Vec2(w, h)

        self.renderer = OpencvDraw(surface=self.screen, test=self)
        self.world.renderer = self.renderer

        self.viewCenter = (0, 20.0)
        self.groundbody = self.world.CreateBody()

    # mouse callback function
    def _on_mouse(self, event, x, y, flags, param):
        p = self.ConvertScreenToWorld(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.MouseDown(p)
        if event == cv2.EVENT_LBUTTONUP:
            self.MouseUp(p)
        elif event == cv2.EVENT_MOUSEMOVE:
            self.MouseMove(p)

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
    viewCenter = property(lambda self: self._viewCenter / self._viewZoom,
                          setCenter,
                          doc='Screen center in camera coordinates')
    viewOffset = property(lambda self: self._viewOffset,
                          doc='The offset of the top-left corner of the screen')

    def run(self):
        """
        Main loop.

        Continues to run while checkEvents indicates the user has requested to
        quit.

        Updates the screen and tells the GUI to paint itself.
        """
        while True:
            self._t1 = time.time()

            dt = 1.0 / self.settings.hz
            key = 0xFF & cv2.waitKey(int(dt * 1000.0))
            if key == 27:
                break
            elif key != 255:
                if key == 32:               # Space
                    self.LaunchRandomBomb()
                elif key == 81:             # Left
                    self.viewCenter -= (0.5, 0)
                elif key == 83:             # Right
                    self.viewCenter += (0.5, 0)
                elif key == 82:             # Up
                    self.viewCenter += (0, 0.5)
                elif key == 84:             # Down
                    self.viewCenter -= (0, 0.5)
                elif key == 80:             # Home
                    self.viewZoom = 1.0
                    self.viewCenter = (0.0, 20.0)
                elif key == ord('z'):       # Zoom in
                    self.viewZoom = min(1.1 * self.viewZoom, 50.0)
                elif key == ord('x'):       # Zoom out
                    self.viewZoom = max(0.9 * self.viewZoom, 0.02)
                else:
                    self.Keyboard(key)

            self.screen.fill(0)
            self.SimulationLoop()
            cv2.imshow(self.name, self.screen)

            dt = self._t1 - self._t0
            self._t0 = self._t1
            self.fps = 1 / dt

        self.world.contactListener = None
        self.world.destructionListener = None
        self.world.renderer = None

    def ConvertScreenToWorld(self, x, y):
        x = (x + self.viewOffset.x) / self.viewZoom
        y = ((self.screenSize.y - y + self.viewOffset.y) / self.viewZoom)
        return b2Vec2(x, y)

    def DrawStringAt(self, x, y, str, color=(229, 153, 153, 255)):
        """
        Draw some text, str, at screen coordinates (x, y).
        """
        color = (color[2], color[1], color[0])
        cv2.putText(self.screen, str, (x, y),
                    self._font_name, self._font_scale, color,
                    self._font_thickness)

    def Print(self, str, color=(229, 153, 153, 255)):
        """
        Draw some text at the top status lines
        and advance to the next line.
        """
        color = (color[2], color[1], color[0])
        cv2.putText(self.screen, str, (5, self.textLine),
                    self._font_name, self._font_scale, color,
                    self._font_thickness)
        self.textLine += self._font_h + 2

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
