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
    Right click - get body/fixture information
    Shift+Left  - drag to create a directed projectile
    Scroll      - zoom

"""
import string
import sys
import re

from PyQt4 import (QtGui, QtCore)
from PyQt4.QtGui import (QTableWidgetItem, QColor)
from PyQt4.QtCore import Qt

from Box2D import (b2AABB, b2CircleShape, b2Color, b2DistanceJoint,
                   b2EdgeShape, b2LoopShape, b2MouseJoint, b2Mul,
                   b2PolygonShape, b2PulleyJoint, b2Vec2)
from Box2D import (b2_pi, b2_staticBody, b2_kinematicBody)

from ..framework import (fwQueryCallback, FrameworkBase, Keys)
from .. import settings
from .pyqt4_gui import Ui_MainWindow


class Pyqt4Draw(object):
    """
    This debug drawing class differs from the other frameworks.  It provides an
    example of how to iterate through all the objects in the world and
    associate (in PyQt4's case) QGraphicsItems with them.

    While DrawPolygon and DrawSolidPolygon are not used for the core shapes in
    the world (DrawPolygonShape is), they are left in for compatibility with
    other frameworks and the tests.

    world_coordinate parameters are also left in for compatibility.  Screen
    coordinates cannot be used, as PyQt4 does the scaling and rotating for us.

    If you utilize this framework and need to add more items to the
    QGraphicsScene for a single step, be sure to add them to the temp_items
    array to be deleted on the next draw.
    """
    MAX_TIMES = 20
    axisScale = 0.4

    def __init__(self, test):
        self.test = test
        self.window = self.test.window
        self.scene = self.window.scene
        self.view = self.window.graphicsView
        self.item_cache = {}
        self.temp_items = []
        self.status_font = QtGui.QFont("Times", 10, QtGui.QFont.Bold)
        self.font_spacing = QtGui.QFontMetrics(self.status_font).lineSpacing()
        self.draw_idx = 0

    def StartDraw(self):
        for item in self.temp_items:
            self.scene.removeItem(item)
        self.temp_items = []

    def EndDraw(self):
        pass

    def SetFlags(self, **kwargs):
        """
        For compatibility with other debug drawing classes.
        """
        pass

    def DrawStringAt(self, x, y, str, color=None):
        item = QtGui.QGraphicsSimpleTextItem(str)
        if color is None:
            color = (255, 255, 255, 255)

        brush = QtGui.QBrush(QColor(255, 255, 255, 255))
        item.setFont(self.status_font)
        item.setBrush(brush)
        item.setPos(self.view.mapToScene(x, y))
        item.scale(1. / self.test._viewZoom, -1. / self.test._viewZoom)
        self.temp_items.append(item)

        self.scene.addItem(item)

    def DrawPoint(self, p, size, color):
        """
        Draw a single point at point p given a pixel size and color.
        """
        self.DrawCircle(p, size / self.test.viewZoom, color, drawwidth=0)

    def DrawAABB(self, aabb, color):
        """
        Draw a wireframe around the AABB with the given color.
        """
        line1 = self.scene.addLine(aabb.lowerBound.x, aabb.lowerBound.y,
                                   aabb.upperBound.x, aabb.lowerBound.y,
                                   pen=QtGui.QPen(QColor(*color.bytes)))
        line2 = self.scene.addLine(aabb.upperBound.x, aabb.upperBound.y,
                                   aabb.lowerBound.x, aabb.upperBound.y,
                                   pen=QtGui.QPen(QColor(*color.bytes)))
        self.temp_items.append(line1)
        self.temp_items.append(line2)

    def DrawSegment(self, p1, p2, color):
        """
        Draw the line segment from p1-p2 with the specified color.
        """
        line = self.scene.addLine(p1[0], p1[1], p2[0], p2[1],
                                  pen=QtGui.QPen(QColor(*color.bytes)))
        self.temp_items.append(line)

    def DrawTransform(self, xf):
        """
        Draw the transform xf on the screen
        """
        p1 = xf.position
        p2 = p1 + self.axisScale * xf.R.x_axis
        p3 = p1 + self.axisScale * xf.R.y_axis

        line1 = self.scene.addLine(p1[0], p1[1], p2[0], p2[1],
                                   pen=QtGui.QPen(QColor(255, 0, 0)))
        line2 = self.scene.addLine(p1[0], p1[1], p3[0], p3[1],
                                   pen=QtGui.QPen(QColor(0, 255, 0)))
        self.temp_items.append(line1)
        self.temp_items.append(line2)

    def DrawCircle(self, center, radius, color, drawwidth=1, shape=None):
        """
        Draw a wireframe circle given the center, radius, axis of orientation
        and color.
        """
        border_color = [c * 255 for c in color] + [255]
        pen = QtGui.QPen(QtGui.QColor(*border_color))
        ellipse = self.scene.addEllipse(center[0] - radius, center[1] - radius,
                                        radius * 2, radius * 2, pen=pen)
        self.temp_items.append(ellipse)

    def DrawSolidCircle(self, center, radius, axis, color, shape=None):
        """
        Draw a solid circle given the center, radius, axis of orientation and
        color.
        """
        border_color = color.bytes + [255]
        inside_color = (color / 2).bytes + [127]
        brush = QtGui.QBrush(QtGui.QColor(*inside_color))
        pen = QtGui.QPen(QtGui.QColor(*border_color))
        ellipse = self.scene.addEllipse(center[0] - radius, center[1] - radius,
                                        radius * 2, radius * 2, brush=brush,
                                        pen=pen)
        line = self.scene.addLine(center[0], center[1],
                                  (center[0] - radius * axis[0]),
                                  (center[1] - radius * axis[1]),
                                  pen=QtGui.QPen(QColor(255, 0, 0)))

        self.temp_items.append(ellipse)
        self.temp_items.append(line)

    def DrawPolygon(self, vertices, color, shape=None):
        """
        Draw a wireframe polygon given the world vertices vertices (tuples)
        with the specified color.
        """
        poly = QtGui.QPolygonF()
        pen = QtGui.QPen(QtGui.QColor(*color.bytes))

        for v in vertices:
            poly += QtCore.QPointF(*v)

        item = self.scene.addPolygon(poly, pen=pen)
        self.temp_items.append(item)

    def DrawSolidPolygon(self, vertices, color, shape=None):
        """
        Draw a filled polygon given the world vertices vertices (tuples) with
        the specified color.
        """
        poly = QtGui.QPolygonF()
        border_color = color.bytes + [255]
        inside_color = (color / 2).bytes + [127]
        brush = QtGui.QBrush(QtGui.QColor(*inside_color))
        pen = QtGui.QPen(QtGui.QColor(*border_color))

        for v in vertices:
            poly += QtCore.QPointF(*v)

        item = self.scene.addPolygon(poly, brush=brush, pen=pen)
        self.temp_items.append(item)

    def DrawCircleShape(self, shape, transform, color, temporary=False):
        center = b2Mul(transform, shape.pos)
        radius = shape.radius
        axis = transform.R.x_axis

        border_color = color.bytes + [255]
        inside_color = (color / 2).bytes + [127]
        brush = QtGui.QBrush(QtGui.QColor(*inside_color))
        pen = QtGui.QPen(QtGui.QColor(*border_color))
        ellipse = self.scene.addEllipse(-radius, -radius,
                                        radius * 2, radius * 2, brush=brush,
                                        pen=pen)
        line = self.scene.addLine(center[0], center[1],
                                  (center[0] - radius * axis[0]),
                                  (center[1] - radius * axis[1]),
                                  pen=QtGui.QPen(QColor(255, 0, 0)))
        ellipse.setPos(*center)
        ellipse.radius = radius

        if temporary:
            self.temp_items.append(ellipse)
            self.temp_items.append(line)
        else:
            self.item_cache[hash(shape)] = [ellipse, line]

    def DrawPolygonShape(self, shape, transform, color, temporary=False):
        poly = QtGui.QPolygonF()
        border_color = color.bytes + [255]
        inside_color = (color / 2).bytes + [127]
        brush = QtGui.QBrush(QtGui.QColor(*inside_color))
        pen = QtGui.QPen(QtGui.QColor(*border_color))

        for v in shape.vertices:
            poly += QtCore.QPointF(*v)

        item = self.scene.addPolygon(poly, brush=brush, pen=pen)
        item.setRotation(transform.angle * 180.0 / b2_pi)
        item.setPos(*transform.position)

        if temporary:
            self.temp_items.append(item)
        else:
            self.item_cache[hash(shape)] = [item]

    def _remove_from_cache(self, shape):
        items = self.item_cache[hash(shape)]

        del self.item_cache[hash(shape)]
        for item in items:
            self.scene.removeItem(item)

    def DrawShape(self, shape, transform, color, selected=False):
        """
        Draw any type of shape
        """
        cache_hit = False
        if hash(shape) in self.item_cache:
            cache_hit = True
            items = self.item_cache[hash(shape)]
            items[0].setRotation(transform.angle * 180.0 / b2_pi)
            if isinstance(shape, b2CircleShape):
                radius = shape.radius
                if items[0].radius == radius:
                    center = b2Mul(transform, shape.pos)
                    items[0].setPos(*center)
                    line = items[1]
                    axis = transform.R.x_axis
                    line.setLine(center[0], center[1],
                                 (center[0] - radius * axis[0]),
                                 (center[1] - radius * axis[1]))
                else:
                    self._remove_from_cache(shape)
                    cache_hit = False
            else:
                items[0].setPos(*transform.position)

            if not selected or cache_hit:
                return

        if selected:
            color = b2Color(1, 1, 1)
            temporary = True
        else:
            temporary = False

        if isinstance(shape, b2PolygonShape):
            self.DrawPolygonShape(shape, transform, color, temporary)
        elif isinstance(shape, b2EdgeShape):
            v1 = b2Mul(transform, shape.vertex1)
            v2 = b2Mul(transform, shape.vertex2)
            self.DrawSegment(v1, v2, color)
        elif isinstance(shape, b2CircleShape):
            self.DrawCircleShape(shape, transform, color, temporary)
        elif isinstance(shape, b2LoopShape):
            vertices = shape.vertices
            v1 = b2Mul(transform, vertices[-1])
            for v2 in vertices:
                v2 = b2Mul(transform, v2)
                self.DrawSegment(v1, v2, color)
                v1 = v2

    def DrawJoint(self, joint):
        """
        Draw any type of joint
        """
        bodyA, bodyB = joint.bodyA, joint.bodyB
        xf1, xf2 = bodyA.transform, bodyB.transform
        x1, x2 = xf1.position, xf2.position
        p1, p2 = joint.anchorA, joint.anchorB
        color = b2Color(0.5, 0.8, 0.8)

        if isinstance(joint, b2DistanceJoint):
            self.DrawSegment(p1, p2, color)
        elif isinstance(joint, b2PulleyJoint):
            s1, s2 = joint.groundAnchorA, joint.groundAnchorB
            self.DrawSegment(s1, p1, color)
            self.DrawSegment(s2, p2, color)
            self.DrawSegment(s1, s2, color)

        elif isinstance(joint, b2MouseJoint):
            pass  # don't draw it here
        else:
            self.DrawSegment(x1, p1, color)
            self.DrawSegment(p1, p2, color)
            self.DrawSegment(x2, p2, color)

    def ManualDraw(self):
        """
        This implements code normally present in the C++ version, which calls
        the callbacks that you see in this class (DrawSegment, DrawSolidCircle,
        etc.).

        This is implemented in Python as an example of how to do it, and also a
        test.
        """
        colors = {
            'active': b2Color(0.5, 0.5, 0.3),
            'static': b2Color(0.5, 0.9, 0.5),
            'kinematic': b2Color(0.5, 0.5, 0.9),
            'asleep': b2Color(0.6, 0.6, 0.6),
            'default': b2Color(0.9, 0.7, 0.7),
        }

        settings = self.test.settings
        world = self.test.world
        if self.test.selected_shapebody:
            sel_shape, sel_body = self.test.selected_shapebody
        else:
            sel_shape = None

        if settings.drawShapes:
            for body in world.bodies:
                transform = body.transform
                for fixture in body.fixtures:
                    shape = fixture.shape

                    if not body.active:
                        color = colors['active']
                    elif body.type == b2_staticBody:
                        color = colors['static']
                    elif body.type == b2_kinematicBody:
                        color = colors['kinematic']
                    elif not body.awake:
                        color = colors['asleep']
                    else:
                        color = colors['default']

                    self.DrawShape(fixture.shape, transform,
                                   color, (sel_shape == shape))

        if settings.drawJoints:
            for joint in world.joints:
                self.DrawJoint(joint)

        # if settings.drawPairs
        #   pass

        if settings.drawAABBs:
            color = b2Color(0.9, 0.3, 0.9)
            # cm = world.contactManager
            for body in world.bodies:
                if not body.active:
                    continue
                transform = body.transform
                for fixture in body.fixtures:
                    shape = fixture.shape
                    for childIndex in range(shape.childCount):
                        self.DrawAABB(shape.getAABB(
                            transform, childIndex), color)

    def to_screen(self, point):
        """
        In here for compatibility with other frameworks.
        """
        return tuple(point)


class GraphicsScene (QtGui.QGraphicsScene):

    def __init__(self, test, parent=None):
        super(GraphicsScene, self).__init__(parent)
        self.test = test

    def keyPressEvent(self, event):
        self.test._Keyboard_Event(event.key(), down=True)

    def keyReleaseEvent(self, event):
        self.test._Keyboard_Event(event.key(), down=False)

    def mousePressEvent(self, event):
        pos = self.test.ConvertScreenToWorld(
            event.scenePos().x(), event.scenePos().y())
        if event.button() == Qt.RightButton:
            self.test.ShowProperties(pos)
        elif event.button() == Qt.LeftButton:
            if event.modifiers() == Qt.ShiftModifier:
                self.test.ShiftMouseDown(pos)
            else:
                self.test.MouseDown(pos)

    def mouseReleaseEvent(self, event):
        pos = event.scenePos().x(), event.scenePos().y()
        if event.button() == Qt.RightButton:
            self.test.MouseUp(pos)
        elif event.button() == Qt.LeftButton:
            self.test.MouseUp(pos)

    def mouseMoveEvent(self, event):
        pos = event.scenePos().x(), event.scenePos().y()
        self.test.MouseMove(self.test.ConvertScreenToWorld(*pos))
        QtGui.QGraphicsScene.mouseMoveEvent(self, event)


class MainWindow(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self, test, parent=None):
        QtGui.QMainWindow.__init__(self)
        self.setupUi(self)
        self.scene = GraphicsScene(test)
        self.test = test
        self.scene.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0)))
        self.graphicsView.setScene(self.scene)
        self.graphicsView.scale(self.test.viewZoom, -self.test.viewZoom)
        self.reset_properties_list()
        self.restoreLayout()

        def increase_font_size(amount=1.0):
            self.setFontSize(app.font().pointSize() + amount)

        def decrease_font_size(amount=1.0):
            self.setFontSize(app.font().pointSize() - amount)

        self.mnuExit.triggered.connect(self.close)
        self.mnuIncreaseFontSize.triggered.connect(increase_font_size)
        self.mnuDecreaseFontSize.triggered.connect(decrease_font_size)
        self.add_settings_widgets()

    def add_settings_widgets(self):
        self.settings_widgets = {}

        gb = self.gbOptions  # the options groupbox
        layout = QtGui.QVBoxLayout()
        gb.setLayout(layout)

        for text, variable in settings.checkboxes:
            if variable:
                widget = QtGui.QCheckBox('&' + text)

                def state_changed(value, variable=variable, widget=widget):
                    setattr(self.test.settings, variable, widget.isChecked())

                widget.stateChanged.connect(state_changed)
                widget.setChecked(getattr(self.test.settings, variable))
                self.settings_widgets[variable] = widget
            else:
                widget = QtGui.QLabel(text)
                widget.setAlignment(Qt.AlignHCenter)

            layout.addWidget(widget)

        for slider in settings.sliders:
            label = QtGui.QLabel(slider['text'])
            label.setAlignment(Qt.AlignHCenter)
            layout.addWidget(label)

            widget = QtGui.QScrollBar(Qt.Horizontal)
            widget.setRange(slider['min'], slider['max'])
            var = slider['name']

            def value_changed(value, slider=slider, label=label):
                variable = slider['name']
                text = slider['text']
                setattr(self.test.settings, variable, value)
                label.setText('%s (%d)' % (text, value))

            widget.valueChanged.connect(value_changed)
            self.settings_widgets[var] = widget
            layout.addWidget(widget)

        self.update_widgets_from_settings()

    def update_widgets_from_settings(self, step_settings=None):
        if step_settings is None:
            step_settings = self.test.settings

        for var, widget in list(self.settings_widgets.items()):
            if isinstance(widget, QtGui.QCheckBox):
                widget.setChecked(getattr(step_settings, var))
            else:
                widget.setValue(getattr(step_settings, var))

        for slider in settings.sliders:
            var = slider['name']
            self.settings_widgets[var].setValue(getattr(step_settings, var))

    def reset_properties_list(self):
        self.twProperties.clear()
        self.twProperties.setRowCount(0)
        self.twProperties.setColumnCount(3)
        self.twProperties.verticalHeader().hide()  # don't show numbers on left
        self.twProperties.setHorizontalHeaderLabels(['class', 'name', 'value'])

    def keyPressEvent(self, event):
        self.test._Keyboard_Event(event.key(), down=True)

    def keyReleaseEvent(self, event):
        self.test._Keyboard_Event(event.key(), down=False)

    @property
    def settings(self):
        return QtCore.QSettings("pybox2d", "Framework")

    def setFontSize(self, size):
        """
        Update the global font size
        """
        if size <= 0.0:
            return

        global app
        font = app.font()
        font.setPointSize(size)
        app.setFont(font)

    def restoreLayout(self):
        """
        Restore the layout of each widget
        """
        settings = self.settings
        try:
            self.restoreGeometry(settings.value("geometry").toByteArray())
            self.restoreState(settings.value("windowState").toByteArray())
            size = settings.value('fontSize').toFloat()[0]
            self.setFontSize(size)
        except:
            pass

    def saveLayout(self):
        """
        Save the layout of each widget
        """
        settings = self.settings
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        settings.setValue("fontSize", app.font().pointSize())

    def closeEvent(self, event):
        QtGui.QMainWindow.closeEvent(self, event)
        self.saveLayout()

app = None


class Pyqt4Framework(FrameworkBase):
    TEXTLINE_START = 0

    def setup_keys(self):
        # Only basic keys are mapped for now: K_[a-z0-9], K_F[1-12] and
        # K_COMMA.

        for letter in string.ascii_uppercase:
            setattr(Keys, 'K_' + letter.lower(),
                    getattr(Qt, 'Key_%s' % letter))
        for i in range(0, 10):
            setattr(Keys, 'K_%d' % i, getattr(Qt, 'Key_%d' % i))
        for i in range(1, 13):
            setattr(Keys, 'K_F%d' % i, getattr(Qt, 'Key_F%d' % i))
        Keys.K_LEFT = Qt.Key_Left
        Keys.K_RIGHT = Qt.Key_Right
        Keys.K_UP = Qt.Key_Up
        Keys.K_DOWN = Qt.Key_Down
        Keys.K_HOME = Qt.Key_Home
        Keys.K_PAGEUP = Qt.Key_PageUp
        Keys.K_PAGEDOWN = Qt.Key_PageDown
        Keys.K_COMMA = Qt.Key_Comma
        Keys.K_SPACE = Qt.Key_Space

    def __reset(self):
        # Screen/rendering-related
        self._viewZoom = 10.0
        self._viewCenter = None
        self._viewOffset = None
        self.screenSize = None
        self.textLine = 0
        self.font = None
        self.fps = 0
        self.selected_shapebody = None, None

        # GUI-related
        self.window = None
        self.setup_keys()

    def __init__(self):
        super(Pyqt4Framework, self).__init__()

        self.__reset()

        if settings.fwSettings.onlyInit:  # testing mode doesn't initialize Pyqt4
            return

        global app
        app = QtGui.QApplication(sys.argv)

        print('Initializing Pyqt4 framework...')

        # Pyqt4 Initialization
        self.window = MainWindow(self)
        self.window.show()

        self.window.setWindowTitle("Python Box2D Testbed - " + self.name)
        self.renderer = Pyqt4Draw(self)

        # Note that in this framework, we override the draw debug data routine
        # that occurs in Step(), and we implement the normal C++ code in
        # Python.
        self.world.DrawDebugData = lambda: self.renderer.ManualDraw()
        self.screenSize = b2Vec2(0, 0)
        self.viewCenter = (0, 10.0 * 20.0)
        self.groundbody = self.world.CreateBody()

    def setCenter(self, value):
        """
        Updates the view offset based on the center of the screen.

        Tells the debug draw to update its values also.
        """
        self._viewCenter = b2Vec2(*value)
        self._viewOffset = self._viewCenter - self.screenSize / 2
        self.window.graphicsView.centerOn(*self._viewCenter)

    def setZoom(self, zoom):
        self._viewZoom = zoom
        self.window.graphicsView.resetTransform()
        self.window.graphicsView.scale(self._viewZoom, -self._viewZoom)
        self.window.graphicsView.centerOn(*self._viewCenter)

    viewZoom = property(lambda self: self._viewZoom, setZoom,
                        doc='Zoom factor for the display')
    viewCenter = property(lambda self: self._viewCenter, setCenter,
                          doc='Screen center in camera coordinates')
    viewOffset = property(lambda self: self._viewOffset,
                          doc='The offset of the top-left corner of the screen')

    def run(self):
        """
        What would be the main loop is instead a call to
        app.exec_() for the event-driven pyqt4.
        """
        global app
        self.step_timer = QtCore.QTimer()

        self.step_timer.timeout.connect(self.SimulationLoop)
        self.window.twProperties.itemChanged.connect(self.prop_cell_changed)
        self.step_timer.start(int((1000.0 / self.settings.hz)))

        app.exec_()

        self.step_timer.stop()
        print('Cleaning up...')
        self.world.contactListener = None
        self.world.destructionListener = None
        self.world.renderer = None
        self.world = None

    def _Keyboard_Event(self, key, down=True):
        """
        Internal keyboard event, don't override this.

        Checks for the initial keydown of the basic testbed keys. Passes the unused
        ones onto the test via the Keyboard() function.
        """
        if down:
            if key == Keys.K_z:       # Zoom in
                self.viewZoom = min(1.10 * self.viewZoom, 50.0)
            elif key == Keys.K_x:     # Zoom out
                self.viewZoom = max(0.9 * self.viewZoom, 0.02)
            elif key == Keys.K_SPACE:  # Launch a bomb
                self.LaunchRandomBomb()
            else:              # Inform the test of the key press
                self.Keyboard(key)
        else:
            self.KeyboardUp(key)

    def CheckKeys(self):
        pass

    def _ShowProperties(self, obj):
        self.selected_shapebody = None, None

        class_ = obj.__class__
        ignore_list = ('thisown',)

        i = 0
        twProperties = self.window.twProperties
        # Get all of the members of the class
        for prop in dir(class_):
            # If they're properties and not to be ignored, add them to the
            # table widget
            if (isinstance(getattr(class_, prop), property)
                    and prop not in ignore_list):
                try:
                    value = getattr(obj, prop)
                except:
                    # Write-only?
                    continue

                widget = None

                # Attempt to determine whether it's read-only or not
                try:
                    setattr(obj, prop, value)
                except:
                    editable = False
                else:
                    editable = True

                # Increase the row count and insert the new item
                twProperties.setRowCount(twProperties.rowCount() + 1)
                i = twProperties.rowCount() - 1
                self.item = QTableWidgetItem(class_.__name__)
                twProperties.setItem(i, 0, QTableWidgetItem(
                    class_.__name__))  # class name
                twProperties.item(i, 0).setFlags(Qt.ItemIsEnabled)

                twProperties.setItem(
                    i, 1, QtGui.QTableWidgetItem(prop))      # prop name
                twProperties.item(i, 1).setFlags(Qt.ItemIsEnabled)

                # and finally, the property values
                # booleans are checkboxes
                if isinstance(value, bool):
                    def state_changed(value, prop=prop):
                        self.property_changed(prop, value == Qt.Checked)

                    widget = QtGui.QCheckBox('')
                    widget.stateChanged.connect(state_changed)
                    if value:
                        widget.setCheckState(Qt.Checked)

                # ints, floats are spinboxes
                elif isinstance(value, (int, float)):
                    def value_changed(value, prop=prop):
                        self.property_changed(prop, value)

                    widget = QtGui.QDoubleSpinBox()
                    widget.valueChanged.connect(value_changed)
                    widget.setValue(value)
                # lists turn into -- lists
                elif isinstance(value, list):
                    widget = QtGui.QListWidget()
                    for entry in value:
                        widget.addItem(str(entry))
                    if value:
                        # sz=widget.item(0).sizeHint()
                        # print(sz, sz.width(), sz.height())
                        # sz.setHeight(sz.height()*2)
                        # widget.setMinimumSize(sz)
                        # widget.setMinimumSize(QtCore.QSize(1,60))
                        pass  # TODO
                # vec2s will be shown as a textbox
                elif isinstance(value, b2Vec2):
                    value = '(%.2f, %.2f)' % (tuple(value))
                else:
                    pass

                if widget:
                    twProperties.setCellWidget(i, 2, widget)
                    if hasattr(widget, 'setReadOnly'):
                        widget.setReadOnly(not editable)
                    elif hasattr(widget, 'setEnabled'):
                        widget.setEnabled(editable)
                else:
                    # Just using the table widget, set the cell text
                    cell = QtGui.QTableWidgetItem(str(value))
                    if editable:
                        cell.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled)
                    else:
                        cell.setFlags(Qt.ItemIsEnabled)
                    twProperties.setItem(i, 2, cell)

                i += 1

    # callback indicating a cell in the table widget was changed
    def prop_cell_changed(self, twi):
        if twi.column() != 2:  # the data column
            return

        row = twi.row()
        prop = str(self.window.twProperties.item(row, 1).text())
        self.property_changed(prop, str(twi.text()))

    # callback indicating one of the property widgets was modified
    def property_changed(self, prop, value=None):
        if not self.selected_shapebody[0]:
            return

        print('Trying to change %s to %s...' % (prop, value))
        shape, body = self.selected_shapebody
        for inst in (shape, body):
            if hasattr(inst, prop):
                try:
                    cur_value = getattr(inst, prop)
                    if isinstance(cur_value, b2Vec2):
                        m = re.search('\(?([\d\.]*)\s*,\s*([\d\.]*)\)?', value)
                        if m:
                            x, y = m.groups()
                            value = (float(x), float(y))
                except:
                    raise
                    pass

                try:
                    setattr(inst, prop, value)
                except:
                    print('Failed - %s' % sys.exc_info()[1])

    def ShowProperties(self, p):
        aabb = b2AABB(lowerBound=p - (0.001, 0.001),
                      upperBound=p + (0.001, 0.001))

        # Query the world for overlapping shapes.
        query = fwQueryCallback(p)
        self.world.QueryAABB(query, aabb)

        if query.fixture:
            self.window.reset_properties_list()

            fixture = query.fixture
            body = fixture.body
            self._ShowProperties(body)

            shape = fixture.shape
            self._ShowProperties(shape)

            self.selected_shapebody = (shape, body)

    def Step(self, settings):
        super(Pyqt4Framework, self).Step(settings)

    def ConvertScreenToWorld(self, x, y):
        """
        PyQt4 gives us transformed positions, so no need to convert
        """
        return b2Vec2(x, y)

    DrawStringAt = lambda self, *args: self.renderer.DrawStringAt(*args)

    def Print(self, str, color=(229, 153, 153, 255)):
        """
        Draw some text at the top status lines and advance to the next line.
        """
        self.DrawStringAt(5, self.textLine, str, color)
        self.textLine += self.renderer.font_spacing

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

    def FixtureDestroyed(self, fixture):
        shape = fixture.shape
        if shape == self.selected_shapebody[0]:
            self.selected_shapebody = None, None
            self.window.reset_properties_list()
        if hash(shape) in self.renderer.item_cache:
            scene_items = self.renderer.item_cache[hash(shape)]
            for item in scene_items:
                self.window.scene.removeItem(item)
            del self.renderer.item_cache[hash(shape)]
