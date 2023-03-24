#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2010 Ken Lauer / sirkne at gmail dot com
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

try:
    from ..pgu import gui
except:
    raise ImportError('Unable to load PGU')

from ..settings import checkboxes
from ..settings import sliders


class fwGUI(gui.Table):
    """
    Deals with the initialization and changing the settings based on the GUI
    controls. Callbacks are not used, but the checkboxes and sliders are polled
    by the main loop.
    """
    form = None

    def __init__(self,settings, **params):
        # The framework GUI is just basically a HTML-like table
        # There are 2 columns right-aligned on the screen
        gui.Table.__init__(self,**params)
        self.form=gui.Form()

        fg = (255,255,255)

        # "Toggle menu"
        self.tr()
        self.td(gui.Label("F1: Toggle Menu",color=(255,0,0)),align=1,colspan=2)

        for slider in sliders:
            # "Slider title"
            self.tr()
            self.td(gui.Label(slider['text'],color=fg),align=1,colspan=2)

            # Create the slider
            self.tr()
            e = gui.HSlider(getattr(settings, slider['name']),slider['min'],slider['max'],size=20,width=100,height=16,name=slider['name'])
            self.td(e,colspan=2,align=1)

        # Add each of the checkboxes.
        for text, variable in checkboxes:
            self.tr()
            if variable == None:
                # Checkboxes that have no variable (i.e., None) are just labels.
                self.td(gui.Label(text, color=fg), align=1, colspan=2)
            else:
                # Add the label and then the switch/checkbox
                self.td(gui.Label(text, color=fg), align=1)
                self.td(gui.Switch(value=getattr(settings, variable),name=variable))

    def updateGUI(self, settings):
        """
        Change all of the GUI elements based on the current settings
        """
        for text, variable in checkboxes:
            if not variable: continue
            if hasattr(settings, variable):
                self.form[variable].value = getattr(settings, variable)

        # Now do the sliders
        for slider in sliders:
            name=slider['name']
            self.form[name].value=getattr(settings, name)

    def updateSettings(self, settings):
        """
        Change all of the settings based on the current state of the GUI.
        """
        for text, variable in checkboxes:
            if variable:
                setattr(settings, variable, self.form[variable].value)

        # Now do the sliders
        for slider in sliders:
            name=slider['name']
            setattr(settings, name, int(self.form[name].value))

        # If we're in single-step mode, update the GUI to reflect that.
        if settings.singleStep:
            settings.pause=True
            self.form['pause'].value = True
            self.form['singleStep'].value = False

