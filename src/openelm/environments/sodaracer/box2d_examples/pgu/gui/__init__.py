"""Modules for creating a widget-based user interface. See the examples folder 
for sample scripts that use this module."""

import pygame

# The basestring class was removed in Python 3, but we want to keep it to maintain 
# compatibility with previous versions of python.
try:
    __builtins__["basestring"]
except KeyError:
    __builtins__["basestring"] = str

from .theme import Theme
from .style import Style
from .widget import Widget
from .surface import subsurface, ProxySurface
from .const import *

from .container import Container
from .app import App, Desktop
from .table import Table
from .document import Document
#html
from .area import SlideBox, ScrollArea, List 

from .form import Form
from .group import Group

from .basic import Spacer, Color, Label, Image, parse_color
from .button import Icon, Button, Switch, Checkbox, Radio, Tool, Link
from .input import Input, Password
from .keysym import Keysym
from .slider import VSlider, HSlider, VScrollBar, HScrollBar
from .select import Select
from .misc import ProgressBar

from .menus import Menus
from .dialog import Dialog, FileDialog
from .textarea import TextArea

from .deprecated import Toolbox, action_open, action_setvalue, action_quit, action_exec
