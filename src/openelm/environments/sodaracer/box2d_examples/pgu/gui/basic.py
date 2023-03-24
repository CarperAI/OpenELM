"""These widgets are all grouped together because they are non-interactive widgets.
"""

import pygame

from .const import *
from . import widget

# Turns a descriptive string or a tuple into a pygame color
def parse_color(desc):
    if (is_color(desc)):
        # Already a color
        return desc
    elif (desc and desc[0] == "#"):
        # Because of a bug in pygame 1.8.1 we need to explicitly define the 
        # alpha value otherwise it will default to transparent.
        if (len(desc) == 7):
            desc += "FF"
    return pygame.Color(desc)

# Determines if the given object is a pygame-compatible color or not
def is_color(col):
    # In every version of pygame (up to 1.8.1 so far) will interpret
    # a tuple as a color.
    if (type(col) == tuple):
        return col
    if (hasattr(pygame, "Color") and type(pygame.Color) == type):
        # This is a recent version of pygame that uses a proper type
        # instance for colors.
        return (isinstance(col, pygame.Color))
    # Otherwise, this version of pygame only supports tuple colors
    return False

class Spacer(widget.Widget):
    """An invisible space widget."""

    def __init__(self,width,height,**params):
        params.setdefault('focusable',False)
        widget.Widget.__init__(self,width=width,height=height,**params)
        

class Color(widget.Widget):
    """A widget that renders as a solid block of color.
    
    Note the color can be changed by setting the 'value' field, and the 
    widget will automatically be repainted, eg:

        c = Color()
        c.value = (255,0,0)
        c.value = (0,255,0)

    """

    _value = None
    
    def __init__(self,value=None,**params):
        params.setdefault('focusable',False)
        if value != None: params['value']=value
        widget.Widget.__init__(self,**params)
    
    def paint(self,s):
        if hasattr(self,'value'): s.fill(self.value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        if (isinstance(val, basestring)):
            # Parse the string as a color
            val = parse_color(val)
        oldval = self._value
        self._value = val
        if (oldval != val):
            # Emit a change signal
            self.send(CHANGE)
            self.repaint()
    

class Label(widget.Widget):
    """A text label widget."""

    def __init__(self, value="", **params):
        params.setdefault('focusable', False)
        params.setdefault('cls', 'label')
        widget.Widget.__init__(self, **params)
        self.value = value
        self.font = self.style.font
        self.style.width, self.style.height = self.font.size(self.value)
    
    def paint(self,s):
        """Renders the label onto the given surface in the upper-left corner."""
        s.blit(self.font.render(self.value, 1, self.style.color),(0,0))

    def set_text(self, txt):
        """Set the text of this label."""
        self.value = txt
        # Signal to the application that we need to resize this widget
        self.chsize()

    def set_font(self, font):
        """Set the font used to render this label."""
        this.font = font
        # Signal to the application that we need a resize
        this.chsize()

    def resize(self,width=None,height=None):
        # Calculate the size of the rendered text
        (self.style.width, self.style.height) = self.font.size(self.value)
        return (self.style.width, self.style.height)


class Image(widget.Widget):
    """An image widget. The constructor takes a file name or a pygame surface."""

    def __init__(self,value,**params):
        params.setdefault('focusable',False)
        widget.Widget.__init__(self,**params)
        if type(value) == str: value = pygame.image.load(value)
        
        ow,oh = iw,ih = value.get_width(),value.get_height()
        sw,sh = self.style.width,self.style.height
        
        if sw and not sh:
            iw,ih = sw,ih*sw/iw
        elif sh and not sw:
            iw,ih = iw*sh/ih,sh
        elif sw and sh:
            iw,ih = sw,sh
        
        if (ow,oh) != (iw,ih):
            value = pygame.transform.scale(value,(iw,ih))
        self.style.width,self.style.height = iw,ih
        self.value = value
    
    def paint(self,s):
        s.blit(self.value,(0,0))

