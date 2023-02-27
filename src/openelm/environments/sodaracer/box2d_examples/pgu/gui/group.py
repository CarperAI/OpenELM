"""
"""
from .const import *
from . import widget

class Group(widget.Widget):
    """An object for grouping together Form elements.
    
    When the value changes, an gui.CHANGE event is sent. Although note, 
    that when the value is a list, it may have to be sent by hand via 
    g.send(gui.CHANGE).

    """

    _value = None
    widgets = None
    
    def __init__(self,name=None,value=None):
        """Create Group instance.

        Arguments:
        name -- name as used in the Form
        value -- values that are currently selected in the group
    
        """
        widget.Widget.__init__(self,name=name,value=value)
        self.widgets = []
    
    def add(self,w):
        """Add a widget to this group."""
        self.widgets.append(w)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        oldval = self._value
        self._value = val
        if (oldval != val):
            self._change()
    
    def _change(self):
        self.send(CHANGE)
        if (self.widgets):
            for w in self.widgets:
                w.repaint()

