from .const import *
from . import widget
from . import pguglobals

class ProgressBar(widget.Widget):
    """A progress bar widget.
    
    Example:
        w = gui.ProgressBar(0,0,100)
        w.value = 25

    """

    _value = None

    def __init__(self,value,min,max,**params):
        params.setdefault('cls','progressbar')
        widget.Widget.__init__(self,**params)
        self.min,self.max,self.value = min,max,value
    
    def paint(self,s):
        if (self.value != None):
            r = pygame.rect.Rect(0,0,self.rect.w,self.rect.h)
            r.w = r.w*(self.value-self.min)/(self.max-self.min)
            self.bar = r
            pguglobals.app.theme.render(s,self.style.bar,r)
        
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        val = int(val)
        val = max(val, self.min)
        val = min(val, self.max)
        oldval = self._value
        self._value = val
        if (oldval != val):
            self.send(CHANGE)
            self.repaint()


