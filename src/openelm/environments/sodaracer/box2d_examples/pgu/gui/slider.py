import pygame
from pygame.locals import *

from .const import *
from . import widget
from . import table
from . import basic
from . import pguglobals

_SLIDER_HORIZONTAL = 0
_SLIDER_VERTICAL = 1

class _slider(widget.Widget):
    _value = None

    def __init__(self,value,orient,min,max,size,step=1,**params):
        params.setdefault('cls','slider')
        widget.Widget.__init__(self,**params)
        self.min,self.max,self.value,self.orient,self.size,self.step = min,max,value,orient,size,step
        
    
    def paint(self,s):
        
        self.value = self.value
        r = pygame.rect.Rect(0,0,self.style.width,self.style.height)
        if self.orient == _SLIDER_HORIZONTAL:
            r.x = (self.value-self.min) * (r.w-self.size) / max(1,self.max-self.min);
            r.w = self.size;
        else:
            r.y = (self.value-self.min) * (r.h-self.size) / max(1,self.max-self.min);
            r.h = self.size;
            
        self.bar = r
        
        pguglobals.app.theme.render(s,self.style.bar,r)
    
    def event(self,e):
        used = None
        r = pygame.rect.Rect(0,0,self.style.width,self.style.height)
        adj = 0
        if e.type == ENTER: self.repaint()
        elif e.type == EXIT: self.repaint()
        elif e.type == MOUSEBUTTONDOWN:
            if self.bar.collidepoint(e.pos):
                self.grab = e.pos[0],e.pos[1]
                self.grab_value = self.value
            else:
                x,y,adj = e.pos[0],e.pos[1],1
                self.grab = None
            self.repaint()
        elif e.type == MOUSEBUTTONUP:
            #x,y,adj = e.pos[0],e.pos[1],1
            self.repaint()
        elif e.type == MOUSEMOTION:
            if 1 in e.buttons and self.container.myfocus is self:
                if self.grab != None:
                    rel = e.pos[0]-self.grab[0],e.pos[1]-self.grab[1]
                    if self.orient == _SLIDER_HORIZONTAL:
                        d = (r.w - self.size)
                        if d != 0: self.value = self.grab_value + ((self.max-self.min) * rel[0] / d)
                    else:
                        d = (r.h - self.size)
                        if d != 0: self.value = self.grab_value + ((self.max-self.min) * rel[1] / d)
                else:
                    x,y,adj = e.pos[0],e.pos[1],1
                    
        elif e.type is KEYDOWN:
            if self.orient == _SLIDER_HORIZONTAL and e.key == K_LEFT:
                self.value -= self.step
                used = True
            elif self.orient == _SLIDER_HORIZONTAL and e.key == K_RIGHT:
                self.value += self.step
                used = True
            elif self.orient == _SLIDER_VERTICAL and e.key == K_UP:
                self.value -= self.step
                used = True
            elif self.orient == _SLIDER_VERTICAL and e.key == K_DOWN:
                self.value += self.step
                used = True

        if adj:
            if self.orient == _SLIDER_HORIZONTAL:
                d = self.size/2 - (r.w/(self.max-self.min+1))/2
                self.value = (x-d) * (self.max-self.min) / (r.w-self.size+1) + self.min
            else:
                d = self.size/2 - (r.h/(self.max-self.min+1))/2
                self.value = (y-d) * (self.max-self.min) / (r.h-self.size+1) + self.min
                
        self.pcls = ""
        if self.container.myhover is self: self.pcls = "hover"
        if (self.container.myfocus is self and 1 in pygame.mouse.get_pressed()): self.pcls = "down"
        
        return used

    # TODO - replace this with property functions and setters
    def __setattr__(self,k,v):
        if k == 'value':
            v = int(v)
            v = max(v,self.min)
            v = min(v,self.max)
        _v = self.__dict__.get(k,NOATTR)
        self.__dict__[k]=v
        if k == 'value' and _v != NOATTR and _v != v: 
            self.send(CHANGE)
            self.repaint()
        if hasattr(self,'size'):
            sz = min(self.size,max(self.style.width,self.style.height))
            sz = max(sz,min(self.style.width,self.style.height))
            self.__dict__['size'] = sz
            #self.size = sz
        if hasattr(self,'max') and hasattr(self,'min'):
            if self.max < self.min: self.max = self.min

#    @property
#    def value(self):
#        return self._value
#
#    @value.setter
#    def value(self, val):
#        val = int(val)
#        val = max(val, self.min)
#        val = min(val, self.max)
#
#        oldval = self._value
#        self._value = val
#        if (oldval != val):
#            self.send(CHANGE)
#            self.repaint()
#            
#        if hasattr(self,'size'):
#            sz = min(self.size,max(self.style.width,self.style.height))
#            sz = max(sz,min(self.style.width,self.style.height))
#            self.size = sz
#            
#        if hasattr(self,'max') and hasattr(self,'min'):
#            if self.max < self.min: self.max = self.min
    

class VSlider(_slider):
    """A verticle slider."""

    def __init__(self,value,min,max,size,step=1,**params):
        """Construct a veritcal slider widget.

        Arguments:
            value -- the default position of the slider, between min and max
            min -- the minimum value for the slider
            max -- the maximum value
            size -- the length of the slider bar in pixels
            step -- how much to jump when using the keyboard

        """
        params.setdefault('cls','vslider')
        _slider.__init__(self,value,_SLIDER_VERTICAL,min,max,size,step,**params)

class HSlider(_slider):
    """A horizontal slider."""

    def __init__(self,value,min,max,size,step=1,**params):
        params.setdefault('cls','hslider')
        _slider.__init__(self,value,_SLIDER_HORIZONTAL,min,max,size,step,**params)

class HScrollBar(table.Table):
    """A horizontal scroll bar."""

    def __init__(self,value,min,max,size,step=1,**params):
        params.setdefault('cls','hscrollbar')
        
        table.Table.__init__(self,**params)
        
        self.slider = _slider(value,_SLIDER_HORIZONTAL,min,max,size,step=step,cls=self.cls+'.slider')
        
        self.minus = basic.Image(self.style.minus)
        self.minus.connect(MOUSEBUTTONDOWN,self._click,-1)
        self.slider.connect(CHANGE,self.send,CHANGE)
        
        self.minus2 = basic.Image(self.style.minus)
        self.minus2.connect(MOUSEBUTTONDOWN,self._click,-1)
        
        self.plus = basic.Image(self.style.plus)
        self.plus.connect(MOUSEBUTTONDOWN,self._click,1)
        
        self.size = size
        
    def _click(self,value):
        self.slider.value += self.slider.step*value
        
    def resize(self,width=None,height=None):
        self.clear()
        self.tr()
        
        w = self.style.width
        h = self.slider.style.height
        ww = 0
        
        if w > (h*2 + self.minus.style.width+self.plus.style.width):
            self.td(self.minus)
            ww += self.minus.style.width
        
        self.td(self.slider)
        
        if w > (h*2 + self.minus.style.width+self.minus2.style.width+self.plus.style.width):
            self.td(self.minus2)
            ww += self.minus2.style.width
        
        if w > (h*2 + self.minus.style.width+self.plus.style.width):
            self.td(self.plus)
            ww += self.plus.style.width
            
            
        #HACK: handle theme sizing properly
        xt,xr,xb,xl = pguglobals.app.theme.getspacing(self.slider)
        ww += xr+xl

        self.slider.style.width = self.style.width - ww
        setattr(self.slider,'size',self.size * self.slider.style.width / max(1,self.style.width))
        #self.slider.size = self.size * self.slider.style.width / max(1,self.style.width)
        return table.Table.resize(self,width,height)

    @property
    def min(self):
        return self.slider.min

    @min.setter
    def min(self, value):
        self.slider.min = value

    @property
    def max(self):
        return self.slider.max

    @max.setter
    def max(self, value):
        self.slider.max = value

    @property
    def value(self):
        return self.slider.value

    @value.setter
    def value(self, value):
        self.slider.value = value

    @property
    def step(self):
        return self.slider.step

    @step.setter
    def step(self, value):
        self.slider.step = value
        
#    def __setattr__(self,k,v):
#        if k in ('min','max','value','step'):
#            return setattr(self.slider,k,v)
#        self.__dict__[k]=v
            
#    def __getattr__(self,k):
#        if k in ('min','max','value','step'):
#            return getattr(self.slider,k)
#        return table.Table.__getattr__(self,k) #self.__dict__[k]

class VScrollBar(table.Table):
    """A vertical scroll bar."""

    def __init__(self,value,min,max,size,step=1,**params):
        params.setdefault('cls','vscrollbar')
        
        table.Table.__init__(self,**params)
        
        self.minus = basic.Image(self.style.minus)
        self.minus.connect(MOUSEBUTTONDOWN,self._click,-1)
        
        self.minus2 = basic.Image(self.style.minus)
        self.minus2.connect(MOUSEBUTTONDOWN,self._click,-1)
        
        self.plus = basic.Image(self.style.plus)
        self.plus.connect(MOUSEBUTTONDOWN,self._click,1)
        
        self.slider = _slider(value,_SLIDER_VERTICAL,min,max,size,step=step,cls=self.cls+'.slider')
        self.slider.connect(CHANGE,self.send,CHANGE)
        
        self.size = size
        
    def _click(self,value):
        self.slider.value += self.slider.step*value
        
    def resize(self,width=None,height=None):
        self.clear()
        
        h = self.style.height
        w = self.slider.style.width
        hh = 0
        
        if h > (w*2 + self.minus.style.height+self.plus.style.height):
            self.tr()
            self.td(self.minus)
            hh += self.minus.style.height
        
        self.tr()
        self.td(self.slider)
        
        if h > (w*2 + self.minus.style.height+self.minus2.style.height+self.plus.style.height):
            self.tr()
            self.td(self.minus2)
            hh += self.minus2.style.height
        
        if h > (w*2 + self.minus.style.height+self.plus.style.height):
            self.tr()
            self.td(self.plus)
            hh += self.plus.style.height
            
            
        #HACK: handle theme sizing properly
        xt,xr,xb,xl = pguglobals.app.theme.getspacing(self.slider)
        hh += xt+xb

        self.slider.style.height = self.style.height - hh
        setattr(self.slider,'size',self.size * self.slider.style.height / max(1,self.style.height))
        return table.Table.resize(self,width,height)
        
    def __setattr__(self,k,v):
        if k in ('min','max','value','step'):
            return setattr(self.slider,k,v)
        self.__dict__[k]=v
            
    def __getattr__(self,k):
        if k in ('min','max','value','step'):
            return getattr(self.slider,k)
        return table.Table.__getattr__(self,k)

