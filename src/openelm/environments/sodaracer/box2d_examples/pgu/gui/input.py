"""
"""
import pygame
from pygame.locals import *

from .const import *
from . import widget

class Input(widget.Widget):
    """A single line text input.
    
    Example:
        w = Input(value="Cuzco the Goat",size=20)
        w = Input("Marbles")
    
    """

    _value = None

    def __init__(self,value="",size=20,**params):
        """Create a new Input widget.

        Keyword arguments:
            value -- initial text
            size -- size for the text box, in characters

        """
        params.setdefault('cls','input')
        widget.Widget.__init__(self,**params)
        self.value = value
        self.pos = len(str(value))
        self.vpos = 0
        self.font = self.style.font
        w,h = self.font.size("e"*size)
        if not self.style.height: self.style.height = h
        if not self.style.width: self.style.width = w
        #self.style.height = max(self.style.height,h)
        #self.style.width = max(self.style.width,w)
        #self.rect.w=w+self.style.padding_left+self.style.padding_right;
        #self.rect.h=h+self.style.padding_top+self.style.padding_bottom;
    
    def paint(self,s):
        r = pygame.Rect(0,0,self.rect.w,self.rect.h)
        
        cs = 2 #NOTE: should be in a style
        
        w,h = self.font.size(self.value[0:self.pos])
        x = w-self.vpos
        if x < 0: self.vpos -= -x
        if x+cs > s.get_width(): self.vpos += x+cs-s.get_width()
        
        s.blit(self.font.render(self.value, 1, self.style.color),(-self.vpos,0))
        
        if self.container.myfocus is self:
            w,h = self.font.size(self.value[0:self.pos])
            r.x = w-self.vpos
            r.w = cs
            r.h = h
            s.fill(self.style.color,r)
    
    def _setvalue(self,v):
        #self.__dict__['value'] = v
        self._value = v
        self.send(CHANGE)
    
    def event(self,e):
        used = None
        if e.type == KEYDOWN:
            if e.key == K_BACKSPACE:
                if self.pos:
                    self._setvalue(self.value[:self.pos-1] + self.value[self.pos:])
                    self.pos -= 1
            elif e.key == K_DELETE:
                if len(self.value) > self.pos:
                    self._setvalue(self.value[:self.pos] + self.value[self.pos+1:])
            elif e.key == K_HOME: 
                self.pos = 0
            elif e.key == K_END:
                self.pos = len(self.value)
            elif e.key == K_LEFT:
                if self.pos > 0: self.pos -= 1
                used = True
            elif e.key == K_RIGHT:
                if self.pos < len(self.value): self.pos += 1
                used = True
            elif e.key == K_RETURN:
                self.next()
            elif e.key == K_TAB:
                pass
            else:
                #c = str(e.unicode)
                try:
                    c = (e.unicode).encode('latin-1')
                    if c:
                        self._setvalue(self.value[:self.pos] + c + self.value[self.pos:])
                        self.pos += 1
                except: #ignore weird characters
                    pass
            self.repaint()
        elif e.type == FOCUS:
            self.repaint()
        elif e.type == BLUR:
            self.repaint()
        
        self.pcls = ""
        if self.container.myfocus is self: self.pcls = "focus"
        
        return used
    
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        if (val == None): 
            val = ""
        val = str(val)
        self.pos = len(val)
        oldval = self._value
        self._value = val
        if (oldval != val):
            self.send(CHANGE)
            self.repaint()


class Password(Input):
    """A password input, in which text is rendered with '*' characters."""

    def paint(self,s):
        hidden="*"
        show=len(self.value)*hidden
        
        #print "self.value:",self.value

        if self.pos == None: self.pos = len(self.value)
        
        r = pygame.Rect(0,0,self.rect.w,self.rect.h)
        
        cs = 2 #NOTE: should be in a style
        
        w,h = self.font.size(show)
        x = w-self.vpos
        if x < 0: self.vpos -= -x
        if x+cs > s.get_width(): self.vpos += x+cs-s.get_width()
        
        s.blit(self.font.render(show, 1, self.style.color),(-self.vpos,0))
        
        if self.container.myfocus is self:
            #w,h = self.font.size(self.value[0:self.pos])            
            w,h = self.font.size(show[0:self.pos])
            r.x = w-self.vpos
            r.w = cs
            r.h = h
            s.fill(self.style.color,r)

