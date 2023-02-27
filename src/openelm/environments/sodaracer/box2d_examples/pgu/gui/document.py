"""
"""
import pygame

from . import container
from . import layout

class _document_widget:
    def __init__(self,w,align=None):
        #w.rect.w,w.rect.h = w.resize()
        #self.rect = w.rect
        self.widget = w
        if align != None: self.align = align

class Document(container.Container):
    """A document is a container that structures widgets in a left-to-right flow."""

    def __init__(self,**params):
        params.setdefault('cls','document')
        container.Container.__init__(self,**params)
        self.layout =  layout.Layout(pygame.Rect(0,0,self.rect.w,self.rect.h))
    
    def add(self,e,align=None):
        """Add a widget to the document flow.

        Arguments:
            e -- widget
            align -- alignment (None,-1,0,1)

        """
        dw = _document_widget(e,align)
        self.layout.add(dw)
        e.container = self
        e._c_dw = dw
        self.widgets.append(e)
        self.chsize()
        
    def remove(self,e):
        self.layout._widgets.remove(e._c_dw)
        self.widgets.remove(e)
        self.chsize()
        
    
    def block(self,align):
        """Start a new block given the alignment (-1, 0, 1)"""
        self.layout.add(align)
    
    def space(self, size):
        """Add a spacer given the size."""
        self.layout.add(size)
    
    def br(self,height):
        """Add a line break, given the height."""
        self.layout.add((0, height))
    
    def resize(self,width=None,height=None):
        if self.style.width: width = self.style.width
        if self.style.height: height = self.style.height
        
        for w in self.widgets:
            w.rect.w,w.rect.h = w.resize()
            
            if (width != None and w.rect.w > width) or (height != None and w.rect.h > height):
                w.rect.w,w.rect.h = w.resize(width,height)
            
            dw = w._c_dw
            dw.rect = pygame.Rect(0,0,w.rect.w,w.rect.h)
        
        if width == None: width = 65535
        self.layout.rect = pygame.Rect(0,0,width,0)
        self.layout.resize()
        
        _max_w = 0
        
        for w in self.widgets:
            #xt,xl,xb,xr = w.getspacing()
            dw = w._c_dw
            w.style.x,w.style.y,w.rect.w,w.rect.h = dw.rect.x,dw.rect.y,dw.rect.w,dw.rect.h
            #w.resize()
            w.rect.x,w.rect.y = w.style.x,w.style.y
            _max_w = max(_max_w,w.rect.right)
        
        #self.rect.w = _max_w #self.layout.rect.w
        #self.rect.h = self.layout.rect.h
        #print 'document',_max_w,self.layout.rect.h
        return _max_w,self.layout.rect.h

