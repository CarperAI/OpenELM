"""
"""
import os

from . import pguglobals
from .const import *
from . import surface
from . import container, table
from . import group
from . import basic, button, slider

class SlideBox(container.Container):
    """A scrollable area with no scrollbars.
    
    Example:
        c = SlideBox(w,100,100)
        c.offset = (10,10)
        c.repaint()
    
    """

    _widget = None
    
    def __init__(self, widget, width, height, **params):
        """SlideBox constructor.

        Arguments:
            widget -- widget to be able to scroll around
            width, height -- size of scrollable area
    
        """
        params.setdefault('width', width)
        params.setdefault('height', height)
        container.Container.__init__(self, **params)
        self.offset = [0, 0]
        self.widget = widget

    @property
    def widget(self):
        return self._widget

    @widget.setter
    def widget(self, val):
        # Remove the old widget first
        if self._widget:
            self.remove(self._widget)
        # Now add in the new widget
        self._widget = val
        self.add(val, 0, 0)
    
    def paint(self, s):
        #if not hasattr(self,'surface'):
        self.surface = pygame.Surface((self.max_rect.w,self.max_rect.h),0,s)
        #self.surface.fill((0,0,0,0))
        pguglobals.app.theme.render(self.surface,self.style.background,pygame.Rect(0,0,self.max_rect.w,self.max_rect.h))
        self.bkgr = pygame.Surface((s.get_width(),s.get_height()),0,s)
        self.bkgr.blit(s,(0,0))
        container.Container.paint(self,self.surface)
        s.blit(self.surface,(-self.offset[0],-self.offset[1]))
        self._offset = self.offset[:]
        return
        
    def paint_for_when_pygame_supports_other_tricks(self,s): 
        #this would be ideal if pygame had support for it!
        #and if pgu also had a paint(self,s,rect) method to paint small parts
        sr = (self.offset[0],self.offset[1],self.max_rect.w,self.max_rect.h)
        cr = (-self.offset[0],-self.offset[1],s.get_width(),s.get_height())
        s2 = s.subsurface(sr)
        s2.set_clip(cr)
        container.Container.paint(self,s2)
        
    def proxy_paint(self, s):
        container.Container.paint(self, surface.ProxySurface(parent=None, 
                                           rect=self.max_rect,
                                           real_surface=s,
                                           offset=self.offset))
    def update(self, s):
        rects = container.Container.update(self,self.surface)
        
        rets = []
        s_rect = pygame.Rect(0,0,s.get_width(),s.get_height())
        
        if self.offset == self._offset:
            for r in rects:
                r2 = r.move((-self.offset[0],-self.offset[1]))
                if r2.colliderect(s_rect):
                    s.blit(self.surface.subsurface(r),r2)
                    rets.append(r2)
        else:
            s.blit(self.bkgr,(0,0))
            sub = pygame.Rect(self.offset[0],self.offset[1],min(s.get_width(),self.max_rect.w-self.offset[0]),min(s.get_height(),self.max_rect.h-self.offset[1]))
#             print sub
#             print self.surface.get_width(),self.surface.get_height()
#             print s.get_width(),s.get_height()
#             print self.offset
#             print self.style.width,self.style.height
            s.blit(self.surface.subsurface(sub),(0,0))
            rets.append(s_rect)
        self._offset = self.offset[:]
        return rets
        
    def proxy_update(self, s):
        rects = container.Container.update(self, surface.ProxySurface(parent=None, 
                                                    rect=self.max_rect,
                                                    real_surface=s,
                                                    offset=self.offset))
        result = []
        for r in rects: result.append(pygame.Rect(r).move(self.offset))
        return result
        
    def resize(self, width=None, height=None):
        container.Container.resize(self)
        self.max_rect = pygame.Rect(self.widget.rect)
        #self.max_rect.w = max(self.max_rect.w,self.style.width)
        #self.max_rect.h = max(self.max_rect.h,self.style.height)
        return self.style.width,self.style.height
        #self.rect = pygame.Rect(self.rect[0], self.rect[1], self.style.width, self.style.height)
    
    def event(self, e):
        if e.type in [MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION]:
            pos = (e.pos[0] + self.offset[0], e.pos[1] + self.offset[1])
            if self.max_rect.collidepoint(pos):
                e_params = {'pos': pos }
                if e.type == MOUSEMOTION: 
                    e_params['buttons'] = e.buttons
                    e_params['rel'] = e.rel
                else:
                    e_params['button'] = e.button
                e = pygame.event.Event(e.type, e_params)
        container.Container.event(self, e)

#class SlideBox(Area):
#    def __init__(self,*args,**params):
#        print 'gui.SlideBox','Scheduled to be renamed to Area.'
#        Area.__init__(self,*args,**params)
    
class ScrollArea(table.Table):
    """A scrollable area with scrollbars."""

    _widget = None

    def __init__(self, widget, width=0, height=0, hscrollbar=True, vscrollbar=True,step=24, **params):
        """ScrollArea constructor.

        Arguments:
            widget -- widget to be able to scroll around
            width, height -- size of scrollable area.  Set either to 0 to default to size of widget.
            hscrollbar -- set to False if you do not wish to have a horizontal scrollbar
            vscrollbar -- set to False if you do not wish to have a vertical scrollbar
            step -- set to how far clicks on the icons will step 

        """
        w= widget
        params.setdefault('cls', 'scrollarea')
        table.Table.__init__(self, width=width,height=height,**params)
        
        self.sbox = SlideBox(w, width=width, height=height, cls=self.cls+".content")
        self.widget = w
        self.vscrollbar = vscrollbar
        self.hscrollbar = hscrollbar
        
        self.step = step

    @property
    def widget(self):
        return self._widget

    @widget.setter
    def widget(self, val):
        self._widget = val
        self.sbox.widget = val

    def resize(self,width=None,height=None):
        widget = self.widget
        box = self.sbox
        
        #self.clear()
        table.Table.clear(self)
        #print 'resize',self,self._rows
        
        self.tr()
        self.td(box)
        
        widget.rect.w, widget.rect.h = widget.resize()
        my_width,my_height = self.style.width,self.style.height
        if not my_width:
            my_width = widget.rect.w
            self.hscrollbar = False
        if not my_height:
            my_height = widget.rect.h
            self.vscrollbar = False
        
        box.style.width,box.style.height = my_width,my_height #self.style.width,self.style.height
        
        box.rect.w,box.rect.h = box.resize()
        
        #print widget.rect
        #print box.rect
        #r = table.Table.resize(self,width,height)
        #print r
        #return r
        
        #print box.offset
        
#         #this old code automatically adds in a scrollbar if needed
#         #but it doesn't always work
#         self.vscrollbar = None
#         if widget.rect.h > box.rect.h:
#             self.vscrollbar = slider.VScrollBar(box.offset[1],0, 65535, 0,step=self.step) 
#             self.td(self.vscrollbar)
#             self.vscrollbar.connect(CHANGE, self._vscrollbar_changed, None)
#             
#             vs = self.vscrollbar
#             vs.rect.w,vs.rect.h = vs.resize()
#             box.style.width = self.style.width - vs.rect.w
#             
#             
#         self.hscrollbar = None
#         if widget.rect.w > box.rect.w:
#             self.hscrollbar = slider.HScrollBar(box.offset[0], 0,65535, 0,step=self.step)
#             self.hscrollbar.connect(CHANGE, self._hscrollbar_changed, None)
#             self.tr()
#             self.td(self.hscrollbar)
#             
#             hs = self.hscrollbar
#             hs.rect.w,hs.rect.h = hs.resize()
#             box.style.height = self.style.height - hs.rect.h

        xt,xr,xb,xl  = pguglobals.app.theme.getspacing(box)
        

        if self.vscrollbar:
            self.vscrollbar = slider.VScrollBar(box.offset[1],0, 65535, 0,step=self.step) 
            self.td(self.vscrollbar)
            self.vscrollbar.connect(CHANGE, self._vscrollbar_changed, None)
            
            vs = self.vscrollbar
            vs.rect.w,vs.rect.h = vs.resize()
            if self.style.width:
                box.style.width = self.style.width - (vs.rect.w + xl+xr)

        if self.hscrollbar:
            self.hscrollbar = slider.HScrollBar(box.offset[0], 0,65535, 0,step=self.step)
            self.hscrollbar.connect(CHANGE, self._hscrollbar_changed, None)
            self.tr()
            self.td(self.hscrollbar)
            
            hs = self.hscrollbar
            hs.rect.w,hs.rect.h = hs.resize()
            if self.style.height:
                box.style.height = self.style.height - (hs.rect.h + xt + xb)
            
        if self.hscrollbar:
            hs = self.hscrollbar
            hs.min = 0
            hs.max = widget.rect.w - box.style.width
            hs.style.width = box.style.width
            hs.size = hs.style.width * box.style.width / max(1,widget.rect.w)
        else:
            box.offset[0] = 0
            
        if self.vscrollbar:
            vs = self.vscrollbar
            vs.min = 0
            vs.max = widget.rect.h - box.style.height
            vs.style.height = box.style.height
            vs.size = vs.style.height * box.style.height / max(1,widget.rect.h)
        else:
            box.offset[1] = 0
            
        #print self.style.width,box.style.width, hs.style.width
            
        r = table.Table.resize(self,width,height)
        return r
    
    def x_resize(self, width=None, height=None):
        w,h = table.Table.resize(self, width, height)
        if self.hscrollbar:
            if self.widget.rect.w <= self.sbox.rect.w:
                self.hscrollbar.size = self.hscrollbar.style.width
            else:
                self.hscrollbar.size = max(20,self.hscrollbar.style.width * self.sbox.rect.w / self.widget.rect.w)
            self._hscrollbar_changed(None)
        if self.widget.rect.h <= self.sbox.rect.h: 
            self.vscrollbar.size = self.vscrollbar.style.height
        else:
            self.vscrollbar.size = max(20,self.vscrollbar.style.height * self.sbox.rect.h / self.widget.rect.h)
        self._vscrollbar_changed(None)
        return w,h

    def _vscrollbar_changed(self, xxx):
        #y = (self.widget.rect.h - self.sbox.rect.h) * self.vscrollbar.value / 1000
        #if y >= 0: self.sbox.offset[1] = -y
        self.sbox.offset[1] = self.vscrollbar.value
        self.sbox.reupdate()

    def _hscrollbar_changed(self, xxx):
        #x = (self.widget.rect.w - self.sbox.rect.w) * self.hscrollbar.value / 1000
        #if x >= 0: self.sbox.offset[0] = -x
        self.sbox.offset[0] = self.hscrollbar.value
        self.sbox.reupdate()


    def set_vertical_scroll(self, percents): 
        #if not self.vscrollbar: return
        if not hasattr(self.vscrollbar,'value'): return
        self.vscrollbar.value = percents #min(max(percents*10, 0), 1000)
        self._vscrollbar_changed(None)

    def set_horizontal_scroll(self, percents): 
        #if not self.hscrollbar: return
        if not hasattr(self.hscrollbar,'value'): return
        self.hscrollbar.value = percents #min(max(percents*10, 0), 1000)
        self._hscrollbar_changed(None)

    def event(self, e):
         #checking for event recipient
         if (table.Table.event(self, e)):
             return True

         #mouse wheel scrolling
         if self.vscrollbar:
             if not hasattr(self.vscrollbar,'value'): 
                 return False
 
             if e.type == pygame.locals.MOUSEBUTTONDOWN:
                 if e.button == 4: #wheel up
                     self.vscrollbar._click(-1)
                     return True
                 elif e.button == 5: #wheel down
                     self.vscrollbar._click(1)
                     return True
         return False

        


class _List_Item(button._button):
    def __init__(self,label=None,image=None,value=None,**params): #TODO label= could conflict with the module label
        #param image: an imagez.Image object (optional)
        #param text: a string object
        params.setdefault('cls','list.item')
        button._button.__init__(self,**params)
        self.group = None
        self.value = value #(self, value)
        self.widget = None
        
        if type(label) == str: 
            label = basic.Label(label, cls=self.cls+".label")

        if image and label:
            self.widget = container.Container()
            self.widget.add(image, 0, 0)
            #HACK: improper use of .resize()
            image.rect.w,image.rect.h = image.resize()
            self.widget.add(label, image.rect.w, 0)
        elif image: self.widget = image
        elif label: self.widget = label
        
        self.pcls = ""
    
    def resize(self,width=None,height=None):
         self.widget.rect.w,self.widget.rect.h = self.widget.resize()
         return self.widget.rect.w,self.widget.rect.h
#         self.widget._resize()
#         self.rect.w,self.rect.h = self.widget.rect_margin.w,self.widget.rect_margin.h
    
    def event(self,e):
        button._button.event(self,e)
        if self.group.value == self.value: self.pcls = "down"
    
    def paint(self,s):
        if self.group.value == self.value: self.pcls = "down"
        self.widget.paint(surface.subsurface(s,self.widget.rect))
    
    def click(self):
        self.group.value = self.value
        for w in self.group.widgets:
            if w != self: w.pcls = ""



class List(ScrollArea):
    """A list of items in an area.
    
    This widget can be a form element, it has a value set to whatever item is selected.

    """
    def _change(self, value):
        self.value = self.group.value
        self.send(CHANGE)

    def __init__(self, width, height, **params):
        params.setdefault('cls', 'list')
        self.table = table.Table(width=width)
        ScrollArea.__init__(self, self.table, width, height,hscrollbar=False ,**params)
        
        self.items = []
        
        g = group.Group()
        self.group = g
        g.connect(CHANGE,self._change,None)
        self.value = self.group.value = None

        self.add = self._add
        self.remove = self._remove
        
    def clear(self):
        """Clear the list."""
        self.items = []
        self.group = group.Group()
        self.group.connect(CHANGE,self._change,None)
        self.table.clear()
        self.set_vertical_scroll(0)
        self.blur(self.myfocus)
        
    def _add(self, label, image = None, value=None):
        item = _List_Item(label,image=image,value=value)
        self.table.tr()
        self.table.add(item)
        self.items.append(item)
        item.group = self.group
        item.group.add(item)
        
    def _remove(self, item):
        for i in self.items:
            if i.value == item: item = i
        if item not in self.items: 
            return
        item.blur()
        self.items.remove(item)
        self.group.widgets.remove(item)
        self.table.remove_row(item.style.row)

