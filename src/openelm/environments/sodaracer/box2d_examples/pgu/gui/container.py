"""
"""
import pygame
from pygame.locals import *

from .const import *
from . import widget, surface
from . import pguglobals

class Container(widget.Widget):
    """The base container widget, can be used as a template as well as stand alone."""

    def __init__(self,**params):
        widget.Widget.__init__(self,**params)
        self.myfocus = None
        self.mywindow = None
        self.myhover = None
        #self.background = 0
        self.widgets = []
        self.windows = []
        self.toupdate = {}
        self.topaint = {}
    
    def update(self,s):
        updates = []
        
        if self.myfocus: self.toupdate[self.myfocus] = self.myfocus
        
        for w in self.topaint:
            if w is self.mywindow:
                continue
            else:
                sub = surface.subsurface(s,w.rect)
                #if (hasattr(w, "_container_bkgr")):
                #    sub.blit(w._container_bkgr,(0,0))
                w.paint(sub)
                updates.append(pygame.rect.Rect(w.rect))
        
        for w in self.toupdate:
            if w is self.mywindow:
                continue
            else:            
                us = w.update(surface.subsurface(s,w.rect))
            if us:
                for u in us:
                    updates.append(pygame.rect.Rect(u.x + w.rect.x,u.y+w.rect.y,u.w,u.h))
        
        for w in self.topaint:
            if w is self.mywindow:
                w.paint(self.top_surface(s,w))
                updates.append(pygame.rect.Rect(w.rect))
            else:
                continue 
        
        for w in self.toupdate:
            if w is self.mywindow:
                us = w.update(self.top_surface(s,w))
            else:            
                continue 
            if us:
                for u in us:
                    updates.append(pygame.rect.Rect(u.x + w.rect.x,u.y+w.rect.y,u.w,u.h))
        
        self.topaint = {}
        self.toupdate = {}
        
        return updates
    
    def repaint(self,w=None):
        if not w:
            return widget.Widget.repaint(self)
        self.topaint[w] = w
        self.reupdate()
    
    def reupdate(self,w=None):
        if not w:
            return widget.Widget.reupdate(self)
        self.toupdate[w] = w
        self.reupdate()
    
    def paint(self,s):
        self.toupdate = {}
        self.topaint = {}
        for w in self.widgets:
            try:
                sub = surface.subsurface(s, w.rect)
            except: 
                print('container.paint(): %s not inside %s' % (
                    w.__class__.__name__,self.__class__.__name__))
                print(s.get_width(), s.get_height(), w.rect)
                print("")
            else:
#                if (not hasattr(w,'_container_bkgr') or 
#                    w._container_bkgr.get_size() != sub.get_size()):
#                         #w._container_bkgr.get_width() == sub.get_width() and 
#                         #w._container_bkgr.get_height() == sub.get_height())):
#                    w._container_bkgr = sub.copy()
#                w._container_bkgr.fill((0,0,0,0))
#                w._container_bkgr.blit(sub,(0,0))
                w.paint(sub)

        for w in self.windows:
            w.paint(self.top_surface(s,w))
    
    def top_surface(self,s,w):
        x,y = s.get_abs_offset()
        s = s.get_abs_parent()
        return surface.subsurface(s,(x+w.rect.x,y+w.rect.y,w.rect.w,w.rect.h))
    
    def event(self,e):
        used = False
        
        if self.mywindow and e.type == MOUSEBUTTONDOWN:
            w = self.mywindow
            if self.myfocus is w:
                if not w.collidepoint(e.pos): self.blur(w)
            if not self.myfocus:
                if w.collidepoint(e.pos): self.focus(w)
        
        if not self.mywindow:
            #### by Gal Koren
            ##
            ## if e.type == FOCUS:
            if e.type == FOCUS and not self.myfocus:
                #self.first()
                pass
            elif e.type == EXIT:
                if self.myhover: self.exit(self.myhover)
            elif e.type == BLUR:
                if self.myfocus: self.blur(self.myfocus)
            elif e.type == MOUSEBUTTONDOWN:
                h = None
                for w in self.widgets:
                    if not w.disabled: 
                        # Focusable not considered, since that is only for tabs
                        if w.collidepoint(e.pos):
                            h = w
                            if self.myfocus is not w: self.focus(w)
                if not h and self.myfocus:
                    self.blur(self.myfocus)
            elif e.type == MOUSEMOTION:
                if 1 in e.buttons:
                    if self.myfocus: ws = [self.myfocus]
                    else: ws = []
                else: ws = self.widgets
                
                h = None
                for w in ws:
                    if w.collidepoint(e.pos):
                        h = w
                        if self.myhover is not w: 
                            self.enter(w)
                        break
                if not h and self.myhover:
                    self.exit(self.myhover)
                w = self.myhover
                
                if w and w is not self.myfocus:
                    sub = pygame.event.Event(e.type,{
                        'buttons':e.buttons,
                        'pos':(e.pos[0]-w.rect.x,e.pos[1]-w.rect.y),
                        'rel':e.rel})
                    used = w._event(sub)
        
        w = self.myfocus
        if w:
            if e.type == MOUSEBUTTONUP or e.type == MOUSEBUTTONDOWN:
                sub = pygame.event.Event(e.type,{
                    'button':e.button,
                    'pos':(e.pos[0]-w.rect.x,e.pos[1]-w.rect.y)})
            elif e.type == CLICK and self.myhover is w:
                sub = pygame.event.Event(e.type,{
                    'button':e.button,
                    'pos':(e.pos[0]-w.rect.x,e.pos[1]-w.rect.y)})
            elif e.type == MOUSEMOTION:
                sub = pygame.event.Event(e.type,{
                    'buttons':e.buttons,
                    'pos':(e.pos[0]-w.rect.x,e.pos[1]-w.rect.y),
                    'rel':e.rel})
            elif (e.type == KEYDOWN or e.type == KEYUP):
                sub = e
            else:
                sub = None

            #elif e.type == CLICK: #a dead click
            #    sub = None

            if (sub):
                used = w._event(sub)

        if not used and e.type is KEYDOWN:
            if e.key is K_TAB and self.myfocus:
                if (e.mod&KMOD_SHIFT) == 0:
                    self.myfocus.next()
                else:
                    self.myfocus.previous()
                    return True
            elif e.key == K_UP: 
                self._move_focus(0,-1)
                return True
            elif e.key == K_RIGHT:
                self._move_focus(1,0)
                return True
            elif e.key == K_DOWN:
                self._move_focus(0,1)
                return True
            elif e.key == K_LEFT:
                self._move_focus(-1,0)
                return True
        return used

    def _move_focus(self,dx_,dy_):
        myfocus = self.myfocus
        if not self.myfocus: return
        
        widgets = self._get_widgets(pguglobals.app)
        #if myfocus not in widgets: return
        #widgets.remove(myfocus)
        if myfocus in widgets:
            widgets.remove(myfocus)
        rect = myfocus.get_abs_rect()
        fx,fy = rect.centerx,rect.centery
        
        def sign(v):
            if v < 0: return -1
            if v > 0: return 1
            return 0
        
        dist = []
        for w in widgets:
            wrect = w.get_abs_rect()
            wx,wy = wrect.centerx,wrect.centery
            dx,dy = wx-fx,wy-fy
            if dx_ > 0 and wrect.left < rect.right: continue
            if dx_ < 0 and wrect.right > rect.left: continue
            if dy_ > 0 and wrect.top < rect.bottom: continue
            if dy_ < 0 and wrect.bottom > rect.top: continue
            dist.append((dx*dx+dy*dy,w))
        if not len(dist): return
        dist.sort()
        d,w = dist.pop(0)
        w.focus()
        
    def _get_widgets(self,c):
        widgets = []
        if c.mywindow:
            widgets.extend(self._get_widgets(c.mywindow))
        else:
            for w in c.widgets:
                if isinstance(w,Container):
                    widgets.extend(self._get_widgets(w))
                elif not w.disabled and w.focusable:
                    widgets.append(w)
        return widgets

    def remove(self,w):
        """Remove a widget from the container."""
        self.blur(w)
        self.widgets.remove(w)
        #self.repaint()
        self.chsize()
    
    def add(self,w,x,y):
        """Add a widget to the container given the position."""
        w.style.x = x
        w.style.y = y 
        w.container = self
        #NOTE: this might fix it, sort of...
        #but the thing is, we don't really want to resize
        #something if it is going to get resized again later
        #for no reason...
        #w.rect.x,w.rect.y = w.style.x,w.style.y
        #w.rect.w, w.rect.h = w.resize()
        self.widgets.append(w)
        self.chsize()
    
    def open(self,w=None,x=None,y=None):
        if (not w):
            w = self

        if (x != None):
            # The position is relative to this container
            rect = self.get_abs_rect()
            pos = (rect.x + x, rect.y + y)
        else:
            pos = None
        # Have the application open the window
        pguglobals.app.open(w, pos)
        
    def focus(self,w=None):
        widget.Widget.focus(self) ### by Gal koren
#        if not w:
#            return widget.Widget.focus(self)
        if not w: return
        if self.myfocus: self.blur(self.myfocus)
        if self.myhover is not w: self.enter(w)
        self.myfocus = w
        w._event(pygame.event.Event(FOCUS))
        
        #print self.myfocus,self.myfocus.__class__.__name__
    
    def blur(self,w=None):
        if not w:
            return widget.Widget.blur(self)
        if self.myfocus is w:
            if self.myhover is w: self.exit(w)
            self.myfocus = None
            w._event(pygame.event.Event(BLUR))
    
    def enter(self,w):
        if self.myhover: self.exit(self.myhover)
        self.myhover = w
        w._event(pygame.event.Event(ENTER))
    
    def exit(self,w):
        if self.myhover and self.myhover is w:
            self.myhover = None
            w._event(pygame.event.Event(EXIT))    
    
    
#     def first(self):
#         for w in self.widgets:
#             if w.focusable:
#                 self.focus(w)
#                 return
#         if self.container: self.container.next(self)
    
#     def next(self,w):
#         if w not in self.widgets: return #HACK: maybe.  this happens in windows for some reason...
#         
#         for w in self.widgets[self.widgets.index(w)+1:]:
#             if w.focusable:
#                 self.focus(w)
#                 return
#         if self.container: return self.container.next(self)
    
    
    def _next(self,orig=None):
        start = 0
        if orig in self.widgets: start = self.widgets.index(orig)+1
        for w in self.widgets[start:]:
            if not w.disabled and w.focusable:
                if isinstance(w,Container):
                    if w._next():
                        return True
                else:
                    self.focus(w)
                    return True
        return False
    
    def _previous(self,orig=None):
        end = len(self.widgets)
        if orig in self.widgets: end = self.widgets.index(orig)
        ws = self.widgets[:end]
        ws.reverse()
        for w in ws:
            if not w.disabled and w.focusable:
                if isinstance(w,Container):
                    if w._previous():
                        return True
                else:
                    self.focus(w)
                    return True
        return False
                
    def next(self,w=None):
        if w != None and w not in self.widgets: return #HACK: maybe.  this happens in windows for some reason...
        
        if self._next(w): return True
        if self.container: return self.container.next(self)
    
    
    def previous(self,w=None):
        if w != None and w not in self.widgets: return #HACK: maybe.  this happens in windows for some reason...
        
        if self._previous(w): return True
        if self.container: return self.container.previous(self)
    
    def resize(self,width=None,height=None):
        #r = self.rect
        #r.w,r.h = 0,0
        ww,hh = 0,0
        if self.style.width: ww = self.style.width
        if self.style.height: hh = self.style.height
        
        for w in self.widgets:
            #w.rect.w,w.rect.h = 0,0
            w.rect.x,w.rect.y = w.style.x,w.style.y
            w.rect.w, w.rect.h = w.resize()
            #w._resize()
            
            ww = max(ww,w.rect.right)
            hh = max(hh,w.rect.bottom)
        return ww,hh

    # Returns the widget with the given name
    def find(self, name):
        for w in self.widgets:
            if (w.name == name):
                return w
            elif (isinstance(w, Container)):
                tmp = w.find(name)
                if (tmp): return tmp
        return None

