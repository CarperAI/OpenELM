# theme.py

"""
"""
import os, re
import pygame

from .const import *
from . import widget
from . import surface
from .basic import parse_color, is_color

__file__ = os.path.abspath(__file__)

def _list_themes(dir):
    d = {}
    for entry in os.listdir(dir):
        if os.path.exists(os.path.join(dir, entry, 'config.txt')):
            d[entry] = os.path.join(dir, entry)
    return d

class Theme:
    """Theme interface.
    
    If you wish to create your own theme, create a class with this interface, and 
    pass it to gui.App via gui.App(theme=MyTheme()).
    
    """
    def __init__(self,dirs='default'):
        """Theme constructor.

        Keyword arguments:
            dirs -- Name of the theme dir to load a theme from.  May be an 
                absolute path to a theme, if pgu is not installed, or if you 
                created your own theme.  May include several dirs in a list if 
                data is spread across several themes.
        
        Example:
            theme = gui.Theme("default")
            theme = gui.Theme(["mytheme","mytheme2"])

        """
        self.config = {}
        self._loaded = []
        self.cache = {}
        self._preload(dirs)
        pygame.font.init()
    
    def _preload(self,ds):
        if not isinstance(ds, list):
            ds = [ds]
        for d in ds:
            if d not in self._loaded:
                self._load(d)
            self._loaded.append(d)
    
    def _load(self, name):
        #theme_dir = themes[name]
        
        #try to load the local dir, or absolute path
        dnames = [name]
        
        #if the package isn't installed and people are just
        #trying out the scripts or examples
        dnames.append(os.path.join(os.path.dirname(__file__),"..","..","data","themes",name))
        
        #if the package is installed, and the package is installed
        #in /usr/lib/python2.3/site-packages/pgu/
        #or c:\python23\lib\site-packages\pgu\
        #the data is in ... lib/../share/ ...
        dnames.append(os.path.join(os.path.dirname(__file__),"..","..","..","..","share","pgu","themes",name))
        dnames.append(os.path.join(os.path.dirname(__file__),"..","..","..","..","..","share","pgu","themes",name))
        dnames.append(os.path.join(os.path.dirname(__file__),"..","..","share","pgu","themes",name)) 
        for dname in dnames:
            if os.path.isdir(dname): break
        if not os.path.isdir(dname): 
            raise Exception('could not find theme '+name)
            
        fname = os.path.join(dname,"config.txt")
        if os.path.isfile(fname):
            try:
                f = open(fname)
                for line in f.readlines():
                    args = line.strip().split()

                    if len(args) < 3:
                        continue

                    pcls = ""
                    (cls, attr, vals) = (args[0], args[1], args[2:])
                    if (":" in cls):
                        (cls, pcls) = cls.split(":")

                    self.config[cls, pcls, attr] = (dname, vals)
            finally:
                f.close()
        fname = os.path.join(dname,"style.ini")
        if os.path.isfile(fname):
            import ConfigParser
            cfg = ConfigParser.ConfigParser()
            f = open(fname,'r')
            cfg.readfp(f)
            for section in cfg.sections():
                cls = section
                pcls = ''
                if cls.find(":")>=0:
                    cls,pcls = cls.split(":")
                for attr in cfg.options(section):
                    vals = cfg.get(section,attr).strip().split()
                    self.config[cls,pcls,attr] = (dname, vals)
    
    image_extensions = (".gif", ".jpg", ".bmp", ".png", ".tga")
    def _get(self, cls, pcls, attr):
        key = (cls, pcls, attr)
        if not key in self.config:
            return

        if key in self.cache:
            # This property is already in the cache
            return self.cache[key]

        (dname, vals) = self.config[key]

        if (os.path.splitext(vals[0].lower())[1] in self.image_extensions):
            # This is an image attribute
            v = pygame.image.load(os.path.join(dname, vals[0]))

        elif (attr == "color" or attr == "background"):
            # This is a color value
            v = parse_color(vals[0])

        elif (attr == "font"):
            # This is a font value
            name = vals[0]
            size = int(vals[1])
            if (name.endswith(".ttf")):
                # Load the font from a file
                v = pygame.font.Font(os.path.join(dname, name), size)
            else:
                # Must be a system font
                v = pygame.font.SysFont(name, size)

        else:
            try:
                v = int(vals[0])
            except:
                v = vals[0]
        self.cache[key] = v
        return v

    def get(self,cls,pcls,attr):
        """Interface method -- get the value of a style attribute.
        
        Arguments:
            cls -- class, for example "checkbox", "button", etc.
            pcls -- pseudo class, for example "hover", "down", etc.
            attr -- attribute, for example "image", "background", "font", "color", etc.
        
        This method is called from [[gui-style]]

        """

        if not self._loaded: 
            # Load the default theme
            self._preload("default")

        o = (cls, pcls, attr)
        
        #if o in self.cache: 
        #    return self.cache[o]

        v = self._get(cls, pcls, attr)
        if v: 
            #self.cache[o] = v
            return v
        
        v = self._get(cls, "", attr)
        if v: 
            #self.cache[o] = v
            return v
        
        v = self._get("default", "", attr)
        if v: 
            #self.cache[o] = v
            return v
        
        self.cache[o] = 0
        return 0

    def box(self,w,s):
        style = w.style
        
        c = (0,0,0)
        if style.border_color != 0: c = style.border_color
        w,h = s.get_width(),s.get_height()
        
        s.fill(c,(0,0,w,style.border_top))
        s.fill(c,(0,h-style.border_bottom,w,style.border_bottom))
        s.fill(c,(0,0,style.border_left,h))
        s.fill(c,(w-style.border_right,0,style.border_right,h))
        
        
    def getspacing(self,w):
        # return the top, right, bottom, left spacing around the widget
        if not hasattr(w,'_spacing'): #HACK: assume spacing doesn't change re pcls
            s = w.style
            xt = s.margin_top+s.border_top+s.padding_top
            xr = s.padding_right+s.border_right+s.margin_right
            xb = s.padding_bottom+s.border_bottom+s.margin_bottom
            xl = s.margin_left+s.border_left+s.padding_left
            w._spacing = xt,xr,xb,xl
        return w._spacing

        
    def resize(self,w,m):
        # Returns the rectangle expanded in each direction
        def expand_rect(rect, left, top, right, bottom):
            return pygame.Rect(rect.x - left, 
                               rect.y - top, 
                               rect.w + left + right, 
                               rect.h + top + bottom)

        def func(width=None,height=None):
            s = w.style
            
            pt,pr,pb,pl = (s.padding_top,s.padding_right,
                           s.padding_bottom,s.padding_left)
            bt,br,bb,bl = (s.border_top,s.border_right,
                           s.border_bottom,s.border_left)
            mt,mr,mb,ml = (s.margin_top,s.margin_right,
                           s.margin_bottom,s.margin_left)
            # Calculate the total space on each side
            top = pt+bt+mt
            right = pr+br+mr
            bottom = pb+bb+mb
            left = pl+bl+ml
            ttw = left+right
            tth = top+bottom
            
            ww,hh = None,None
            if width != None: ww = width-ttw
            if height != None: hh = height-tth
            ww,hh = m(ww,hh)

            if width == None: width = ww
            if height == None: height = hh
            
            #if the widget hasn't respected the style.width,
            #style height, we'll add in the space for it...
            width = max(width-ttw, ww, w.style.width)
            height = max(height-tth, hh, w.style.height)
            
            #width = max(ww,w.style.width-tw)
            #height = max(hh,w.style.height-th)

            r = pygame.Rect(left,top,width,height)
            
            w._rect_padding = expand_rect(r, pl, pt, pr, pb)
            w._rect_border = expand_rect(w._rect_padding, bl, bt, br, bb)
            w._rect_margin = expand_rect(w._rect_border, ml, mt, mr, mb)

            # align it within it's zone of power.   
            rect = pygame.Rect(left, top, ww, hh)
            dx = width-rect.w
            dy = height-rect.h
            rect.x += (w.style.align+1)*dx/2
            rect.y += (w.style.valign+1)*dy/2

            w._rect_content = rect

            return (w._rect_margin.w, w._rect_margin.h)
        return func


    def paint(self,w,m):
        def func(s):
#             if w.disabled:
#                 if not hasattr(w,'_disabled_bkgr'):
#                     w._disabled_bkgr = s.convert()
#                 orig = s
#                 s = w._disabled_bkgr.convert()

#             if not hasattr(w,'_theme_paint_bkgr'):
#                 w._theme_paint_bkgr = s.convert()
#             else:
#                 s.blit(w._theme_paint_bkgr,(0,0))
#             
#             if w.disabled:
#                 orig = s
#                 s = w._theme_paint_bkgr.convert()

            if w.disabled:
                if (not (hasattr(w,'_theme_bkgr') and 
                         w._theme_bkgr.get_width() == s.get_width() and 
                         w._theme_bkgr.get_height() == s.get_height())):
                    w._theme_bkgr = s.copy()
                orig = s
                s = w._theme_bkgr
                s.fill((0,0,0,0))
                s.blit(orig,(0,0))
                
            if w.background:
                w.background.paint(surface.subsurface(s,w._rect_border))

            self.box(w,surface.subsurface(s,w._rect_border))
            r = m(surface.subsurface(s,w._rect_content))
            
            if w.disabled:
                s.set_alpha(128)
                orig.blit(s,(0,0))
            
#             if w.disabled:
#                 orig.blit(w._disabled_bkgr,(0,0))
#                 s.set_alpha(128)
#                 orig.blit(s,(0,0))
            
            w._painted = True
            return r
        return func
    
    def event(self,w,m):
        def func(e):
            rect = w._rect_content
            if (not rect):
                # This should never be the case, but it sometimes happens that _rect_content isn't
                # set before a mouse event is received. In this case we'll ignore the event.
                return m(e)

            if e.type == MOUSEBUTTONUP or e.type == MOUSEBUTTONDOWN:
                sub = pygame.event.Event(e.type,{
                    'button':e.button,
                    'pos':(e.pos[0]-rect.x,e.pos[1]-rect.y)})
            elif e.type == CLICK:
                sub = pygame.event.Event(e.type,{
                    'button':e.button,
                    'pos':(e.pos[0]-rect.x,e.pos[1]-rect.y)})
            elif e.type == MOUSEMOTION:
                sub = pygame.event.Event(e.type,{
                    'buttons':e.buttons,
                    'pos':(e.pos[0]-rect.x,e.pos[1]-rect.y),
                    'rel':e.rel})
            else:
                sub = e
            return m(sub)

        return func
    
    def update(self,w,m):
        def func(s):
            if w.disabled: return []
            r = m(surface.subsurface(s,w._rect_content))
            if type(r) == list:
                dx,dy = w._rect_content.topleft
                for rr in r:
                    rr.x,rr.y = rr.x+dx,rr.y+dy
            return r
        return func
        
    def open(self,w,m):
        def func(widget=None,x=None,y=None):
            if not hasattr(w,'_rect_content'):
                # HACK: so that container.open won't resize again!
                w.rect.w,w.rect.h = w.resize()
            rect = w._rect_content
            ##print w.__class__.__name__, rect
            if x != None: x += rect.x
            if y != None: y += rect.y
            return m(widget,x,y)
        return func
            
    #def open(self,w,m):
    #    def func(widget=None):
    #        return m(widget)
    #    return func
        
    def decorate(self,widget,level):
        """Interface method -- decorate a widget.
        
        The theme system is given the opportunity to decorate a widget 
        methods at the end of the Widget initializer.

        Arguments:
            widget -- the widget to be decorated
            level -- the amount of decoration to do, False for none, True for 
                normal amount, 'app' for special treatment of App objects.
        
        """        

        w = widget
        if level == False: return
        
        if type(w.style.background) != int:
            w.background = Background(w,self)    
        
        if level == 'app': return
        
        for k,v in list(w.style.__dict__.items()):
            if k in ('border','margin','padding'):
                for kk in ('top','bottom','left','right'):
                    setattr(w.style,'%s_%s'%(k,kk),v)

        w.paint = self.paint(w,w.paint)
        w.event = self.event(w,w.event)
        w.update = self.update(w,w.update)
        w.resize = self.resize(w,w.resize)
        w.open = self.open(w,w.open)

    def render(self,s,box,r):
        """Interface method - render a special widget feature.

        Arguments:
            s -- a pygame surface
            box -- box data, a value returned from Theme.get, typically a surface
            r -- pygame.Rect with the size that the box data should be rendered
        
        """
        
        if box == 0: return
        
        if is_color(box):
            s.fill(box,r)
            return
        
        x,y,w,h=r.x,r.y,r.w,r.h
        ww,hh=int(box.get_width()/3),int(box.get_height()/3)
        xx,yy=x+w,y+h
        src = pygame.rect.Rect(0,0,ww,hh)
        dest = pygame.rect.Rect(0,0,ww,hh)
        
        s.set_clip(pygame.Rect(x+ww,y+hh,w-ww*2,h-hh*2))
        src.x,src.y = ww,hh
        for dest.y in range(y+hh,yy-hh,hh):
            for dest.x in range(x+ww,xx-ww,ww): s.blit(box,dest,src)
        
        s.set_clip(pygame.Rect(x+ww,y,w-ww*3,hh))
        src.x,src.y,dest.y = ww,0,y
        for dest.x in range(x+ww,xx-ww*2,ww): s.blit(box,dest,src)
        dest.x = xx-ww*2
        s.set_clip(pygame.Rect(x+ww,y,w-ww*2,hh))
        s.blit(box,dest,src)
        
        s.set_clip(pygame.Rect(x+ww,yy-hh,w-ww*3,hh))
        src.x,src.y,dest.y = ww,hh*2,yy-hh
        for dest.x in range(x+ww,xx-ww*2,ww): s.blit(box,dest,src)
        dest.x = xx-ww*2
        s.set_clip(pygame.Rect(x+ww,yy-hh,w-ww*2,hh))
        s.blit(box,dest,src)
    
        s.set_clip(pygame.Rect(x,y+hh,xx,h-hh*3))
        src.y,src.x,dest.x = hh,0,x
        for dest.y in range(y+hh,yy-hh*2,hh): s.blit(box,dest,src)
        dest.y = yy-hh*2
        s.set_clip(pygame.Rect(x,y+hh,xx,h-hh*2))
        s.blit(box,dest,src)
    
        s.set_clip(pygame.Rect(xx-ww,y+hh,xx,h-hh*3))
        src.y,src.x,dest.x=hh,ww*2,xx-ww
        for dest.y in range(y+hh,yy-hh*2,hh): s.blit(box,dest,src)
        dest.y = yy-hh*2
        s.set_clip(pygame.Rect(xx-ww,y+hh,xx,h-hh*2))
        s.blit(box,dest,src)
       	
        s.set_clip(s.get_rect())
        src.x,src.y,dest.x,dest.y = 0,0,x,y
        s.blit(box,dest,src)
        
        src.x,src.y,dest.x,dest.y = ww*2,0,xx-ww,y
        s.blit(box,dest,src)
        
        src.x,src.y,dest.x,dest.y = 0,hh*2,x,yy-hh
        s.blit(box,dest,src)
        
        src.x,src.y,dest.x,dest.y = ww*2,hh*2,xx-ww,yy-hh
        s.blit(box,dest,src)

        
class Background(widget.Widget):
    def __init__(self,value,theme,**params):
        params['decorate'] = False
        widget.Widget.__init__(self,**params)
        self.value = value
        self.theme = theme
    
    def paint(self,s):
        r = pygame.Rect(0,0,s.get_width(),s.get_height())
        v = self.value.style.background
        if is_color(v):
            s.fill(v)
        else: 
            self.theme.render(s,v,r)

