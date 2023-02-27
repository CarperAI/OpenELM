"""This modules defines the Widget class, which is the base of the PGU widget
hierarchy."""

import pygame

from . import pguglobals
from . import style

class SignalCallback:
    # The function to call
    func = None
    # The parameters to pass to the function (as a list)
    params = None

class Widget(object):
    """Base class for all PGU graphical objects.
        
    Example - Creating your own Widget:

        class Draw(gui.Widget):
            def paint(self,s):
                # Paint the pygame.Surface
                return
            
            def update(self,s):
                # Update the pygame.Surface and return the update rects
                return [pygame.Rect(0,0,self.rect.w,self.rect.h)]
                
            def event(self,e):
                # Handle the pygame.Event
                return
                
            def resize(self,width=None,height=None):
                # Return the width and height of this widget
                return 256,256
    """

    # The name of the widget (or None if not defined)
    name = None
    # The container this widget belongs to
    container = None
    # Whether this widget has been painted yet
    _painted = False
    # The widget used to paint the background
    background = None
    # ...
    _rect_content = None
    # A dictionary of signal callbacks, hashed by signal ID
    connects = None
    
    def __init__(self, **params): 
        """Create a new Widget instance given the style parameters.

        Keyword arguments:
            decorate -- whether to call theme.decorate(self) to allow the 
                theme a chance to decorate the widget. (default is true)
            style -- a dict of style parameters.
            x, y -- position parameters
            width, height -- size parameters
            align, valign -- alignment parameters, passed along to style
            font -- the font to use with this widget
            color -- the color property, if applicable
            background -- the widget used to paint the background
            cls -- class name as used by Theme
            name -- name of widget as used by Form.  If set, will call 
                form.add(self,name) to add the widget to the most recently 
                created Form.
            focusable -- True if this widget can receive focus via Tab, etc.
                (default is True)
            disabled -- True of this widget is disabled (defaults is False)
            value -- initial value

        """
        #object.Object.__init__(self) 
        self.connects = {}
        params.setdefault('decorate',True)
        params.setdefault('style',{})
        params.setdefault('focusable',True)
        params.setdefault('disabled',False)
        
        self.focusable = params['focusable']
        self.disabled = params['disabled']
        
        self.rect = pygame.Rect(params.get('x',0),
                                params.get('y',0),
                                params.get('width',0),
                                params.get('height',0))
        
        s = params['style']
        #some of this is a bit "theme-ish" but it is very handy, so these
        #things don't have to be put directly into the style.
        for att in ('align','valign','x','y','width','height','color','font','background'):
            if att in params: s[att] = params[att]
        self.style = style.Style(self,s)
        
        self.cls = 'default'
        if 'cls' in params: self.cls = params['cls']
        if 'name' in params:    
            from . import form
            self.name = params['name']
            if form.Form.form:
                form.Form.form.add(self)
                self.form = form.Form.form
        if 'value' in params: self.value = params['value']
        self.pcls = ""
        
        if params['decorate'] != False:
            if (not pguglobals.app):
                # TODO - fix this somehow
                from . import app
                app.App()
            pguglobals.app.theme.decorate(self,params['decorate'])

    def focus(self):
        """Focus this Widget."""
        if self.container: 
            if self.container.myfocus != self:  ## by Gal Koren
                self.container.focus(self)

    def blur(self): 
        """Blur this Widget."""
        if self.container: self.container.blur(self)

    def open(self):
        """Open this widget as a modal dialog."""
        #if getattr(self,'container',None) != None: self.container.open(self)
        pguglobals.app.open(self)

    def close(self, w=None):
        """Close this widget, if it is currently an open dialog."""
        #if getattr(self,'container',None) != None: self.container.close(self)
        if (not w):
            w = self
        pguglobals.app.close(w)

    def is_open(self):
        return (self in pguglobals.app.windows)

    def is_hovering(self):
        """Returns true if the mouse is hovering over this widget."""
        if self.container:
            return (self.container.myhover is self)
        return False

    def resize(self,width=None,height=None):
        """Resize this widget and all sub-widgets, returning the new size.

        This should be implemented by a subclass.

        """
        return (self.style.width, self.style.height)

    def chsize(self):
        """Signal that this widget has changed its size."""
        
        if (not self._painted): 
            return
        
        if (not self.container): 
            return
        
        if (pguglobals.app):
            pguglobals.app.chsize()

    def update(self,s):
        """Updates the surface and returns a rect list of updated areas

        This should be implemented by a subclass.

        """
        return
        
    def paint(self,s):
        """Render this widget onto the given surface

        This should be implemented by a subclass.

        """
        return

    def repaint(self): 
        """Request a repaint of this Widget."""
        if self.container: self.container.repaint(self)

    def repaintall(self):
        """Request a repaint of all Widgets."""
        if self.container: self.container.repaintall()

    def reupdate(self): 
        """Request a reupdate of this Widget."""
        if self.container: self.container.reupdate(self)

    def next(self): 
        """Pass focus to next Widget.
        
        Widget order determined by the order they were added to their container.

        """
        if self.container: self.container.next(self)

    def previous(self): 
        """Pass focus to previous Widget.
        
        Widget order determined by the order they were added to their container.

        """
        
        if self.container: self.container.previous(self)
    
    def get_abs_rect(self):
        """Returns the absolute rect of this widget on the App screen."""
        x, y = self.rect.x, self.rect.y
        cnt = self.container
        while cnt:
            x += cnt.rect.x
            y += cnt.rect.y
            if cnt._rect_content:
                x += cnt._rect_content.x
                y += cnt._rect_content.y
            cnt = cnt.container
        return pygame.Rect(x, y, self.rect.w, self.rect.h)

    def connect(self,code,func,*params):
        """Connect an event code to a callback function.
        
        Note that there may be multiple callbacks per event code.

        Arguments:
            code -- event type
            fnc -- callback function
            *values -- values to pass to callback.  

        Please note that callbacks may also have "magicaly" parameters.  
        Such as:

            _event -- receive the event
            _code -- receive the event code
            _widget -- receive the sending widget
        
        Example:
            def onclick(value):
                print 'click', value
            
            w = Button("PGU!")
            w.connect(gui.CLICK,onclick,'PGU Button Clicked')

        """
        if (not code in self.connects):
            self.connects[code] = []
        for cb in self.connects[code]:
            if (cb.func == func):
                # Already connected to this callback function
                return
        # Wrap the callback function and add it to the list
        cb = SignalCallback()
        cb.func = func
        cb.params = params
        self.connects[code].append(cb)

    # Remove signal handlers from the given event code. If func is specified,
    # only those handlers will be removed. If func is None, all handlers
    # will be removed.
    def disconnect(self, code, func=None):
        if (not code in self.connects):
            return
        if (not func):
            # Remove all signal handlers
            del self.connects[code]
        else:
            # Remove handlers that call 'func'
            n = 0
            callbacks = self.connects[code]
            while (n < len(callbacks)):
                if (callbacks[n].func == func):
                    # Remove this callback
                    del callbacks[n]
                else:
                    n += 1

    def send(self,code,event=None):
        """Send a code, event callback trigger."""
        if (not code in self.connects):
            return
        # Trigger all connected signal handlers
        for cb in self.connects[code]:
            func = cb.func
            values = list(cb.params)

            # Attempt to be compatible with previous versions of python
            try:
                code = func.__code__
            except:
                code = func.func_code

            nargs = code.co_argcount
            names = list(code.co_varnames)[:nargs]

            # If the function is bound to an instance, remove the first argument name. Again
            # we keep compatibility with older versions of python.
            if (hasattr(func, "__self__") and hasattr(func.__self__, "__class__") or 
                hasattr(func,'im_class')): 
                names.pop(0)
            
            args = []
            magic = {'_event':event,'_code':code,'_widget':self}
            for name in names:
                if name in magic.keys():
                    args.append(magic[name])
                elif len(values):
                    args.append(values.pop(0))
                else:
                    break
            args.extend(values)
            func(*args)
    
    def _event(self,e):
        if self.disabled: return
        self.send(e.type,e)
        return self.event(e)
        
    def event(self,e):
        """Called when an event is passed to this object.
        
        Please note that if you use an event, returning the value True
        will stop parent containers from also using the event.  (For example, if
        your widget handles TABs or arrow keys, and you don't want those to 
        also alter the focus.)

        This should be implemented by a subclass.

        """
        return

    def get_toplevel(self):
        """Returns the top-level widget (usually the Desktop) by following the
        chain of 'container' references."""
        top = self
        while (top.container):
            top = top.container
        return top

    def collidepoint(self, pos):
        """Test if the given point hits this widget. Over-ride this function
        for more advanced collision testing."""
        return self.rect.collidepoint(pos)


