"""
"""

import traceback

from .const import *
from .button import Button
from .basic import Label, Image
from .table import Table

class Select(Table):
    """A combo dropdown box widget.
    
    Example:
        w = Select(value="goats")
        w.add("Cats","cats")
        w.add("Goats","goats")
        w.add("Dogs","Dogs")
        w.value = 'dogs' #changes the value from goats to dogs
    
    """

    # The drop-down arrow button for the selection widget
    top_arrow = None
    # A button displaying the currently selected item
    top_selection = None
    # The first option added to the selector
    firstOption = None
    # The PGU table of options
    options = None
    _value = None

    def __init__(self,value=None,**params):
        params.setdefault('cls','select')
        Table.__init__(self,**params)
        
        label = Label(" ",cls=self.cls+".option.label")
        self.top_selected = Button(label, cls=self.cls+".selected")
        Table.add(self,self.top_selected) #,hexpand=1,vexpand=1)#,0,0)
        
        self.top_arrow = Button(Image(self.style.arrow), cls=self.cls+".arrow")
        Table.add(self,self.top_arrow) #,hexpand=1,vexpand=1) #,1,0)
        
        self.options = Table(cls=self.cls+".options")
        self.options.connect(BLUR,self._close,None)
        self.options.name = "pulldown-table"
        
        self.values = []
        self.value = value

    def resize(self,width=None,height=None):
        max_w,max_h = 0,0
        for w in self.options.widgets:
            w.rect.w,w.rect.h = w.resize()
            max_w,max_h = max(max_w,w.rect.w),max(max_h,w.rect.h)
        
        #xt,xr,xb,xl = self.top_selected.getspacing()
        self.top_selected.style.width = max_w #+ xl + xr
        self.top_selected.style.height = max_h #+ xt + xb
        
        self.top_arrow.connect(CLICK,self._open,None)
        self.top_selected.connect(CLICK,self._open,None)
        
        w,h = Table.resize(self,width,height)
        
        self.options.style.width = w
        #HACK: sort of, but not a big one..
        self.options.resize()
        
        return w,h
        
    def _open(self,value):
        opts = self.options
        
        opts.rect.w, opts.rect.h = opts.resize()
        
#        y = self.rect.y
#        c = self.container
#        while hasattr(c, 'container'):
#            y += c.rect.y
#            if (not c.container): 
#                break
#            c = c.container
            
#        if y + self.rect.h + opts.rect.h <= c.rect.h: #down
#            dy = self.rect.y + self.rect.h
#        else: #up
#            dy = self.rect.y - self.rect.h

        opts.rect.w, opts.rect.h = opts.resize()

        # TODO - make sure there is enough space to open down
        # ...
        yp = self.rect.bottom-1

        self.container.open(opts, self.rect.x, yp)
        self.firstOption.focus()

        # TODO - this is a hack
        for opt in self.options.widgets:
            opt.repaint()

    def _close(self,value):
        self.options.close()
        self.top_selected.focus()
    
    def _setvalue(self,value):
        self.value = value._value
        if self.container:
            #self.chsize()
            #HACK: improper use of resize()
            #self.resize() #to recenter the new value, etc.
            pass
        #    #self._resize()
        
        self._close(None)
        #self.repaint() #this will happen anyways
        
    
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        mywidget = None
        for w in self.values:
            if w._value == val:
                mywidget = w
        oldval = self._value
        self._value = val
        if (oldval != val):
            self.send(CHANGE)
            self.repaint()
        if not mywidget:
            mywidget = Label(" ",cls=self.cls+".option.label")
        self.top_selected.value = mywidget
        
    
    def add(self,w,value=None):
        """Add a widget and associated value to the dropdown box."""
        
        if type(w) == str: w = Label(w,cls=self.cls+".option.label")
        
        w.style.align = -1
        btn = Button(w,cls=self.cls+".option")
        btn.connect(CLICK,self._setvalue,w)
        
        self.options.tr()
        self.options.add(btn)
        
        if (not self.firstOption):
            self.firstOption = btn
        
        if value != None: w._value = value
        else: w._value = w
        if self.value == w._value:
            self.top_selected.value = w
        self.values.append(w)

