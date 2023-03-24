"""
"""
from .const import *
from . import table
from . import basic, button

class _Menu_Options(table.Table):
    def __init__(self,menu,**params):
        table.Table.__init__(self,**params)
        
        self.menu = menu
    
    def event(self,e):
        handled = False
        arect = self.get_abs_rect()
        
        if e.type == MOUSEMOTION:
            abspos = e.pos[0]+arect.x,e.pos[1]+arect.y
            for w in self.menu.container.widgets:
                if not w is self.menu:
                    mrect = w.get_abs_rect()
                    if mrect.collidepoint(abspos):
                        self.menu._close(None)
                        w._open(None)
                        handled = True
        
        if not handled: table.Table.event(self,e)

class _Menu(button.Button):
    def __init__(self,parent,widget=None,**params): #TODO widget= could conflict with module widget
        params.setdefault('cls','menu')
        button.Button.__init__(self,widget,**params)
        
        self.parent = parent
        
        self._cls = self.cls
        self.options = _Menu_Options(self, cls=self.cls+".options")
        
        self.connect(CLICK,self._open,None)
        
        self.pos = 0
    
    def _open(self,value):
        self.parent.value = self
        self.pcls = 'down'
        
        self.repaint()
        self.container.open(self.options,self.rect.x,self.rect.bottom)
        self.options.connect(BLUR,self._close,None)
        self.options.focus()
        self.repaint()
        
    def _pass(self,value):
        pass
        
    def _close(self,value):
        self.pcls = ''
        self.parent.value = None
        self.repaint()
        self.options.close()
    
    def _valuefunc(self,value):
        self._close(None)
        if value['fnc'] != None:
            value['fnc'](value['value'])
            
    def event(self,e):
        button.Button.event(self,e)
        
        if self.parent.value == self:
            self.pcls = 'down'
    
    def add(self,w,fnc=None,value=None):
        w.style.align = -1
        b = button.Button(w,cls=self.cls+".option")
        b.connect(CLICK,self._valuefunc,{'fnc':fnc,'value':value})
        
        self.options.tr()
        self.options.add(b)
        
        return b

class Menus(table.Table):
    """A drop down menu bar.

    Example:
        data = [
            ('File/Save', fnc_save, None),
            ('File/New', fnc_new, None),
            ('Edit/Copy', fnc_copy, None),
            ('Edit/Cut', fnc_cut, None),
            ('Help/About', fnc_help, help_about_content),
            ('Help/Reference', fnc_help, help_reference_content),
            ]
        w = Menus(data)

    """
    
    def __init__(self,data,menu_cls='menu',**params):
        params.setdefault('cls','menus')
        table.Table.__init__(self,**params)
        
        self.value = None
        
        n,m,mt = 0,None,None
        for path,cmd,value in data:
            parts = path.split("/")
            if parts[0] != mt:
                mt = parts[0]
                m = _Menu(self,basic.Label(mt,cls=menu_cls+".label"),cls=menu_cls)
                self.add(m,n,0)
                n += 1
            print ("add", parts[1], cmd, value)
            m.add(basic.Label(parts[1],cls=m.cls+".option.label"),cmd,value)

