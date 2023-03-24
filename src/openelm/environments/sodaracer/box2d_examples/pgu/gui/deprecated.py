import pygame

from .const import *
from . import table
from . import group
from . import button, basic
from . import pguglobals

def action_open(value):
    print('gui.action_open',"Scheduled to be deprecated.")
    value.setdefault('x',None)
    value.setdefault('y',None)
    value['container'].open(value['window'],value['x'],value['y'])

def action_setvalue(value):
    print('gui.action_setvalue',"Scheduled to be deprecated.")
    a,b = value
    b.value = a.value

def action_quit(value):
    print('gui.action_quit',"Scheduled to be deprecated.")
    value.quit()

def action_exec(value):
    print('gui.action_exec',"Scheduled to be deprecated.")
    exec(value['script'],globals(),value['dict'])

class Toolbox(table.Table):
    def __setattr__(self,k,v):
        _v = self.__dict__.get(k,NOATTR)
        self.__dict__[k]=v
        if k == 'value' and _v != NOATTR and _v != v: 
            self.group.value = v
            for w in self.group.widgets:
                if w.value != v: w.pcls = ""
                else: w.pcls = "down"
            self.repaint()
    
    def _change(self,value):
        self.value = self.group.value
        self.send(CHANGE)
    
    def __init__(self,data,cols=0,rows=0,tool_cls='tool',value=None,**params):
        print('gui.Toolbox','Scheduled to be deprecated.')
        params.setdefault('cls','toolbox')
        table.Table.__init__(self,**params)
        
        if cols == 0 and rows == 0: cols = len(data)
        if cols != 0 and rows != 0: rows = 0
        
        self.tools = {}
        
        _value = value
        
        g = group.Group()
        self.group = g
        g.connect(CHANGE,self._change,None)
        self.group.value = _value
        
        x,y,p,s = 0,0,None,1
        for ico,value in data:
            #from __init__ import theme
            img = pguglobals.app.theme.get(tool_cls+"."+ico,"","image")
            if img:
                i = basic.Image(img)
            else: i = basic.Label(ico,cls=tool_cls+".label")
            p = button.Tool(g,i,value,cls=tool_cls)
            self.tools[ico] = p
            #p.style.hexpand = 1
            #p.style.vexpand = 1
            self.add(p,x,y)
            s = 0
            if cols != 0: x += 1
            if cols != 0 and x == cols: x,y = 0,y+1
            if rows != 0: y += 1
            if rows != 0 and y == rows: x,y = x+1,0
