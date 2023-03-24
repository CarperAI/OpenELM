"""
"""
import os

from .const import *
from . import table, area
from . import basic, input, button
from . import pguglobals

class Dialog(table.Table):
    """A dialog window with a title bar and an "close" button on the bar.
    
    Example:
        title = gui.Label("My Title")
        main = gui.Container()
        #add stuff to the container...
        
        d = gui.Dialog(title,main)
        d.open()

    """
    def __init__(self,title,main,**params):
        """Dialog constructor.

        Arguments:
            title -- title widget, usually a label
            main -- main widget, usually a container

        """        
        params.setdefault('cls','dialog')
        table.Table.__init__(self,**params)
        
        
        self.tr()
        self.td(title,align=-1,cls=self.cls+'.bar')
        clos = button.Icon(self.cls+".bar.close")
        clos.connect(CLICK,self.close,None) 
        self.td(clos,align=1,cls=self.cls+'.bar')
        
        self.tr()
        self.td(main,colspan=2,cls=self.cls+".main")
        
        
#         self.tr()
#         
#         
#         t = table.Table(cls=self.cls+".bar")
#         t.tr()
#         t.td(title)
#         clos = button.Icon(self.cls+".bar.close")
#         t.td(clos,align=1)
#         clos.connect(CLICK,self.close,None) 
#         self.add(t,0,0)
#         
#         main.rect.w,main.rect.h = main.resize()
#         clos.rect.w,clos.rect.h = clos.resize()
#         title.container.style.width = main.rect.w - clos.rect.w
#         
#         self.tr()
#         self.td(main,cls=self.cls+".main")
# 
        
        
class FileDialog(Dialog):
    """A file picker dialog window."""
    
    def __init__(self, title_txt="File Browser", button_txt="Okay", cls="dialog", path=None):
        """FileDialog constructor.

        Keyword arguments:
            title_txt -- title text
            button_txt -- button text
            path -- initial path

        """

        cls1 = 'filedialog'
        if not path: self.curdir = os.getcwd()
        else: self.curdir = path
        self.dir_img = basic.Image(
            pguglobals.app.theme.get(cls1+'.folder', '', 'image'))
        td_style = {'padding_left': 4,
                    'padding_right': 4,
                    'padding_top': 2,
                    'padding_bottom': 2}
        self.title = basic.Label(title_txt, cls=cls+".title.label")
        self.body = table.Table()
        self.list = area.List(width=350, height=150)
        self.input_dir = input.Input()
        self.input_file = input.Input()
        self._list_dir_()
        self.button_ok = button.Button(button_txt)
        self.body.tr()
        self.body.td(basic.Label("Folder"), style=td_style, align=-1)
        self.body.td(self.input_dir, style=td_style)
        self.body.tr()
        self.body.td(self.list, colspan=3, style=td_style)
        self.list.connect(CHANGE, self._item_select_changed_, None)
        self.button_ok.connect(CLICK, self._button_okay_clicked_, None)
        self.body.tr()
        self.body.td(basic.Label("File"), style=td_style, align=-1)
        self.body.td(self.input_file, style=td_style)
        self.body.td(self.button_ok, style=td_style)
        self.value = None
        Dialog.__init__(self, self.title, self.body)
        
    def _list_dir_(self):
        self.input_dir.value = self.curdir
        self.input_dir.pos = len(self.curdir)
        self.input_dir.vpos = 0
        dirs = []
        files = []
        try:
            for i in os.listdir(self.curdir):
                if os.path.isdir(os.path.join(self.curdir, i)): dirs.append(i)
                else: files.append(i)
        except:
            self.input_file.value = "Opps! no access"
        #if '..' not in dirs: dirs.append('..')
        dirs.sort()
        dirs = ['..'] + dirs
        
        files.sort()
        for i in dirs:
            #item = ListItem(image=self.dir_img, text=i, value=i)
            self.list.add(i,image=self.dir_img,value=i)
        for i in files:
            #item = ListItem(image=None, text=i, value=i)
            self.list.add(i,value=i)
        #self.list.resize()
        self.list.set_vertical_scroll(0)
        #self.list.repaintall()
        
        
    def _item_select_changed_(self, arg):
        self.input_file.value = self.list.value
        fname = os.path.abspath(os.path.join(self.curdir, self.input_file.value))
        if os.path.isdir(fname):
            self.input_file.value = ""
            self.curdir = fname
            self.list.clear()
            self._list_dir_()


    def _button_okay_clicked_(self, arg):
        if self.input_dir.value != self.curdir:
            if os.path.isdir(self.input_dir.value):
                self.input_file.value = ""
                self.curdir = os.path.abspath(self.input_dir.value)
                self.list.clear()
                self._list_dir_()
        else:
            self.value = os.path.join(self.curdir, self.input_file.value)
            self.send(CHANGE)
            self.close()

