"""Document layout engine."""

class Layout:
    """The document layout engine."""
    
    def __init__(self,rect=None):
        """initialize the object with the size of the box."""
        self._widgets = []
        self.rect = rect
        
    def add(self,e): 
        """Add a document element to the layout.
        
        The document element may be
        * a tuple (w,h) if it is a whitespace element
        * a tuple (0,h) if it is a linebreak element
        * an integer -1,0,1 if it is a command to start a new block of elements 
            that are aligned either left,center, or right.
        * an object with a .rect (for size) -- such as a word element
        * an object with a .rect (for size) and .align -- such as an image element

        """
        
        self._widgets.append(e)
        
        
    def resize(self):
        """Resize the layout.

        This method recalculates the position of all document elements after 
        they have been added to the document.  .rect.x,y will be updated for
        all objects.

        """
        self.init()
        self.widgets = []
        for e in self._widgets:
            if type(e) is tuple and e[0] != 0:
                self.do_space(e)
            elif type(e) is tuple and e[0] == 0:
                self.do_br(e[1])
            elif type(e) is int:
                self.do_block(align=e)
            elif hasattr(e,'align'):
                self.do_align(e)
            else:
                self.do_item(e)
        self.line()
        self.rect.h = max(self.y,self.left_bottom,self.right_bottom)
            
    def init(self):
        self.x,self.y = self.rect.x,self.rect.y
        self.left = self.rect.left
        self.right = self.rect.right
        self.left_bottom = 0
        self.right_bottom = 0
        self.y = self.rect.y
        self.x = self.rect.x
        self.h = 0
        
        self.items = []
        self.align = -1
        
    def getleft(self):
        if self.y > self.left_bottom:
            self.left = self.rect.left
        return self.left
        
    def getright(self):
        if self.y > self.right_bottom:
            self.right = self.rect.right
        return self.right
        
    def do_br(self,h): 
        self.line()
        self.h = h
    
    def do_block(self,align=-1):
        self.line()
        self.align = align

    def do_align(self,e):
        align = e.align
        ox,oy,oh = self.x,self.y,self.h
        w,h = e.rect.w,e.rect.h
        
        if align == 0: 
            self.line()
            self.x = self.rect.left + (self.rect.width-w)/2
            self.fit = 0
        elif align == -1: 
            self.line()
            self.y = max(self.left_bottom,self.y + self.h)
            self.h = 0
            self.x = self.rect.left
        elif align == 1: 
            self.line()
            self.y = max(self.right_bottom,self.y + self.h)
            self.h = 0
            self.x = self.rect.left + (self.rect.width-w)

        e.rect.x,e.rect.y = self.x,self.y        
            
        self.x = self.x + w
        self.y = self.y
        
        if align == 0:
            self.h = max(self.h,h)
            self.y = self.y + self.h
            self.x = self.getleft()
            self.h = 0
        elif align == -1:
            self.left = self.x
            self.left_bottom = self.y + h
            self.x,self.y,self.h = ox + w,oy,oh
        elif align == 1: 
            self.right = self.x - w
            self.right_bottom = self.y + h
            self.x,self.y,self.h = ox,oy,oh
        
        self.widgets.append(e)

    def do_space(self,e):
        w,h = e
        if self.x+w >= self.getright(): 
            self.line()
        else: 
            self.items.append(e)
            self.h = max(self.h,h)
            self.x += w
    
    def do_item(self,e):
        w,h = e.rect.w,e.rect.h
        if self.x+w >= self.getright(): 
            self.line()
        self.items.append(e)
        self.h = max(self.h,h)
        self.x += w
    
    def line(self):
        x1 = self.getleft()
        x2 = self.getright()
        align = self.align
        y = self.y
        
        if len(self.items) != 0 and type(self.items[-1]) == tuple:
            del self.items[-1]
        
        w = 0
        for e in self.items:
            if type(e) == tuple: w += e[0]
            else: w += e.rect.w
            
        if align == -1: x = x1
        elif align == 0: 
            x = x1 + ((x2-x1)-w)/2
            self.fit = 0
        elif align == 1: x = x2 - w
            
        for e in self.items:
            if type(e) == tuple: x += e[0]
            else:
                e.rect.x,e.rect.y = x,y
                self.widgets.append(e)
                x += e.rect.w
        
        self.items = []
        self.y = self.y + self.h
        self.x = self.getleft()
        self.h = 0
        

