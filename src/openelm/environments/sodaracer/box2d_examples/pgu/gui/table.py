"""
"""

import sys

from .const import *
from . import container

class Table(container.Container):
    """A table style container widget.
    
    Example:
        t = gui.Table()
        
        # This starts a new row of the table
        t.tr()
        # The 'td' call creates a new table cell
        t.td(gui.Label("Name:"), align=-1)
        t.td(gui.Input())

        t.tr()
        # The table cells can span multiple columns
        t.td(gui.Label("Email"), align=-1, colspan=2)

        t.tr()
        t.td(gui.Input(), colspan=2)
        
    """
    
    
    def __init__(self, **params):
        params.setdefault('cls','table')
        container.Container.__init__(self, **params)
        self._rows = []
        self._curRow = 0
        self._trok = False
        self._hpadding = params.get("hpadding", 0)
        self._vpadding = params.get("vpadding", 0)
    
    def getRows(self):
        return len(self._rows)
    
    def getColumns(self):
        if self._rows:
            return len(self._rows[0])
        else:
            return 0
    
    def remove_row(self, n): #NOTE: won't work in all cases.
        if n >= self.getRows():
            print("Trying to remove a nonexistant row:", n, "there are only", self.getRows(), "rows")
            return
        
        for cell in self._rows[n]:
            if isinstance(cell, dict) and cell["widget"] in self.widgets:
                #print 'removing widget'
                self.widgets.remove(cell["widget"])
        del self._rows[n]
        #print "got here"
        
        for w in self.widgets:
            if w.style.row > n: w.style.row -= 1
        
        if self._curRow >= n:
            self._curRow -= 1
        
        #self.rect.w, self.rect.h = self.resize()
        #self.repaint()
        
        self.chsize()
    
    def clear(self):
        self._rows = []
        self._curRow = 0
        self._trok = False

        self.widgets = []
        
        self.chsize()
        
        #print 'clear',self,self._rows
    
    def _addRow(self):
        self._rows.append([None for x in range(self.getColumns())])
    
    def tr(self):
        """Start on the next row."""
        if not self._trok:
            self._trok = True
            return 
        self._curRow += 1
        if self.getRows() <= self._curRow:
            self._addRow()
    
    def _addColumn(self):
        if not self._rows:
            self._addRow()
        for row in self._rows:
            row.append(None)
    
    def _setCell(self, w, col, row, colspan=1, rowspan=1):
        #make room for the widget by adding columns and rows
        while self.getColumns() < col + colspan:
            self._addColumn()
        while self.getRows() < row + rowspan:
            self._addRow()
            
        #print w.__class__.__name__,col,row,colspan,rowspan
        
        #actual widget setting and modification stuff
        w.container = self
        w.style.row = row #HACK - to work with gal's list
        w.style.col = col #HACK - to work with gal's list
        self._rows[row][col] = {"widget":w, "colspan":colspan, "rowspan":rowspan}
        self.widgets.append(self._rows[row][col]["widget"])
        
        #set the spanned columns
        #for acell in range(col + 1, col + colspan):
        #    self._rows[row][acell] = True
        
        #set the spanned rows and the columns on them
        #for arow in range(row + 1, row + rowspan):
        #    for acell in range(col, col + colspan): #incorrect?
        #        self._rows[arow][acell] = True
        
        for arow in range(row, row + rowspan):
            for acell in range(col, col + colspan): #incorrect?
                if row != arow or col != acell:
                    self._rows[arow][acell] = True
    
    
    def td(self, w, col=None, row=None, colspan=1, rowspan=1, **params):
        """Add a widget to a table after wrapping it in a TD container.

        Keyword arguments:        
            w -- widget
            col -- column
            row -- row
            colspan -- colspan
            rowspan -- rowspan
            align -- horizontal alignment (-1,0,1)
            valign -- vertical alignment (-1,0,1)
            params -- other params for the TD container, style information, etc

        """
        
        Table.add(self,_Table_td(w, **params), col=col, row=row, colspan=colspan, rowspan=rowspan)
    
    def add(self, w, col=None, row=None, colspan=1, rowspan=1):
        """Add a widget directly into the table, without wrapping it in a TD container.
        
        See Table.td for an explanation of the parameters.

        """
        self._trok = True
        #if no row was specifically specified, set it to the current row
        if row is None:
            row = self._curRow
            #print row
        
        #if its going to be a new row, have it be on the first column
        if row >= self.getRows():
            col = 0
        
        #try to find an open cell for the widget
        if col is None:
            for cell in range(self.getColumns()):
                if col is None and not self._rows[row][cell]:
                    col = cell
                    break
        
        #otherwise put the widget in a new column
        if col is None:
            col = self.getColumns()
        
        self._setCell(w, col, row, colspan=colspan, rowspan=rowspan)
        
        self.chsize()
        return
        
    def remove(self,w):
        if hasattr(w,'_table_td'): w = w._table_td
        row,col = w.style.row,w.style.col
        cell = self._rows[row][col]
        colspan,rowspan = cell['colspan'],cell['rowspan']
        
        for arow in range(row , row + rowspan):
            for acell in range(col, col + colspan): #incorrect?
                self._rows[arow][acell] = False
        self.widgets.remove(w)
        self.chsize()
        
        
    
    def resize(self, width=None, height=None):
        #if 1 or self.getRows() == 82:
            #print ''
            #print 'resize',self.getRows(),self.getColumns(),width,height
            #import inspect
            #for obj,fname,line,fnc,code,n in inspect.stack()[9:20]:
            #    print fname,line,':',fnc,code[0].strip()

        
        #resize the widgets to their smallest size
        for w in self.widgets:
            w.rect.w, w.rect.h = w.resize()
        
        #calculate row heights and column widths
        rowsizes = [0 for y in range(self.getRows())]
        columnsizes = [0 for x in range(self.getColumns())]
        for row in range(self.getRows()):
            for cell in range(self.getColumns()):
                if self._rows[row][cell] and self._rows[row][cell] is not True:
                    if not self._rows[row][cell]["colspan"] > 1:
                        columnsizes[cell] = max(columnsizes[cell], self._rows[row][cell]["widget"].rect.w)
                    if not self._rows[row][cell]["rowspan"] > 1:
                        rowsizes[row] = max(rowsizes[row], self._rows[row][cell]["widget"].rect.h)
        
        #distribute extra space if necessary for wide colspanning/rowspanning
        def _table_div(a,b,c):
            v,r = a/b, a%b
            if r != 0 and (c%b)<r: v += 1
            return v

        for row in range(self.getRows()):
            for cell in range(self.getColumns()):
                if self._rows[row][cell] and self._rows[row][cell] is not True:
                    if self._rows[row][cell]["colspan"] > 1:
                        columns = range(cell, cell + self._rows[row][cell]["colspan"])
                        totalwidth = 0
                        for acol in columns:
                            totalwidth += columnsizes[acol]
                        if totalwidth < self._rows[row][cell]["widget"].rect.w:
                            for acol in columns:
                                columnsizes[acol] += _table_div(self._rows[row][cell]["widget"].rect.w - totalwidth, self._rows[row][cell]["colspan"],acol)
                    if self._rows[row][cell]["rowspan"] > 1:
                        rows = range(row, row + self._rows[row][cell]["rowspan"])
                        totalheight = 0
                        for arow in rows:
                            totalheight += rowsizes[arow]
                        if totalheight < self._rows[row][cell]["widget"].rect.h:
                            for arow in rows:
                                rowsizes[arow] += _table_div(self._rows[row][cell]["widget"].rect.h - totalheight, self._rows[row][cell]["rowspan"],arow)

        # Now calculate the total width and height occupied by the rows and columns
        rowsizes = [sz+2*self._vpadding for sz in rowsizes]
        columnsizes = [sz+2*self._hpadding for sz in columnsizes]

        # Now possibly expand the table cells to fill out the specified width
        w = sum(columnsizes)
        if (w > 0 and w < self.style.width):
            amount = (self.style.width - w)/float(w)
            for n in range(0, len(columnsizes)):
                columnsizes[n] += columnsizes[n] * amount

        # Do the same for the table height
        h = sum(rowsizes)
        if (h > 0 and h < self.style.height):
            amount = (self.style.height - h) / float(h)
            for n in range(0, len(rowsizes)):
                rowsizes[n] += rowsizes[n] * amount
        
        #set the widget's position by calculating their row/column x/y offset
        cellpositions = [[[sum(columnsizes[0:cell]), sum(rowsizes[0:row])] for cell in range(self.getColumns())] for row in range(self.getRows())]
        for row in range(self.getRows()):
            for cell in range(self.getColumns()):
                if self._rows[row][cell] and self._rows[row][cell] is not True:
                    x, y = cellpositions[row][cell]
                    w = sum(columnsizes[cell:cell+self._rows[row][cell]["colspan"]])
                    h = sum(rowsizes[row:row+self._rows[row][cell]["rowspan"]])
                    
                    widget = self._rows[row][cell]["widget"]
                    widget.rect.x = x
                    widget.rect.y = y
                    if 1 and (w,h) != (widget.rect.w,widget.rect.h):
#                         if h > 20:
#                             print widget.widget.__class__.__name__, (widget.rect.w,widget.rect.h),'=>',(w,h)
                        widget.rect.w, widget.rect.h = widget.resize(w, h)
                    
                    #print self._rows[row][cell]["widget"].rect
        
        #print columnsizes
        #print sum(columnsizes)
        #size = sum(columnsizes), sum(rowsizes); print size
        
        #return the tables final size
        return sum(columnsizes),sum(rowsizes)


class _Table_td(container.Container):
    def __init__(self,widget,**params):#hexpand=0,vexpand=0,
        container.Container.__init__(self,**params)
        self.widget = widget
        #self.hexpand=hexpand
        #self.vexpand=vexpand
        widget._table_td = self
        self.add(widget,0,0)
    
    def resize(self,width=None,height=None):
        w = self.widget
        
        #expansion code, but i didn't like the idea that much..
        #a bit obscure, fairly useless when a user can just
        #add a widget to a table instead of td it in.
        #ww,hh=None,None
        #if self.hexpand: ww = self.style.width
        #if self.vexpand: hh = self.style.height
        #if self.hexpand and width != None: ww = max(ww,width)
        #if self.vexpand and height != None: hh = max(hh,height)
        #w.rect.w,w.rect.h = w.resize(ww,hh)
        
        #why bother, just do the lower mentioned item...
        w.rect.w,w.rect.h = w.resize()
        
        #this should not be needed, widgets should obey their sizing on their own.
        
#         if (self.style.width!=0 and w.rect.w > self.style.width) or (self.style.height!=0 and w.rect.h > self.style.height):
#             ww,hh = None,None
#             if self.style.width: ww = self.style.width
#             if self.style.height: hh = self.style.height
#             w.rect.w,w.rect.h = w.resize(ww,hh)
      
  
        #in the case that the widget is too big, we try to resize it
        if (width != None and width < w.rect.w) or (height != None and height < w.rect.h):
            (w.rect.w, w.rect.h) = w.resize(width, height)

        # In python3 max and min no longer accept None as an argument
        if (width == None): width = -sys.maxsize
        if (height == None): height = -sys.maxsize
        
        width = max(width, w.rect.w, self.style.width) #,self.style.cell_width)
        height = max(height, w.rect.h, self.style.height) #,self.style.cell_height)
        
        dx = width-w.rect.w
        dy = height-w.rect.h
        w.rect.x = (self.style.align+1)*dx/2
        w.rect.y = (self.style.valign+1)*dy/2
        
        return width,height

