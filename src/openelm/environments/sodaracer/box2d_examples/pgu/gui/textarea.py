"""
"""
import pygame
from pygame.locals import *

from .const import *
from . import widget

class TextArea(widget.Widget):
    """A multi-line text input.
    
    Example:
        w = TextArea(value="Cuzco the Goat",size=20)
        w = TextArea("Marbles")
        w = TextArea("Groucho\nHarpo\nChico\nGummo\nZeppo\n\nMarx", 200, 400, 12)
    
    """
    def __init__(self,value="",width = 120, height = 30, size=20,**params):
        params.setdefault('cls','input')
        params.setdefault('width', width)
        params.setdefault('height', height)
        
        widget.Widget.__init__(self,**params)
        self.value = value                # The value of the TextArea
        self.pos = len(str(value))        # The position of the cursor
        self.vscroll = 0                # The number of lines that the TextArea is currently scrolled
        self.font = self.style.font        # The font used for rendering the text
        self.cursor_w = 2                 # Cursor width (NOTE: should be in a style)
        w,h = self.font.size("e"*size)    
        if not self.style.height: self.style.height = h
        if not self.style.width: self.style.width = w
    
## BUG: This causes textarea to grow every time table._Table_td calculates its
## size.
##    def resize(self,width=None,height=None):
##        if (width != None) and (height != None):
##            print 'TextArea RESIZE'
##            self.rect = pygame.Rect(self.rect.x, self.rect.y, width, height)
##        return self.rect.w, self.rect.h
        
    def paint(self,s):
        
        # TODO: What's up with this 20 magic number? It's the margin of the left and right sides, but I'm not sure how this should be gotten other than by trial and error.
        max_line_w = self.rect.w - 20
                
        # Update the line allocation for the box's value
        self.doLines(max_line_w)
        
        # Make sure that the vpos and hpos of the cursor is set properly
        self.updateCursorPos()

        # Make sure that we're scrolled vertically such that the cursor is visible
        if (self.vscroll < 0):
            self.vscroll = 0
        if (self.vpos < self.vscroll):
            self.vscroll = self.vpos
        elif ((self.vpos - self.vscroll + 1) * self.line_h > self.rect.h):
            self.vscroll = - (self.rect.h / self.line_h - self.vpos - 1)

        # Blit each of the lines in turn
        cnt = 0
        for line in self.lines:
            line_pos = (0, (cnt - self.vscroll) * self.line_h)
            if (line_pos[1] >= 0) and (line_pos[1] < self.rect.h):
                s.blit( self.font.render(line, 1, self.style.color), line_pos )
            cnt += 1
        
        # If the textarea is focused, then also show the cursor
        if self.container.myfocus is self:
            r = self.getCursorRect()
            s.fill(self.style.color,r)
    
    # This function updates self.vpos and self.hpos based on self.pos
    def updateCursorPos(self):
        self.vpos = 0 # Reset the current line that the cursor is on
        self.hpos = 0
        
        line_cnt = 0
        char_cnt = 0

        for line in self.lines:
            line_char_start = char_cnt # The number of characters at the start of the line
            
            # Keep track of the character count for words
            char_cnt += len(line)
            
            # If our cursor count is still less than the cursor position, then we can update our cursor line to assume that it's at least on this line
            if (char_cnt > self.pos):
                self.vpos = line_cnt
                self.hpos = self.pos - line_char_start

                break # Now that we know where our cursor is, we exit the loop

            line_cnt += 1
        
        if (char_cnt <= self.pos) and (len(self.lines) > 0):
            self.vpos = len(self.lines) - 1
            self.hpos = len(self.lines[ self.vpos ] )

    # Returns a rectangle that is of the size and position of where the cursor is drawn    
    def getCursorRect(self):
        lw = 0
        if (len(self.lines) > 0):
            lw, lh = self.font.size( self.lines[ self.vpos ][ 0:self.hpos ] )
            
        r = pygame.Rect(lw, (self.vpos - self.vscroll) * self.line_h, self.cursor_w, self.line_h)
        return r
    
    # This function sets the cursor position according to an x/y value (such as by from a mouse click)
    def setCursorByXY(self, pos):
        (x, y) = pos
        self.vpos = ((int) (y / self.line_h)) + self.vscroll
        if (self.vpos >= len(self.lines)):
            self.vpos = len(self.lines) - 1
            
        currentLine = self.lines[ self.vpos ]
        
        for cnt in range(0, len(currentLine) ):
            self.hpos = cnt
            lw, lh = self.font.size( currentLine[ 0:self.hpos + 1 ] )
            if (lw > x):
                break
        
        lw, lh = self.font.size( currentLine )
        if (lw < x):
            self.hpos = len(currentLine)
            
        self.setCursorByHVPos()
        
    # This function sets the cursor position by the horizontal/vertical cursor position.    
    def setCursorByHVPos(self):
        line_cnt = 0
        char_cnt = 0
        
        for line in self.lines:
            line_char_start = char_cnt # The number of characters at the start of the line
            
            # Keep track of the character count for words
            char_cnt += len(line)

            # If we're on the proper line
            if (line_cnt == self.vpos):
                # Make sure that we're not trying to go over the edge of the current line
                if ( self.hpos > len(line) ):
                    self.hpos = len(line) - 1
                # Set the cursor position
                self.pos = line_char_start + self.hpos
                break    # Now that we've set our cursor position, we exit the loop
                
            line_cnt += 1        
    
    # Splits up the text found in the control's value, and assigns it into the lines array
    def doLines(self, max_line_w):
        self.line_h = 10
        self.lines = [] # Create an empty starter list to start things out.
        
        inx = 0
        line_start = 0
        while inx >= 0:
            # Find the next breakable whitespace
            # HACK: Find a better way to do this to include tabs and system characters and whatnot.
            prev_word_start = inx # Store the previous whitespace
            spc_inx = self.value.find(' ', inx+1)
            nl_inx = self.value.find('\n', inx+1)
            
            if (min(spc_inx, nl_inx) == -1):
                inx = max(spc_inx, nl_inx)
            else:
                inx = min(spc_inx, nl_inx)
                
            # Measure the current line
            lw, self.line_h = self.font.size( self.value[ line_start : inx ] )
            
            # If we exceeded the max line width, then create a new line
            if (lw > max_line_w):
                #Fall back to the previous word start
                self.lines.append(self.value[ line_start : prev_word_start + 1 ])
                line_start = prev_word_start + 1
                # TODO: Check for extra-long words here that exceed the length of a line, to wrap mid-word
                
            # If we reached the end of our text
            if (inx < 0):
                # Then make sure we added the last of the line
                if (line_start < len( self.value ) ):
                    self.lines.append( self.value[ line_start : len( self.value ) ] )
                else:
                    self.lines.append('')
            # If we reached a hard line break
            elif (self.value[inx] == "\n"):
                # Then make a line break here as well.
                newline = self.value[ line_start : inx + 1 ]
                newline = newline.replace("\n", " ") # HACK: We know we have a newline character, which doesn't print nicely, so make it into a space. Comment this out to see what I mean.
                self.lines.append( newline )
                
                line_start = inx + 1
            else:
                # Otherwise, we just continue progressing to the next space
                pass
        
    def _setvalue(self,v):
        self.__dict__['value'] = v
        self.send(CHANGE)
    
    def event(self,e):
        used = None
        if e.type == KEYDOWN:    
            used = True
            if e.key == K_BACKSPACE:
                if self.pos:
                    self._setvalue(self.value[:self.pos-1] + self.value[self.pos:])
                    self.pos -= 1
            elif e.key == K_DELETE:
                if len(self.value) > self.pos:
                    self._setvalue(self.value[:self.pos] + self.value[self.pos+1:])
            elif e.key == K_HOME: 
                # Find the previous newline
                newPos = self.value.rfind('\n', 0, self.pos)
                if (newPos >= 0):
                    self.pos = newPos
            elif e.key == K_END:
                # Find the previous newline
                newPos = self.value.find('\n', self.pos, len(self.value) )
                if (newPos >= 0):
                    self.pos = newPos
            elif e.key == K_LEFT:
                if self.pos > 0: self.pos -= 1
#                used = True
            elif e.key == K_RIGHT:
                if self.pos < len(self.value): self.pos += 1
#                used = True
            elif e.key == K_UP:
                self.vpos -= 1
                self.setCursorByHVPos()
            elif e.key == K_DOWN:
                self.vpos += 1
                self.setCursorByHVPos()
            # The following return/tab keys are standard for PGU widgets, but I took them out here to facilitate multi-line text editing
#            elif e.key == K_RETURN:
#                self.next()
#            elif e.key == K_TAB:
#                pass                
            else:
                #c = str(e.unicode)
                used = None
                try:
                    if (e.key == K_RETURN):
                        c = "\n"
                    elif (e.key == K_TAB):
                        c = "  "
                    else:
                        c = (e.unicode).encode('latin-1')
                    if c:
                        used = True
                        self._setvalue(self.value[:self.pos] + c + self.value[self.pos:])
                        self.pos += len(c)
                except: #ignore weird characters
                    pass
            self.repaint()
        elif e.type == MOUSEBUTTONDOWN:
            self.setCursorByXY(e.pos)
            self.repaint()
            
        elif e.type == FOCUS:
            self.repaint()
        elif e.type == BLUR:
            self.repaint()
        
        self.pcls = ""
        if self.container.myfocus is self: self.pcls = "focus"
        
        return used
    
    def __setattr__(self,k,v):
        if k == 'value':
            if v == None: v = ''
            v = str(v)
            self.pos = len(v)
        _v = self.__dict__.get(k,NOATTR)
        self.__dict__[k]=v
        if k == 'value' and _v != NOATTR and _v != v: 
            self.send(CHANGE)
            self.repaint()
            
# The first version of this code was done by Clint Herron, and is a modified version of input.py (by Phil Hassey).
# It is under the same license as the rest of the PGU library.

