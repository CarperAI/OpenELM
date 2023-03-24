"""
"""
from . import widget

class Form(widget.Widget):
    """A form that automatically will contain all named widgets.
    
    After a form is created, all named widget that are subsequently created are 
    added to that form.  You may use dict style access to access named widgets.
    
    Example:

        f = gui.Form()
        
        w = gui.Input("Phil",name="firstname")
        w = gui.Input("Hassey",name="lastname")
        
        print f.results()
        print ''
        print f.items()
        print ''
        print f['firstname'].value
        print f['lastname'].value

    """

    # The current form instance
    form = None
    # The list of PGU widgets that are tracked by this form
    _elist = None
    # A mapping of PGU widgets tracked by this form (name -> instance)
    _emap = None
    # The dirty flag is set when a new widget is added to the form
    _dirty = 0
    
    def __init__(self):
        widget.Widget.__init__(self,decorate=False)
        self._elist = []
        self._emap = {}
        self._dirty = 0
        # Register this form as the one used by new widgets
        Form.form = self
    
    def add(self,e,name=None,value=None):
        """Adds a PGU widget to this form"""
        if name != None: e.name = name
        if value != None: e.value = value
        self._elist.append(e)
        self._dirty = 1
    
    def _clean(self):
        # Remove elements from our list if they no longer have an assigned name
        for e in self._elist[:]:
            if not hasattr(e,'name') or e.name == None:
                self._elist.remove(e)
        # Update the name-to-widget mapping
        self._emap = {}
        for e in self._elist:
            self._emap[e.name] = e
        self._dirty = 0
    
    def __getitem__(self,k):
        """Returns the widget instance given the name of the widget"""
        if self._dirty: self._clean()
        return self._emap[k]
    
    def __contains__(self,k):
        """Returns true if this form contains the named widget"""
        if self._dirty: self._clean()
        if k in self._emap: return True
        return False
    
    def results(self):
        """Return a dict of name, widget-value pairs."""
        if self._dirty: self._clean()
        r = {}
        for e in self._elist:
            # Make sure the widget has a 'value' (eg tables do not)
            if (hasattr(e, "value")):
                r[e.name] = e.value
            else:
                r[e.name] = None
        return r
    
    def items(self):
        """Return a list of name, widget pairs."""
        return self.results().items()
    

