"""
"""

from . import pguglobals

class Style:
    """The class used by widget for the widget.style
    
    This object is used mainly as a dictionary, accessed via widget.style.attr, 
    as opposed to widget.style['attr'].  It automatically grabs information 
    from the theme via value = theme.get(widget.cls,widget.pcls,attr)

    """
    def __init__(self, obj, dict):
        self.obj = obj
        for k,v in dict.items(): self.__dict__[k]=v

    def __getattr__(self, attr):
        value = pguglobals.app.theme.get(self.obj.cls, self.obj.pcls, attr)

        if attr in (
            'border_top','border_right','border_bottom','border_left',
            'padding_top','padding_right','padding_bottom','padding_left',
            'margin_top','margin_right','margin_bottom','margin_left',
            'align','valign','width','height',
            ): self.__dict__[attr] = value
        return value

    def __setattr__(self, attr, value):
        self.__dict__[attr] = value

