"""Constants.

From pygame:
    QUIT
    MOUSEBUTTONDOWN
    MOUSEBUTTONUP
    MOUSEMOTION
    KEYDOWN

PGU specific:
    ENTER
    EXIT
    BLUR
    FOCUS
    CLICK
    CHANGE
    OPEN
    CLOSE
    INIT

Other:
    NOATTR

"""
import pygame

from pygame.locals import QUIT, MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION, KEYDOWN, USEREVENT
ENTER = pygame.locals.USEREVENT + 0
EXIT = pygame.locals.USEREVENT + 1
BLUR = pygame.locals.USEREVENT + 2
FOCUS = pygame.locals.USEREVENT + 3
CLICK = pygame.locals.USEREVENT + 4
CHANGE = pygame.locals.USEREVENT + 5
OPEN = pygame.locals.USEREVENT + 6
CLOSE = pygame.locals.USEREVENT + 7
INIT = 'init'

class NOATTR: 
    pass


