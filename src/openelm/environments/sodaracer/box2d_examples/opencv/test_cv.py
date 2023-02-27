#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Based on the paper:
    "Simulation of 2D physics of objects captured by web camera using OpenCV
    and Box2D"
    by Michal Sedlak (sedlak@stuba.sk)
    Presented at the 18th International Conference on Process Control, 6/14-17,
        2011
    Available here: http://www.kirp.chtf.stuba.sk/pc11/data/papers/053.pdf

If no webcam is detected, the image from the paper will be used.
Requires: pygame, OpenCV
pygame: www.pygame.org
OpenCV: opencv.willowgarage.com (simple installer for windows here
        http://www.lfd.uci.edu/~gohlke/pythonlibs/ )
"""

from __future__ import print_function
import cv
import Box2D as b2
from triangulate_seidel import Triangulator

THRESHOLD = 55
PPM = 20.0  # pixels per meter
TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
SCREEN_OFFSETX, SCREEN_OFFSETY = SCREEN_WIDTH * 1.0 / 4.0, SCREEN_HEIGHT


class CVObject(object):

    def __init__(self, world, image=None):
        self.world = world
        self.contour_bodies = []
        self.x_scale = 30.0
        self.y_scale = 30.0
        self.object_bodies = {}
        self._max_recursion = 200
        if image is not None:
            self.bodies_from_image(image)

    def create_body(self, cont, h, v, density):
        cont_m = [(point[0] / self.x_scale, point[1] / self.y_scale)
                  for point in cont]

        if v == 0:
            body = self.world.CreateStaticBody(position=(0, 0))
            body.CreateEdgeChain(cont_m)
        elif v == 1:
            try:
                body = self.object_bodies[h]
            except:
                body = self.world.CreateDynamicBody(position=(0, 0))
                body.CreateEdgeChain(cont_m)

                t = Triangulator(cont_m)
                for poly in t.triangles():
                    # poly = ((x1, y1), (x2, y2), (x3, y3))
                    body.CreatePolygonFixture(vertices=poly, density=1.0,
                                              restitution=0.0, friction=0.0)

    def _objects_from_contours(self, recurs, cont, h=0, v=0, ret=None):
        if recurs > self._max_recursion:
            return

        if v > 0:
            density = 1.0
        else:
            density = 0.0

        if len(cont) > 8:
            body = self.create_body(cont, h, v, density)
            if ret is not None:
                ret.append(body)

        if cont.v_next():
            v += 1
            self._objects_from_contours(recurs + 1, cont.v_next(), h, v, ret)
            v -= 1

        if cont.h_next():
            h += 1
            self._objects_from_contours(recurs + 1, cont.h_next(), h, v, ret)
            h -= 1

    def objects_from_contours(self, cont):
        ret = []
        self._objects_from_contours(0, cont, 0, 0, ret)
        return ret

    def detect_outline(self, image, threshold=THRESHOLD):
        img_size = cv.GetSize(image)
        grayscale = cv.CreateImage(img_size, 8, 1)
        cv.CvtColor(image, grayscale, cv.CV_BGR2GRAY)
        cv.EqualizeHist(grayscale, grayscale)

        storage = cv.CreateMemStorage(0)
        cv.Threshold(grayscale, grayscale, threshold, 255, cv.CV_THRESH_BINARY)
        contours = cv.FindContours(grayscale, cv.CreateMemStorage(),
                                   cv.CV_RETR_TREE, cv.CV_CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            return cv.ApproxPoly(contours, storage, cv.CV_POLY_APPROX_DP, 1.5, 1)
        return contours

    def bodies_from_image(self, image, flip=True):
        if flip:
            flipped = cv.CreateImage(
                (image.width, image.height), image.depth, image.channels)
            cv.Flip(image, flipped, flipMode=0)
            image = flipped

        contours = self.detect_outline(image)
        return self.objects_from_contours(contours)

camera = None


def get_image():
    # Grab an image from the camera
    global camera
    try:
        # raise # <-- uncomment to force using the journal image
        camera = cv.CaptureFromCAM(-1)  # need to hold onto this instance
        image = cv.QueryFrame(camera)
        if image is None:
            raise Exception('Invalid image captured?')
    except Exception as ex:
        print('Unable to grab an image from the camera. (%s)' % ex)
        print('Using the journal image instead.')
        image = cv.LoadImage('journal_image.png')
    return image


def main():
    # Create the world
    world = b2.b2World(gravity=(0, -10), doSleep=True)

    # Get the OpenCV/Box2D object
    image = get_image()
    # Bring that image into Box2D
    cvo = CVObject(world, image)

    # ... and the rest is standard pygame visualization
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
    pygame.display.set_caption('pybox2d/OpenCV example')
    clock = pygame.time.Clock()

    # And a static body to hold the ground shape
    world.CreateStaticBody(
        position=(0, 0),
        shapes=b2.b2PolygonShape(box=(50, 1)),
    )

    colors = {
        b2.b2_staticBody: (255, 255, 255, 255),
        b2.b2_dynamicBody: (127, 127, 127, 255),
    }

    def fix_vertices(vertices):
        return [(int(SCREEN_OFFSETX + v[0]), int(SCREEN_OFFSETY - v[1]))
                for v in vertices]

    def _draw_polygon(polygon, screen, body, fixture):
        transform = body.transform
        vertices = fix_vertices(
            [transform * v * PPM for v in polygon.vertices])
        pygame.draw.polygon(screen,
                            [c / 2.0 for c in colors[body.type]],
                            vertices, 0)
        pygame.draw.polygon(screen, colors[body.type], vertices, 1)
    b2.b2PolygonShape.draw = _draw_polygon

    def _draw_circle(circle, screen, body, fixture):
        position = fix_vertices([body.transform * circle.pos * PPM])[0]
        pygame.draw.circle(screen, colors[body.type], position,
                           int(circle.radius * PPM))
    b2.b2CircleShape.draw = _draw_circle

    def _draw_edge(edge, screen, body, fixture):
        vertices = fix_vertices([body.transform * edge.vertex1 * PPM,
                                 body.transform * edge.vertex2 * PPM])
        pygame.draw.line(screen, colors[body.type], vertices[0], vertices[1])
    b2.b2EdgeShape.draw = _draw_edge

    display_image = True
    if display_image:
        # opencv image -> pygame image
        image_rgb = cv.CreateMat(image.height, image.width, cv.CV_8UC3)
        cv.CvtColor(image, image_rgb, cv.CV_BGR2RGB)
        pg_img = pygame.image.frombuffer(
            image_rgb.tostring(), cv.GetSize(image_rgb), "RGB")
        pg_img.set_alpha(100)

    running = True
    while running:
        # Check the event queue
        for event in pygame.event.get():
            if (event.type == QUIT
                    or (event.type == KEYDOWN and event.key == K_ESCAPE)):
                # The user closed the window or pressed escape
                running = False

        screen.fill((0, 0, 0, 0))
        if display_image:
            screen.blit(pg_img, (0, 0))

        # Draw the world
        for body in world.bodies:
            for fixture in body.fixtures:
                fixture.shape.draw(screen, body, fixture)

        # Make Box2D simulate the physics of our world for one step.
        world.Step(TIME_STEP, 10, 10)

        # Flip the screen and try to keep at the target FPS
        pygame.display.flip()
        clock.tick(TARGET_FPS)

    pygame.quit()
    print('Done!')


if __name__ == '__main__':
    import pygame
    from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)
    main()
