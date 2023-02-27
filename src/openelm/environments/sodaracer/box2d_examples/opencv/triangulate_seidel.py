#
# Poly2Tri
# Copyright (c) 2009, Mason Green
# http://code.google.com/p/poly2tri/
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice,
# self list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# self list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of Poly2Tri nor the names of its contributors may be
# used to endorse or promote products derived from self software without specific
# prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
from random import shuffle
from math import atan2, sqrt

#
# Based on Raimund Seidel'e paper "A simple and fast incremental randomized
# algorithm for computing trapezoidal decompositions and for triangulating
# polygons" (Ported from poly2tri)
#

# Shear transform. May effect numerical robustness
SHEAR = 1e-3


class Point(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.next, self.prev = None, None

    def __sub__(self, other):
        if isinstance(other, Point):
            return Point(self.x - other.x, self.y - other.y)
        else:
            return Point(self.x - other, self.y - other)

    def __add__(self, other):
        if isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        else:
            return Point(self.x + other, self.y + other)

    def __mul__(self, f):
        return Point(self.x * f, self.y * f)

    def __div__(self, a):
        return Point(self.x / a, self.y / a)

    def cross(self, p):
        return self.x * p.y - self.y * p.x

    def dot(self, p):
        return self.x * p.x + self.y * p.y

    def length(self):
        return sqrt(self.x * self.x + self.y * self.y)

    def normalize(self):
        return self / self.length()

    def less(self, p):
        return self.x < p.x

    def neq(self, other):
        return other.x != self.x or other.y != self.y

    def clone(self):
        return Point(self.x, self.y)


def orient2d(pa, pb, pc):
    acx = pa.x - pc.x
    bcx = pb.x - pc.x
    acy = pa.y - pc.y
    bcy = pb.y - pc.y
    return acx * bcy - acy * bcx


class Edge(object):

    def __init__(self, p, q):
        self.p = p
        self.q = q
        self.slope = (q.y - p.y) / (q.x - p.x) if q.x - p.x != 0 else 0
        self.b = p.y - (p.x * self.slope)
        self.above, self.below = None, None
        self.mpoints = [p, q]

    def is_above(self, point):
        return orient2d(self.p, self.q, point) < 0

    def is_below(self, point):
        return orient2d(self.p, self.q, point) > 0

    def add_mpoint(self, point):
        for mp in self.mpoints:
            if not mp.neq(point):
                return
        self.mpoints.append(point)


class Trapezoid(object):

    def __init__(self, left_point, right_point, top, bottom):
        self.left_point = left_point
        self.right_point = right_point
        self.top = top
        self.bottom = bottom
        self.upper_left = None
        self.upper_right = None
        self.lower_left = None
        self.lower_right = None
        self.inside = True
        self.sink = None
        self.key = hash(self)

    def update_left(self, ul, ll):
        self.upper_left = ul
        if ul is not None:
            ul.upper_right = self
        self.lower_left = ll
        if ll is not None:
            ll.lower_right = self

    def update_right(self, ur, lr):
        self.upper_right = ur
        if ur is not None:
            ur.upper_left = self
        self.lower_right = lr
        if lr is not None:
            lr.lower_left = self

    def update_left_right(self, ul, ll, ur, lr):
        self.upper_left = ul
        if ul is not None:
            ul.upper_right = self
        self.lower_left = ll
        if ll is not None:
            ll.lower_right = self
        self.upper_right = ur
        if ur is not None:
            ur.upper_left = self
        self.lower_right = lr
        if lr is not None:
            lr.lower_left = self

    def trim_neighbors(self):
        if self.inside:
            self.inside = False
            if self.upper_left is not None:
                self.upper_left.trim_neighbors()
            if self.lower_left is not None:
                self.lower_left.trim_neighbors()
            if self.upper_right is not None:
                self.upper_right.trim_neighbors()
            if self.lower_right is not None:
                self.lower_right.trim_neighbors()

    def contains(self, point):
        return (point.x > self.left_point.x and point.x < self.right_point.x and
                self.top.is_above(point) and self.bottom.is_below(point))

    def vertices(self):
        v1 = line_intersect(self.top, self.left_point.x)
        v2 = line_intersect(self.bottom, self.left_point.x)
        v3 = line_intersect(self.bottom, self.right_point.x)
        v4 = line_intersect(self.top, self.right_point.x)
        return v1, v2, v3, v4

    def add_points(self):
        if self.left_point is not self.bottom.p:
            self.bottom.add_mpoint(self.left_point)
        if self.right_point is not self.bottom.q:
            self.bottom.add_mpoint(self.right_point)
        if self.left_point is not self.top.p:
            self.top.add_mpoint(self.left_point)
        if self.right_point is not self.top.q:
            self.top.add_mpoint(self.right_point)

    def area(self):
        p = list(self.vertices())
        return 0.5 * abs(sum(x0 * y1 - x1 * y0
                             for ((x0, y0), (x1, y1)) in self.segments(p)))

    def segments(self, p):
        return zip(p, p[1:] + [p[0]])


def line_intersect(edge, x):
    y = edge.slope * x + edge.b
    return x, y


class Triangulator(object):

    ##
    # Number of points should be > 3
    ##
    def __init__(self, poly_line):
        self.polygons = []
        self.trapezoids = []
        self.xmono_poly = []
        self.edge_list = self.init_edges(poly_line)
        self.trapezoidal_map = TrapezoidalMap()
        self.bounding_box = self.trapezoidal_map.bounding_box(self.edge_list)
        self.query_graph = QueryGraph(isink(self.bounding_box))

        self.process()

    def triangles(self):
        triangles = []
        for p in self.polygons:
            verts = []
            for v in p:
                verts.append((v.x, v.y))
            triangles.append(verts)
        return triangles

    def trapezoid_map(self):
        return self.trapezoidal_map.map

    # Build the trapezoidal map and query graph
    def process(self):
        for edge in self.edge_list:
            traps = self.query_graph.follow_edge(edge)
            for t in traps:
                # Remove old trapezods
                del self.trapezoidal_map.map[t.key]
                # Bisect old trapezoids and create new
                cp = t.contains(edge.p)
                cq = t.contains(edge.q)
                if cp and cq:
                    tlist = self.trapezoidal_map.case1(t, edge)
                    self.query_graph.case1(t.sink, edge, tlist)
                elif cp and not cq:
                    tlist = self.trapezoidal_map.case2(t, edge)
                    self.query_graph.case2(t.sink, edge, tlist)
                elif not cp and not cq:
                    tlist = self.trapezoidal_map.case3(t, edge)
                    self.query_graph.case3(t.sink, edge, tlist)
                else:
                    tlist = self.trapezoidal_map.case4(t, edge)
                    self.query_graph.case4(t.sink, edge, tlist)
                # Add new trapezoids to map
                for t in tlist:
                    self.trapezoidal_map.map[t.key] = t
            self.trapezoidal_map.clear()

        # Mark outside trapezoids w/ depth-first search
        for k, t in self.trapezoidal_map.map.items():
            self.mark_outside(t)

        # Collect interior trapezoids
        for k, t in self.trapezoidal_map.map.items():
            if t.inside:
                self.trapezoids.append(t)
                t.add_points()

        # Generate the triangles
        self.create_mountains()

    def mono_polies(self):
        polies = []
        for x in self.xmono_poly:
            polies.append(x.monoPoly)
        return polies

    def create_mountains(self):
        for edge in self.edge_list:
            if len(edge.mpoints) > 2:
                mountain = MonotoneMountain()
                points = merge_sort(edge.mpoints)
                for p in points:
                    mountain.add(p)
                mountain.process()
                for t in mountain.triangles:
                    self.polygons.append(t)
                self.xmono_poly.append(mountain)

    def mark_outside(self, t):
        if t.top is self.bounding_box.top or t.bottom is self.bounding_box.bottom:
            t.trim_neighbors()

    def init_edges(self, points):
        edge_list = []
        size = len(points)
        for i in range(size):
            j = i + 1 if i < size - 1 else 0
            p = points[i][0], points[i][1]
            q = points[j][0], points[j][1]
            edge_list.append((p, q))
        return self.order_edges(edge_list)

    def order_edges(self, edge_list):
        edges = []
        for e in edge_list:
            p = shear_transform(e[0])
            q = shear_transform(e[1])
            if p.x > q.x:
                edges.append(Edge(q, p))
            else:
                edges.append(Edge(p, q))
        # Randomized incremental algorithm
        shuffle(edges)
        return edges


def shear_transform(point):
    return Point(point[0] + SHEAR * point[1], point[1])


def merge_sort(l):
    if len(l) > 1:
        lleft = merge_sort(l[:len(l) / 2])
        lright = merge_sort(l[len(l) / 2:])
        p1, p2, p = 0, 0, 0
        while p1 < len(lleft) and p2 < len(lright):
            if lleft[p1].x < lright[p2].x:
                l[p] = lleft[p1]
                p += 1
                p1 += 1
            else:
                l[p] = lright[p2]
                p += 1
                p2 += 1
        if p1 < len(lleft):
            l[p:] = lleft[p1:]
        elif p2 < len(lright):
            l[p:] = lright[p2:]
        else:
            print("internal error")
    return l


class TrapezoidalMap(object):

    def __init__(self):
        self.map = {}
        self.margin = 50.0
        self.bcross = None
        self.tcross = None

    def clear(self):
        self.bcross = None
        self.tcross = None

    def case1(self, t, e):
        trapezoids = []
        trapezoids.append(Trapezoid(t.left_point, e.p, t.top, t.bottom))
        trapezoids.append(Trapezoid(e.p, e.q, t.top, e))
        trapezoids.append(Trapezoid(e.p, e.q, e, t.bottom))
        trapezoids.append(Trapezoid(e.q, t.right_point, t.top, t.bottom))
        trapezoids[0].update_left(t.upper_left, t.lower_left)
        trapezoids[1].update_left_right(
            trapezoids[0], None, trapezoids[3], None)
        trapezoids[2].update_left_right(
            None, trapezoids[0], None, trapezoids[3])
        trapezoids[3].update_right(t.upper_right, t.lower_right)
        return trapezoids

    def case2(self, t, e):
        rp = e.q if e.q.x == t.right_point.x else t.right_point
        trapezoids = []
        trapezoids.append(Trapezoid(t.left_point, e.p, t.top, t.bottom))
        trapezoids.append(Trapezoid(e.p, rp, t.top, e))
        trapezoids.append(Trapezoid(e.p, rp, e, t.bottom))
        trapezoids[0].update_left(t.upper_left, t.lower_left)
        trapezoids[1].update_left_right(
            trapezoids[0], None, t.upper_right, None)
        trapezoids[2].update_left_right(
            None, trapezoids[0], None, t.lower_right)
        self.bcross = t.bottom
        self.tcross = t.top
        e.above = trapezoids[1]
        e.below = trapezoids[2]
        return trapezoids

    def case3(self, t, e):
        lp = e.p if e.p.x == t.left_point.x else t.left_point
        rp = e.q if e.q.x == t.right_point.x else t.right_point
        trapezoids = []
        if self.tcross is t.top:
            trapezoids.append(t.upper_left)
            trapezoids[0].update_right(t.upper_right, None)
            trapezoids[0].right_point = rp
        else:
            trapezoids.append(Trapezoid(lp, rp, t.top, e))
            trapezoids[0].update_left_right(
                t.upper_left, e.above, t.upper_right, None)
        if self.bcross is t.bottom:
            trapezoids.append(t.lower_left)
            trapezoids[1].update_right(None, t.lower_right)
            trapezoids[1].right_point = rp
        else:
            trapezoids.append(Trapezoid(lp, rp, e, t.bottom))
            trapezoids[1].update_left_right(
                e.below, t.lower_left, None, t.lower_right)
        self.bcross = t.bottom
        self.tcross = t.top
        e.above = trapezoids[0]
        e.below = trapezoids[1]
        return trapezoids

    def case4(self, t, e):
        lp = e.p if e.p.x == t.left_point.x else t.left_point
        trapezoids = []
        if self.tcross is t.top:
            trapezoids.append(t.upper_left)
            trapezoids[0].right_point = e.q
        else:
            trapezoids.append(Trapezoid(lp, e.q, t.top, e))
            trapezoids[0].update_left(t.upper_left, e.above)
        if self.bcross is t.bottom:
            trapezoids.append(t.lower_left)
            trapezoids[1].right_point = e.q
        else:
            trapezoids.append(Trapezoid(lp, e.q, e, t.bottom))
            trapezoids[1].update_left(e.below, t.lower_left)
        trapezoids.append(Trapezoid(e.q, t.right_point, t.top, t.bottom))
        trapezoids[2].update_left_right(trapezoids[0], trapezoids[
                                        1], t.upper_right, t.lower_right)
        return trapezoids

    def bounding_box(self, edges):
        margin = self.margin
        max = edges[0].p + margin
        min = edges[0].q - margin
        for e in edges:
            if e.p.x > max.x:
                max = Point(e.p.x + margin, max.y)
            if e.p.y > max.y:
                max = Point(max.x, e.p.y + margin)
            if e.q.x > max.x:
                max = Point(e.q.x + margin, max.y)
            if e.q.y > max.y:
                max = Point(max.x, e.q.y + margin)
            if e.p.x < min.x:
                min = Point(e.p.x - margin, min.y)
            if e.p.y < min.y:
                min = Point(min.x, e.p.y - margin)
            if e.q.x < min.x:
                min = Point(e.q.x - margin, min.y)
            if e.q.y < min.y:
                min = Point(min.x, e.q.y - margin)
        top = Edge(Point(min.x, max.y), Point(max.x, max.y))
        bottom = Edge(Point(min.x, min.y), Point(max.x, min.y))
        left = top.p
        right = top.q
        trap = Trapezoid(left, right, top, bottom)
        self.map[trap.key] = trap
        return trap


class Node(object):

    def __init__(self, lchild, rchild):
        self.parent_list = []
        self.lchild = lchild
        self.rchild = rchild
        if lchild is not None:
            lchild.parent_list.append(self)
        if rchild is not None:
            rchild.parent_list.append(self)

    def replace(self, node):
        for parent in node.parent_list:
            if parent.lchild is node:
                parent.lchild = self
            else:
                parent.rchild = self
        self.parent_list += node.parent_list


class Sink(Node):

    def __init__(self, trapezoid):
        super(Sink, self).__init__(None, None)
        self.trapezoid = trapezoid
        trapezoid.sink = self

    def locate(self, edge):
        return self


def isink(trapezoid):
    if trapezoid.sink is None:
        return Sink(trapezoid)
    return trapezoid.sink


class XNode(Node):

    def __init__(self, point, lchild, rchild):
        super(XNode, self).__init__(lchild, rchild)
        self.point = point

    def locate(self, edge):
        if edge.p.x >= self.point.x:
            return self.rchild.locate(edge)
        return self.lchild.locate(edge)


class YNode(Node):

    def __init__(self, edge, lchild, rchild):
        super(YNode, self).__init__(lchild, rchild)
        self.edge = edge

    def locate(self, edge):
        if self.edge.is_above(edge.p):
            return self.rchild.locate(edge)
        if self.edge.is_below(edge.p):
            return self.lchild.locate(edge)
        if edge.slope < self.edge.slope:
            return self.rchild.locate(edge)
        return self.lchild.locate(edge)


class QueryGraph:

    def __init__(self, head):
        self.head = head

    def locate(self, edge):
        return self.head.locate(edge).trapezoid

    def follow_edge(self, edge):
        trapezoids = [self.locate(edge)]
        while(edge.q.x > trapezoids[-1].right_point.x):
            if edge.is_above(trapezoids[-1].right_point):
                trapezoids.append(trapezoids[-1].upper_right)
            else:
                trapezoids.append(trapezoids[-1].lower_right)
        return trapezoids

    def replace(self, sink, node):
        if sink.parent_list:
            node.replace(sink)
        else:
            self.head = node

    def case1(self, sink, edge, tlist):
        yNode = YNode(edge, isink(tlist[1]), isink(tlist[2]))
        qNode = XNode(edge.q, yNode, isink(tlist[3]))
        pNode = XNode(edge.p, isink(tlist[0]), qNode)
        self.replace(sink, pNode)

    def case2(self, sink, edge, tlist):
        yNode = YNode(edge, isink(tlist[1]), isink(tlist[2]))
        pNode = XNode(edge.p, isink(tlist[0]), yNode)
        self.replace(sink, pNode)

    def case3(self, sink, edge, tlist):
        yNode = YNode(edge, isink(tlist[0]), isink(tlist[1]))
        self.replace(sink, yNode)

    def case4(self, sink, edge, tlist):
        yNode = YNode(edge, isink(tlist[0]), isink(tlist[1]))
        qNode = XNode(edge.q, yNode, isink(tlist[2]))
        self.replace(sink, qNode)

PI_SLOP = 3.1


class MonotoneMountain:

    def __init__(self):
        self.size = 0
        self.tail = None
        self.head = None
        self.positive = False
        self.convex_points = set()
        self.mono_poly = []
        self.triangles = []
        self.convex_polies = []

    def add(self, point):
        if self.size is 0:
            self.head = point
            self.size = 1
        elif self.size is 1:
            self.tail = point
            self.tail.prev = self.head
            self.head.next = self.tail
            self.size = 2
        else:
            self.tail.next = point
            point.prev = self.tail
            self.tail = point
            self.size += 1

    def remove(self, point):
        next = point.next
        prev = point.prev
        point.prev.next = next
        point.next.prev = prev
        self.size -= 1

    def process(self):
        self.positive = self.angle_sign()
        self.gen_mono_poly()
        p = self.head.next
        while p.neq(self.tail):
            a = self.angle(p)
            if a >= PI_SLOP or a <= -PI_SLOP or a == 0:
                self.remove(p)
            elif self.is_convex(p):
                self.convex_points.add(p)
            p = p.next
        self.triangulate()

    def triangulate(self):
        while self.convex_points:
            ear = self.convex_points.pop()
            a = ear.prev
            b = ear
            c = ear.next
            triangle = (a, b, c)
            self.triangles.append(triangle)
            self.remove(ear)
            if self.valid(a):
                self.convex_points.add(a)
            if self.valid(c):
                self.convex_points.add(c)
        # assert self.size <= 3, "Triangulation bug, please report"

    def valid(self, p):
        return p.neq(self.head) and p.neq(self.tail) and self.is_convex(p)

    def gen_mono_poly(self):
        p = self.head
        while(p is not None):
            self.mono_poly.append(p)
            p = p.next

    def angle(self, p):
        a = p.next - p
        b = p.prev - p
        return atan2(a.cross(b), a.dot(b))

    def angle_sign(self):
        a = self.head.next - self.head
        b = self.tail - self.head
        return atan2(a.cross(b), a.dot(b)) >= 0

    def is_convex(self, p):
        if self.positive != (self.angle(p) >= 0):
            return False
        return True
