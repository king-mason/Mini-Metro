from point import Point
from config import background_color

import pygame
from math import sqrt


class Droplet:

    def __init__(self, center: Point, radius: int):
        self.center = center
        self.radius = radius
        self.top = center - Point(0, radius)
    
    def draw(self, surface, color, edge_color="black", edge_width=0):

        # draw circle
        pygame.draw.circle(surface, color, self.center, self.radius)
        if edge_width:
            pygame.draw.circle(surface, edge_color, self.center, self.radius, width=edge_width)

        # remove sections using triangles
        left_triangle = [self.top, self.top + Point(-self.radius, 0), Point(self.top.x - self.radius, self.top.rotate(self.center, -110).y)]
        right_triangle = [self.top, self.top + Point(self.radius, 0), Point(self.top.x + self.radius, self.top.rotate(self.center, 110).y)]
        pygame.draw.polygon(surface, background_color, left_triangle)
        pygame.draw.polygon(surface, background_color, right_triangle)

        # draw line segments
        pygame.draw.line(surface, edge_color, self.top, self.top.rotate(self.center, -110), edge_width)
        pygame.draw.line(surface, edge_color, self.top, self.top.rotate(self.center, 110), edge_width)


