import pygame
from point import Point

class Diamond:

    def __init__(self, center, length):
        self.center = center
        self.length = length
        self.bottom = center + Point(0, length)
        self.top = center - Point(0, length)

    def draw(self, surface, color, edge_color="black", edge_width=0):
        points = (self.bottom,
                  self.top.rotate(self.bottom, -40),
                  self.top - Point(self.length*.75, 0),
                  self.top + Point(self.length*.75, 0),
                  self.top.rotate(self.bottom, 40))
        pygame.draw.polygon(surface, color, points)
        if edge_width:
            pygame.draw.polygon(surface, edge_color, points, width=int(edge_width*1.2))