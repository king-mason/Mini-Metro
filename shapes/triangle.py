import pygame
from math import sqrt

class Triangle:

    def __init__(self, center, base):
        self.center = center
        self.base = base
        self.height = sqrt(3) * base / 2

    def draw(self, surface, color, edge_color="black", edge_width=0):
        points = ((self.center.x, self.center.y - self.height * 2/3),
                  (self.center.x - self.base / 2, self.center.y + self.height / 3),
                  (self.center.x + self.base / 2, self.center.y + self.height / 3))
        pygame.draw.polygon(surface, color, points)
        if edge_width:
            pygame.draw.polygon(surface, edge_color, points, width=edge_width)
