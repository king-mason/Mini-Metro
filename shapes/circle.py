from point import Point

import pygame

class Circle:

    def __init__(self, center: Point, radius: int):
        self.center = center
        self.radius = radius
    
    def draw(self, surface, color, edge_color="black", edge_width=0):
        pygame.draw.circle(surface, color, self.center, self.radius)
        if edge_width:
            pygame.draw.circle(surface, edge_color, self.center, self.radius, width=edge_width)