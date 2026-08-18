import pygame

from point import Point
from math import sqrt


class Rhombus:

    def __init__(self, center, circumradius):
        self.center = center
        self.circumradius = circumradius
        self.edge_length = circumradius * sqrt(2)
    
    def draw(self, surface: pygame.surface.Surface, color, edge_color="black", edge_width=0):

        # square = pygame.Surface((self.edge_length, self.edge_length), pygame.SRCALPHA)
        # square.fill(color)
        # pygame.draw.rect(square, edge_color, (self.center.x, self.center.y, self.edge_length, self.edge_length), edge_width)  # Draw a black border

        # rotated_square = pygame.transform.rotate(square, 45)
        
        # surface.blit(rotated_square, self.center)

        starting_point = Point(self.center.x, self.center.y - self.circumradius)
        points = []
        for i in range(4):
            points.append(starting_point.rotate(self.center, i*90))

        pygame.draw.polygon(surface, color, points)

        if edge_width:
            pygame.draw.polygon(surface, edge_color, points, width=int(edge_width*1.5))