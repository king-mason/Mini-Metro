import pygame

from point import Point


class Pentagon:

    def __init__(self, center, circumradius):
        self.center = center
        self.circumradius = circumradius
        self.edge_length = circumradius * 0.8507
    
    def draw(self, surface: pygame.surface.Surface, color, edge_color="black", edge_width=0):

        starting_point = Point(self.center.x, self.center.y - self.circumradius)
        points = []
        for i in range(5):
            points.append(starting_point.rotate(self.center, i*360/5))

        pygame.draw.polygon(surface, color, points)

        if edge_width:
            pygame.draw.polygon(surface, edge_color, points, width=int(edge_width))