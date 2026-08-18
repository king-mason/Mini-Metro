import pygame

from point import Point


class Star:

    def __init__(self, center, circumradius, inradius):
        self.center = center
        self.circumradius = circumradius
        self.inradius = inradius
    
    def draw(self, surface: pygame.surface.Surface, color, edge_color="black", edge_width=0):

        starting_point = Point(self.center.x, self.center.y - self.circumradius)
        points = []

        for i in range(10):
            if i % 2:
                starting_point.y = self.center.y - self.inradius
                points.append(starting_point.rotate(self.center, i*360/10))
            else:
                starting_point.y = self.center.y - self.circumradius
                points.append(starting_point.rotate(self.center, i*360/10))
                

        pygame.draw.polygon(surface, color, points)

        if edge_width:
            pygame.draw.polygon(surface, edge_color, points, width=int(edge_width))