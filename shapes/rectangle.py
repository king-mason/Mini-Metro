from point import Point

import pygame

class Rectangle:

    def __init__(self, center: Point, width: float, height: float):
        self.center = center
        self.width = width
        self.height = height
    
    def draw(self, surface, color, edge_color="black", edge_width=0):
        rect = pygame.rect.Rect(0, 0, self.width, self.height)
        rect.center = tuple(self.center)
        
        pygame.draw.rect(surface, color, rect)
        if edge_width:
            pygame.draw.rect(surface, edge_color, rect, width=edge_width)
