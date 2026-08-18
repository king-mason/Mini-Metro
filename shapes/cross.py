from point import Point

import pygame

class Cross:

    def __init__(self, center: Point, length: float):
        self.center = center
        self.length = length
    
    def draw(self, surface, color, edge_color="black", edge_width=0):

        if edge_width:
            outline1 = pygame.rect.Rect(0, 0, self.length, self.length*.5, width=edge_width)
            outline1.center = tuple(self.center)
            outline2 = pygame.rect.Rect(0, 0, self.length*.5, self.length, width=edge_width)
            outline2.center = tuple(self.center)

            pygame.draw.rect(surface, edge_color, outline1)
            pygame.draw.rect(surface, edge_color, outline2)

        rect1 = pygame.rect.Rect(0, 0, self.length - 2*edge_width, self.length*.5 - 2*edge_width)
        rect1.center = tuple(self.center)
        rect2 = pygame.rect.Rect(0, 0, self.length*.5 - 2*edge_width, self.length - 2*edge_width)
        rect2.center = tuple(self.center)
        
        pygame.draw.rect(surface, color, rect1)
        pygame.draw.rect(surface, color, rect2)
        
        
