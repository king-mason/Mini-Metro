import pygame
from point import Point
import math


class Lemon:

    def __init__(self, center, length):
        self.center = center
        self.length = length

    def draw(self, surface, color, edge_color="black", edge_width=0):
        radius = self.length * 0.6
        hypotenuse = math.sqrt(radius**2 - (self.length**2 / 4))
        leg = hypotenuse / math.sqrt(2)
        offset = Point(leg, leg)
        
        circle_1_surface = pygame.Surface((2 * radius, 2 * radius), pygame.SRCALPHA)
        circle_1_surface.fill((0,0,0))
        circle_1_surface.set_colorkey((0,0,0))
        circle_1_rect = pygame.draw.circle(circle_1_surface, color, (radius, radius), radius)
        circle_1_mask = pygame.mask.from_surface(circle_1_surface)
        circle_1_rect.center = self.center - offset
        # surface.blit(circle_1_surface, circle_1_rect)

        circle_2_surface = pygame.Surface((2 * radius, 2 * radius), pygame.SRCALPHA)
        circle_2_surface.fill((0,0,0))
        circle_2_surface.set_colorkey((0,0,0))
        circle_2_rect = pygame.draw.circle(circle_2_surface, color, (radius, radius), radius)
        circle_2_mask = pygame.mask.from_surface(circle_2_surface)
        circle_2_rect.center = self.center + offset
        # surface.blit(circle_2_surface, circle_2_rect)
        # w, h = circle_2_surface.get_size()
        # for x in range(w):
        #     for y in range(h):
        #         if circle_1_surface.get_at((x, y))[0] != 0:
        #             print(x, y)
        # surface.blit(circle_2_mask.to_surface(), circle_2_rect)

        new_mask = circle_1_mask.overlap_mask(circle_2_mask, offset + offset)
        new_surf = new_mask.to_surface()
        new_surf.set_colorkey((0,0,0))
        new_surf_rect = new_surf.get_rect()
        surface.blit(new_surf, circle_1_rect)

        if edge_width:
            arc_1_angle = math.acos(1 - self.length**2 / (2 * radius**2))
            start_angle = (7 * math.pi / 4) - arc_1_angle / 2
            stop_angle = (7 * math.pi / 4) + arc_1_angle / 2
            pygame.draw.arc(surface, edge_color, circle_1_rect, start_angle, stop_angle, edge_width)
            pygame.draw.arc(surface, edge_color, circle_2_rect, start_angle - math.pi, stop_angle - math.pi, edge_width)
        
