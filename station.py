from point import Point
# from carrier import Carrier
from typing import List
from old_files.config import station_capacity, station_radius

import pygame

# from shapes.rectangle import Rectangle
# from shapes.circle import Circle
# from shapes.triangle import Triangle
# from shapes.cross import Cross
# from shapes.rhombus import Rhombus
# from shapes.pentagon import Pentagon
# from shapes.star import Star
# from shapes.droplet import Droplet
# from shapes.diamond import Diamond
# from shapes.lemon import Lemon


EDGE_WIDTH = 4


class Station():

    def __init__(self, shape: str, position: Point, contour=None, lines: List[int]=None):
        
        self.shape_type = shape.upper()
        self.position = position
        self.set_shape(self.shape_type, station_radius)
        self.contour = contour
        if lines:
            self.lines = lines
        else:
            self.lines = []

    def set_shape(self, shape: str, size: int):
        shapes = ["SQUARE", "CIRCLE", "TRIANGLE", "CROSS", "RHOMBUS",
                  "PENTAGON", "STAR", "DROPLET", "DIAMOND", "LEMON"]
        self.shape_index = shapes.index(shape.upper())
        # shape_map = {
        #     "SQUARE": Rectangle(self.position, size*2, size*2),
        #     "CIRCLE": Circle(self.position, size),
        #     "TRIANGLE": Triangle(self.position, size*2),
        #     "CROSS": Cross(self.position, size*2),
        #     "RHOMBUS": Rhombus(self.position, size),
        #     "PENTAGON": Pentagon(self.position, size),
        #     "STAR": Star(self.position, size, size/2),
        #     "DROPLET": Droplet(self.position, size),
        #     "DIAMOND": Diamond(self.position, size),
        #     "LEMON": Lemon(self.position, size*2)
        # }
        # self.shape = shape_map[shape.upper()]

    def get_distance(self, other):
        if type(other) is Station:
            return ((self.position[0] - other.position[0]) ** 2 + (self.position[1] - other.position[1]) ** 2) ** 0.5
        if type(other) is Point:
            return ((self.position[0] - other.x) ** 2 + (self.position[1] - other.y) ** 2) ** 0.5
        raise TypeError('Unsupported type for function get_distance():', type(other))
    
    def upgrade_station(self):
        pass

    def draw(self, surface: pygame.surface.Surface):
        self.shape.draw(surface, color="white", edge_color="black", edge_width=EDGE_WIDTH)

    def __str__(self):
        return f"Station: {self.shape_type} at {tuple(self.position)}"

    def __repr__(self):
        return f"Station{tuple(self.position)}:{self.shape_type}"
    
    def __hash__(self):
        return hash(str(self))

    def __eq__(self,other):
        return self.shape_type == other.shape_type and self.position == other.position
