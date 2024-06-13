import math


class Point:

    def __init__(self, x: float = 0, y: float = 0):
        self.x = x
        self.y = y
    
    def rotate(self, center, angle, degrees=True):
        
        if degrees:
            # Convert angle to radians
            angle = math.radians(angle)

        # Translate the point to the origin
        translated_point = Point(self.x - center.x, self.y - center.y)

        rotated_point = Point()

        # Perform the rotation
        rotated_point.x = translated_point.x * math.cos(angle) - translated_point.y * math.sin(angle)
        rotated_point.y = translated_point.x * math.sin(angle) + translated_point.y * math.cos(angle)

        # Translate the point back to its original position
        rotated_point = round(rotated_point + center, 2)

        return rotated_point
    
    def get_distance(self, other):
        if type(other) is Point:
            return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5
        raise TypeError('Unsupported type for function get_distance():', type(other))
    
    def __str__(self):
        return f"Point: ({self.x}, {self.y})"

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    def __iter__(self):
        yield self.x
        yield self.y
    
    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        else:
            raise IndexError("Point index out of range")
    
    def __len__(self):
        return 2

    def __add__(self, other_point):
        return Point(self.x + other_point.x, self.y + other_point.y)

    def __sub__(self, other_point):
        return Point(self.x - other_point.x, self.y - other_point.y)

    def __round__(self, decimals):
        return Point(round(self.x, decimals), round(self.y, decimals))
    
    def __eq__(self, other_point):
        return self.x == other_point.x and self.y == other_point.y
    
    def __ne__(self, other_point):
        return self.x != other_point.x or self.y != other_point.y
    

# a = Point(100, 100)

# b = Point(110, 100)

# print(b.rotate(a, 45))