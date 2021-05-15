class Rectangle:
    def __init__(self, lenght, width, **kwargs):
        self.lenght = lenght
        self.width = width
        super().__init__(**kwargs)

    def area(self):
        return self.lenght * self.width

    def perimeter(self):
        return 2 * self.lenght + 2 * self.width

class Square(Rectangle):
    def __init__(self, lenght, **kwargs):
        super().__init__(lenght=lenght, width=lenght, **kwargs)

class Cube(Square):
    def surface_area(self):
        face_area = super().area()
        return 6 * face_area

    def volume(self):
        face_area = super().area()
        return face_area * self.lenght

class Triangle:
    def __init__(self, base, height, **kwargs) -> None:
        self.base = base
        self.height = height
        super().__init__(**kwargs)
    def tri_area(self):
        return .5 * self.base * self.height
    
class RightPyramid(Square, Triangle):
    def __init__(self, base, slant_height, **kwargs) -> None:
        self.base = base
        self.slant_height = slant_height
        kwargs['height'] = slant_height
        kwargs['lenght'] = base
        super().__init__(base=base, **kwargs)
    def area(self):
        base_area = super().area()
        perimeter = super().perimeter()
        return 0.5 * perimeter * self.slant_height + base_area
    def area2(self):
        base_area = super().area()
        triangle_area = super().tri_area()
        return triangle_area * 4 + base_area

