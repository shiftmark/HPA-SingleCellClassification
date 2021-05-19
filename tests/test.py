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

import math

class Rectangle2:

    def __init__(self, length, width):
        self.length = length
        self.width = width

    def do_area(self, e):
        area = self.length * self.width * e
        print(f"The area of the rectangle is: {area}")

    def do_perimeter(self):
        perimeter = (self.length + self.width) * 2
        print(f"The perimeter of the rectangle is: {perimeter}")

    def do_diagonal(self):
        diagonal = math.sqrt(self.length ** 2 + self.width ** 2)
        print(f"The diagonal of the rectangle is: {diagonal}")

    def solve_for(self, name, r):
        do = f"do_{name}"
        func = getattr(self, do)
        if hasattr(self, do) and callable(func):
            func(r)

#rectangle2 = Rectangle(31, 5)

#rectangle2.solve_for('area', 73)
#rectangle2.solve_for('perimeter')
#rectangle2.solve_for('diagonal')
#rectangle2.do_area(3)