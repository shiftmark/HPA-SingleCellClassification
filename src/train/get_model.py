import tensorflow as tf

import numpy as np
class GetModel():
    """
    Define the model using a backbone from tf.keras.applications and add custom heads.
    """
    def __init__(self, backbone_name):
        self.backbone_name = backbone_name
        self.applications = tf.keras.applications
        
    def set_backbone(self, input_shape, include_top=False, weights='imagenet', pooling='avg'):
        attributes = getattr(self.applications, self.backbone_name)
        if hasattr(self.applications, self.backbone_name) and callable(attributes):
            attributes(input_shape, include_top, weights, pooling)

    def add_head(dense_neurons, ):
        pass   

b = GetModel('append').set_backbone()

#a = np.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]])
print(b)
# set backbone
# add head
import math

class Rectangle:

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

rectangle = Rectangle(31, 5)

rectangle.solve_for('area', 73)
#rectangle.solve_for('perimeter')
#rectangle.solve_for('diagonal')
rectangle.do_area(3)