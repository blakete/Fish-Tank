from random import randint 
import numpy as np
# import tensorflow as tf

class Cell:
    def __init__(self, canvas, x, y, r=10, color="blue", species=0, vision_distance=40):
        self.x = x
        self.y = y
        self.r = r
        self.color = color
        self.species = species
        self.circle = canvas.create_oval(x-r, y-r, x+r, y+r, fill=color)
        # create receptive field lines
        self.vision_distance = vision_distance
        self.horizontal_eye = canvas.create_line(x-vision_distance, y, x+vision_distance, y)
        self.vertical_eye = canvas.create_line(x, y-vision_distance, x, y+vision_distance)
        self.consumed = 0 # num foods eatin
        self.fov = np.asarray([0,0,0,0])

    def calculate_vector(self):
        return randint(-2, 2), randint(-2, 2)

    # def calc_movement(self):
    #     # 1x4 --> 


    def advance(self, canvas):
        # TODO calculate movement vector with fov
        x_vel, y_vel = 0, 1# self.calculate_vector()
        self.x += x_vel
        self.y += y_vel
        # move cell body
        canvas.move(self.circle, x_vel, y_vel)
        # move eyes
        canvas.coords(self.horizontal_eye, self.x-self.vision_distance, self.y, self.x+self.vision_distance, self.y)
        canvas.coords(self.vertical_eye, self.x, self.y-self.vision_distance, self.x, self.y+self.vision_distance)

    def eat(self, food):
        self.consumed += food.points

    def get_eye_coords(self, canvas, eye):
        if eye == "h":
            return canvas.coords(self.horizontal_eye)
        elif eye == "v":
            return canvas.coords(self.vertical_eye)
        return None

    def get_coords(self):
        return self.x, self.y
        
