from random import randint 
import numpy as np
from six import byte2int
import tensorflow as tf

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
        initializer = tf.random_normal_initializer(mean=0, stddev=2.)
        k=3
        self.W1 = tf.Variable(initializer(shape=[4, 5], dtype=tf.float32))
        self.W2 = tf.Variable(initializer(shape=[5, 2], dtype=tf.float32))
        self.b1 = tf.Variable(initializer(shape=[5], dtype=tf.float32))
        self.b2 = tf.Variable(initializer(shape=[2], dtype=tf.float32))
    
    def multilayer_perceptron(self, x):
        l0 = np.expand_dims(x.astype('float32'), axis=0)
        l1 = tf.nn.sigmoid(tf.matmul(l0, self.W1) + self.b1)
        l2 = tf.matmul(l1, self.W2) + self.b2
        return l2

    def calculate_vector(self):
        return randint(-2, 2), randint(-2, 2)

    def calc_movement(self):
        # 1x4 --> 4x6 --> 6x2
        print("---Calculating Movement---")
        print(f"fov: {self.fov}")
        nn_output = self.multilayer_perceptron(self.fov)
        return nn_output[0]



    def advance(self, canvas, w, h):
        # TODO calculate movement vector with fov
        move_vector = self.calc_movement()
        print(f"nn output: {move_vector}")
        if (move_vector[0] < 0):
            x_vel = -1
        else:
            x_vel = 1
        if (move_vector[1] < 0):
            y_vel = -1
        else: 
            y_vel = 1
        print(f"cell move vector: ({x_vel}, {y_vel})")
        # x_vel, y_vel = 0, 1# self.calculate_vector()

        # make sure cell not moving outside canvas
        future_x = self.x + x_vel
        if (future_x < self.r or future_x > w - self.r):
            print("hit x boundry")
            x_vel = 0
        else:
            self.x = future_x
        future_y = self.y + y_vel
        if (future_y < self.r or future_y > h - self.r):
            print("hit y boundry")
            y_vel = 0
        else:
            self.y = future_y

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
        
