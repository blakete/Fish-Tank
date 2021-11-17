from random import randint 
import numpy as np
from six import byte2int
import tensorflow as tf

class Cell:
    def __init__(self, canvas, x, y, r=10, color="blue", species=0, vision_distance=20):
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
        initializer = tf.random_normal_initializer(mean=0, stddev=1, seed=None)
        initializer2 = tf.random_normal_initializer(mean=0, stddev=2, seed=None)
        zeros_initializer = tf.zeros_initializer()
        self.R1 = tf.Variable(zeros_initializer(shape=[1,5]))
        self.W1 = tf.Variable(initializer(shape=[4+5, 5], dtype=tf.float32))
        self.W2 = tf.Variable(initializer(shape=[5, 2], dtype=tf.float32))
        self.b1 = tf.Variable(initializer(shape=[5], dtype=tf.float32))
        self.b2 = tf.Variable(initializer(shape=[2], dtype=tf.float32))
    

    def set_nn_weights(self, weights):
        self.R1, self.W1, self.W2, self.b1, self.b2 = weights
    
    def get_nn_weights(self):
        return self.R1, self.W1, self.W2, self.b1, self.b2

    def multilayer_perceptron(self, x):
        x = np.expand_dims(x.astype('float32'), axis=0)
        l0 = tf.concat([x, self.R1], axis=1)
        l1 = tf.nn.sigmoid(tf.matmul(l0, self.W1) + self.b1)
        self.R1 = l1
        l2 = tf.matmul(l1, self.W2) + self.b2
        return l2

    def calculate_vector(self):
        return randint(-2, 2), randint(-2, 2)

    def calc_movement(self):
        # 1x4 --> 4x6 --> 6x2
        # print(f"fov: {self.fov}")
        nn_output = self.multilayer_perceptron(self.fov)
        return nn_output[0]

    def advance(self, canvas, w, h):
        # TODO calculate movement vector with fov
        move_vector = self.calc_movement()
        # print(f"nn output: {move_vector}")
        if (move_vector[0] < 0):
            x_vel = -1
        else:
            x_vel = 1
        if (move_vector[1] < 0):
            y_vel = -1
        else: 
            y_vel = 1
        # print(f"move vector: ({x_vel}, {y_vel})")
        # x_vel, y_vel = 0, 1# self.calculate_vector()

        # make sure cell not moving outside canvas
        future_x = self.x + x_vel
        if (future_x < self.r or future_x > w - self.r):
            x_vel = 0
        else:
            self.x = future_x
        future_y = self.y + y_vel
        if (future_y < self.r or future_y > h - self.r):
            y_vel = 0
        else:
            self.y = future_y

        # move cell body
        canvas.move(self.circle, x_vel, y_vel)
        # move eyes
        canvas.coords(self.horizontal_eye, self.x-self.vision_distance, self.y, self.x+self.vision_distance, self.y)
        canvas.coords(self.vertical_eye, self.x, self.y-self.vision_distance, self.x, self.y+self.vision_distance)

    def eat(self, food):
        self.consumed += 1
        print(f"Cell consumed {self.consumed}")
        

    def get_eye_coords(self, canvas, eye):
        if eye == "h":
            return canvas.coords(self.horizontal_eye)
        elif eye == "v":
            return canvas.coords(self.vertical_eye)
        return None

    def get_coords(self):
        return self.x, self.y
    
    def self_destruct(self, canvas):
        canvas.delete(self.circle)
        canvas.delete(self.horizontal_eye)
        canvas.delete(self.vertical_eye)

    # TODO save cell neural network weights to file
    # TODO save cell death state, fitness, nn weights