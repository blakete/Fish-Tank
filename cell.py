from random import randint 
import numpy as np
from six import byte2int
import tensorflow as tf
from tkinter import *

class Cell:
    def __init__(self, canvas, x, y, r=10, color="blue", species=0, vision_distance=75):
        self.x = x
        self.y = y
        self.r = r
        self.color = color
        self.species = species
        self.vision_distance = vision_distance
        self.init_body(canvas)
        self.generation = 0
        self.fitness = 0 # num foods eatin this lifetime
        self.health = 1
        self.constant_decay = 0.001
        self.fitness_history = [] 
        self.init_brain()
        
    def init_body(self, canvas):
        # create receptive field lines
        self.horizontal_eye = canvas.create_line(self.x-self.vision_distance, self.y, self.x+self.vision_distance, self.y)
        self.vertical_eye = canvas.create_line(self.x, self.y-self.vision_distance, self.x, self.y+self.vision_distance)
        # create body
        self.circle = canvas.create_oval(self.x-self.r, self.y-self.r, self.x+self.r, self.y+self.r, fill=self.color)
        self.stat_text = canvas.create_text(self.x, self.y-2*self.r, text='text')

    def init_brain(self):
        self.fov = np.asarray([0,0,0,0,0,0,0,0])
        initializer = tf.random_normal_initializer(mean=0, stddev=1, seed=None)
        # initializer2 = tf.random_normal_initializer(mean=0, stddev=2, seed=None)
        zeros_initializer = tf.zeros_initializer()
        self.W1 = tf.Variable(initializer(shape=[len(self.fov)+16, 16], dtype=tf.float32))
        self.W2 = tf.Variable(initializer(shape=[16, 2], dtype=tf.float32))
        self.R1 = tf.Variable(zeros_initializer(shape=[1,16]))
        self.b1 = tf.Variable(initializer(shape=[16], dtype=tf.float32))
        self.b2 = tf.Variable(initializer(shape=[2], dtype=tf.float32))
    
    def clear_brain_memory(self):
        zeros_initializer = tf.zeros_initializer()
        self.R1 = tf.Variable(zeros_initializer(shape=[1,16])) 

    def set_nn_weights(self, weights):
        self.W1, self.W2, self.b1, self.b2 = weights
    
    def get_nn_weights(self):
        return (self.W1, self.W2, self.b1, self.b2)
    
    def mutate_weights(self, stddev):
        initializer = tf.random_normal_initializer(mean=0, stddev=stddev, seed=None)
        W1_mut = tf.Variable(initializer(shape=self.W1.shape, dtype=tf.float32))
        W2_mut = tf.Variable(initializer(shape=self.W2.shape, dtype=tf.float32))
        R1_mut = tf.Variable(initializer(shape=self.R1.shape, dtype=tf.float32))
        b1_mut = tf.Variable(initializer(shape=self.b1.shape, dtype=tf.float32))
        b2_mut = tf.Variable(initializer(shape=self.b2.shape, dtype=tf.float32))
        self.W1 = self.W1 + W1_mut
        self.W2 = self.W2 + W2_mut
        self.R1 = self.R1 + R1_mut
        self.b1 = self.b1 + b1_mut
        self.b2 = self.b2 + b2_mut


    def multilayer_perceptron(self, x):
        x = np.expand_dims(x.astype('float32'), axis=0)
        # print(f"recurrent feedback: {np.sum(self.R1)}")
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
        '''
        Returns false if cell dies, true if cell continues
        '''
        self.fitness += 1
        self.health -= self.constant_decay
        if self.health <= 0:
            self.self_destruct(canvas)
            return False
        
        # calculate call movement from nn output
        move_vector = self.calc_movement()
        if abs(move_vector[0]) <= 1:
            x_vel = 0
        elif (move_vector[0] < -1):
            x_vel = -1 
        else:
            x_vel = 1

        if abs(move_vector[1]) <= 1:
            y_vel = 0
        elif move_vector[1] < -1:
            y_vel = -1
        else:
            y_vel = 1

        # make sure cell not moving outside canvas
        future_x = self.x + x_vel
        if (future_x < self.r or future_x > w - self.r):
            x_vel = 0
            self.self_destruct(canvas)
            return False
        else:
            self.x = future_x
        future_y = self.y + y_vel
        if (future_y < self.r or future_y > h - self.r):
            y_vel = 0
            self.self_destruct(canvas)
            return False
        else:
            self.y = future_y

        # move eyes
        canvas.coords(self.horizontal_eye, self.x-self.vision_distance, self.y, self.x+self.vision_distance, self.y)
        canvas.coords(self.vertical_eye, self.x, self.y-self.vision_distance, self.x, self.y+self.vision_distance)
        # move cell body
        canvas.move(self.circle, x_vel, y_vel)
        # move cell stat text
        canvas.move(self.stat_text, x_vel, y_vel)

        # TODO redraw 
        canvas.itemconfigure(self.stat_text, text=f"{round(self.fitness, 3)} {round(self.health, 3)}")
        
        return True

    def eat(self, food):
        self.health += 1  

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
        canvas.delete(self.stat_text)

    # TODO save cell neural network weights to file
    # TODO save cell death state, fitness, nn weights

    def end_epoch(self, canvas, new_x, new_y):
        '''
        Returns cell average fitness and number of generations it has been alive
        '''
        # append cell's current fitness to their history of fitness
        # print(f"Cell fitness: {self.fitness}")
        self.fitness_history.append(self.fitness)
        self.fitness = 0
        # calculate each cell average fitness
        avg_fitness = self.avg_fitness()
        self.self_destruct(canvas)
        zeros_initializer = tf.zeros_initializer()
        self.R1 = tf.Variable(zeros_initializer(shape=[1,16]))
        self.x = new_x
        self.y = new_y
        self.init_body(canvas)
        return avg_fitness, len(self.fitness_history)

    def reset(self):
        # re-randomize their neural weights
        # reset their fitness history to blank []
        # set their current fitness to 0
        self.fitness = 0
        self.fitness_history = []
        self.init_brain()
        
    def avg_fitness(self):
        if len(self.fitness_history) < 1:
            return self.fitness
        return sum(self.fitness_history)/len(self.fitness_history)