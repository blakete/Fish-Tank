from tkinter import *
import numpy as np
import time
import math
from random import randint
from cell import Cell
from food import Food

frame_rate = 5
screen_size = 500
num_foods = 10
num_cells = 10
passthrough_rate = 0.5

cell_vision_distance = 75
timesteps = 0
epoch_timesteps = 700
epoch = 0

window = Tk()
window.title("Fish Tank")
window.geometry(f"{screen_size}x{screen_size}")
window.resizable(False, False)
canvas = Canvas(window, width=screen_size, height=screen_size)
canvas.pack()

def generate_cell_coordinate(r):
    '''
    Generates a coordinate pair within the window
    '''
    return randint(0+r, screen_size-r), randint(0+r, screen_size-r)

cells = []
foods = []
time_steps = 0



def is_collision(a, b):
    dist = math.sqrt(math.pow(a.x - b.x, 2) + math.pow(a.y - b.y, 2))
    return dist < (a.r + b.r)
def distance_point_to_point(p1, p2):
    return math.sqrt(math.pow((p2[0]-p1[0]), 2) + math.pow((p2[1]-p1[1]), 2))

def distance_point_to_line(food_coords, line_coords):
    # by = ax + c, b = 1
    b = 1
    if line_coords[2]-line_coords[0] == 0:
        return abs(food_coords[0] - line_coords[0])
    else:
        a = (line_coords[3]-line_coords[1])/(line_coords[2]-line_coords[0])
    c = line_coords[1] - a*line_coords[0]
    # print(f"a: {a}\tb: {b}\tc: {c}\tx0: {food_coords[0]}\ty0: {food_coords[1]}")
    numerator = abs(a*food_coords[0] + b*food_coords[1]-c)
    denominator = math.sqrt(math.pow(a, 2) + math.pow(b,2))
    # print(f"{numerator} / {denominator}")
    return numerator/denominator

def calculate_cell_food_vision(cell_coords, food_coords):
    receptive_field = [0,0] # n,s,e,w
    diff_x = food_coords[0] - cell_coords[0]
    diff_y = food_coords[1] - cell_coords[1]
    n = diff_y < 0
    w = diff_x < 0

def repopulate():
    global cells
    # keep only passthrough rate number of cells for next generation
    cell_fitnesses = np.asarray([cell.consumed for cell in cells])
    print(f"Avg cell fitness: {sum(cell_fitnesses)/len(cell_fitnesses)}")
    sorted_idxs = np.argsort(cell_fitnesses)
    print(f"Cell fitness distribution: {cell_fitnesses[sorted_idxs]}")
    # keep only passthrough rate
    sorted_idxs = sorted_idxs[-int(num_cells*passthrough_rate):]
    # create new cells for next gen that inherit the NN weights
    new_cells = []
    for i in sorted_idxs:
        x,y = generate_cell_coordinate(10)
        new_cell = Cell(canvas, x, y)
        new_cell.set_nn_weights(cells[i].get_nn_weights())
        new_cells.append(new_cell)
    for i in range(0, num_cells-int(num_cells*passthrough_rate)):
        x,y = generate_cell_coordinate(10)
        new_cells.append(Cell(canvas, x, y))
    cells = new_cells

    # regen foods
    global foods
    foods = []
    for i in range(0, num_foods):
        x,y = generate_cell_coordinate(10)
        foods.append(Food(canvas, x, y))

def move():
    global cells
    global foods
    global epoch
    global time_steps
    global frame_rate
    time_steps += 1
    if time_steps == 1 or time_steps % epoch_timesteps == 0:
        # if first generation, create the cells
        print(f"---------------- Epoch {epoch} ------------------")
        if epoch == 0:
            # create cells
            for i in range(0, num_cells):
                # x,y = generate_cell_coordinate(10)
                x, y = int(screen_size/2), int(screen_size/2)
                cells.append(Cell(canvas, x, y))
            # create foods
            for i in range(0, num_foods):
                x,y = generate_cell_coordinate(10)
                foods.append(Food(canvas, x, y))
        else:
            # recreate eaten foods
            for i in range(num_cells-len(foods)):
                x,y = generate_cell_coordinate(10)
                foods.append(Food(canvas, x, y))
            # reset foods
            for food in foods:
                new_x, new_y = generate_cell_coordinate(10)
                food.end_epoch(canvas, new_x, new_y)
            # reset cells for next generation
            avg_fitnesses = []
            lifetimes = []
            for cell in cells:
                # new_x, new_y = generate_cell_coordinate(10)
                new_x, new_y = int(screen_size/2), int(screen_size/2)
                avg_fitness, lifelength = cell.end_epoch(canvas, new_x, new_y)
                avg_fitnesses.append(avg_fitness)
                lifetimes.append(lifelength)
            # arg sort cells by their avg fitness
            sorted_idxs = np.argsort(np.asarray(avg_fitnesses))
            print("---------Cell Report - Pre Reset ---------")
            for idx in sorted_idxs:
                print(f"Cell {idx}, life length: {lifetimes[idx]}, fitness: {avg_fitnesses[idx]}, fitness_history: {cells[idx].fitness_history}")
            
            for reset_idx in sorted_idxs[:int(num_cells*passthrough_rate)]:
                cells[reset_idx].reset()
            
            print("---------Cell Report - Post Reset ---------")
            for idx in sorted_idxs:
                print(f"Cell {idx}, life length: {len(cells[idx].fitness_history)},  fitness: {cells[idx].avg_fitness()}, fitness_history: {cells[idx].fitness_history}")
        epoch += 1

    # check for collisions between cells and food
    for cell in cells:
        purge_indexes = []
        for i in range(len(foods)): 
            if is_collision(cell, foods[i]):
                cell.eat(foods[i])
                purge_indexes.append(i)    
        # delete consumed foods
        for i in sorted(purge_indexes, reverse=True):
            foods[i].self_destruct(canvas)
            del foods[i]
    
    # calculate cell fov vector
    # [N, S, E, W]
    for cell in cells:
        fov = [0,0,0,0]
        fov_walls = [0,0,0,0]
        cell_point = cell.get_coords()
        # calculate if cell sees walls
        if cell_point[0] < cell_vision_distance:
            fov_walls[3] = 1
        elif cell_point[0] > screen_size - cell_vision_distance:
            fov_walls[2] = 1
        if cell_point[1] < cell_vision_distance:
            fov_walls[0] = 1
        elif cell_point[1] > screen_size - cell_vision_distance:
            fov_walls[1] = 1
        # calculate if cell sees foods
        for food in foods:
            food_point = food.get_coords()
            hypotenuse = distance_point_to_point(cell_point, food_point)
            
            h_eye_line = cell.get_eye_coords(canvas, "h")
            h_eye_dist = distance_point_to_line(food_point, h_eye_line)
            h_eye_prox = math.sqrt(math.pow(hypotenuse, 2) - math.pow(h_eye_dist, 2))

            v_eye_line = cell.get_eye_coords(canvas, "v")
            v_eye_dist = distance_point_to_line(food_point, v_eye_line)
            v_eye_prox = math.sqrt(math.pow(hypotenuse, 2) - math.pow(v_eye_dist, 2))

            if (h_eye_dist < 10 and h_eye_prox < cell_vision_distance):
                if cell_point[0] < food_point[0]:
                    fov[2] += 1
                else:
                    fov[3] += 1
            if (v_eye_dist < 10 and v_eye_prox < cell_vision_distance):
                if cell_point[1] > food_point[1]:
                    fov[0] += 1
                else:
                    fov[1] += 1
        fov.extend(fov_walls)
        cell.fov = np.asarray(fov)
        # print(f"cell fov: {cell.fov}")

    # move cells based on fov
    for cell in cells:
        cell.advance(canvas, screen_size, screen_size)

    # delete stale foods
    purge_indexes = []
    for i in range(len(foods)):
        if not foods[i].advance(canvas):
            purge_indexes.append(i)
    for i in sorted(purge_indexes, reverse=True):
        foods[i].self_destruct(canvas)
        del foods[i]

    window.after(frame_rate, move)

print("Starting movement...")
move()
window.mainloop()
