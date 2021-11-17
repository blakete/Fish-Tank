from tkinter import *
import numpy as np
import time
import math
from random import randint
from cell import Cell
from food import Food

screen_size = 700
num_foods = 10
num_cells = 10
passthrough_rate = 0.5

timesteps = 0

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

# create cells
cells = []
for i in range(0, num_cells):
    x,y = generate_cell_coordinate(10)
    cells.append(Cell(canvas, x, y))

# create foods 
foods = []
# x,y = 90,50
# foods.append(Food(canvas, x, y))
# x,y = 90,60
# foods.append(Food(canvas, x, y))
for i in range(0, num_foods):
    x,y = generate_cell_coordinate(10)
    foods.append(Food(canvas, x, y))

urmom = []



def is_collision(a, b):
    dist = math.sqrt(math.pow(a.x - b.x, 2) + math.pow(a.y - b.y, 2))
    return dist < (a.r + b.r)


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
    urmom.append(1)
    if len(urmom) % 200 == 0:
        print(f"Timestep: {len(urmom)}")
    if len(urmom) % 1000 == 0:
        print("============================= Repopulating =============================")
        # purge previous cells from canvas
        for cell in cells:
            cell.self_destruct(canvas)
        for food in foods:
            food.self_destruct(canvas)
        repopulate()
        print(f"Timestep: {len(urmom)}")
        print(f"Total population: {len(cells)}")

    # print("---Cell Status Report---")
    # print("id\tconsumed\tfov")
    # for i, cell in enumerate(cells):
    #     print(f"{i}\t{cell.consumed}\t{cell.fov}")

    # check for collisions between cells and food
    # TODO randomize the order of cell updates for easy race condition solution
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

    # add new foods if needed
    # TODO spawn foods not in collision with cells
    foods_to_create = num_foods - len(foods)
    if foods_to_create > 0:
        x,y = generate_cell_coordinate(10)
        foods.append(Food(canvas, x, y))
    

    # TODO calculate cell receptive field vector
    # [N, S, E, W]
    for cell in cells:
        fov = [0,0,0,0]
        for food in foods:
            food_point = food.get_coords()
            cell_point = cell.get_coords()
            h_eye_line = cell.get_eye_coords(canvas,"h")
            h_eye_dist = distance_point_to_line(food_point, h_eye_line)
            v_eye_line = cell.get_eye_coords(canvas,"v")
            v_eye_dist = distance_point_to_line(food_point, v_eye_line)
            if (h_eye_dist < 10):
                if cell_point[0] < food_point[0]:
                    fov[2] += 1
                else:
                    fov[3] += 1
            if (v_eye_dist < 10):
                if cell_point[1] > food_point[1]:
                    fov[0] += 1
                else:
                    fov[1] += 1
        cell.fov = np.asarray(fov)

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

    window.after(5, move)


move()
window.mainloop()
