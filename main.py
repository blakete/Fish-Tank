from tkinter import *
import numpy as np
import time
import math
from random import randint
from cell import Cell
from food import Food

frame_rate = 1
screen_size = 800
num_foods = 30
num_cells = 10
cell_vision_distance = 75
timesteps = 0
cell_fitness_reproduction_level = 2 # fitness must be this much for cell to clone itself

window = Tk()
window.title("Fish Tank")
window.geometry(f"{screen_size}x{screen_size}")
window.resizable(False, False)
canvas = Canvas(window, width=screen_size, height=screen_size)
canvas.pack()

cells = []
foods = []
time_steps = 0

def generate_cell_coordinate(r):
    '''
    Generates a coordinate pair within the window
    '''
    return randint(0+r, screen_size-r), randint(0+r, screen_size-r)

def generate_hex_color():
    r, g, b = hex(randint(0,255)), hex(randint(0,255)), hex(randint(0,255))
    r = r[-2:] if len(r) == 4 else f"0{r[-1:]}"
    g = g[-2:] if len(g) == 4 else f"0{g[-1:]}"
    b = b[-2:] if len(b) == 4 else f"0{b[-1:]}"
    return f"#{r}{g}{b}"

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

def move():
    global cells
    global foods
    global epoch
    global time_steps
    global frame_rate

    if time_steps % 50 == 0:
        print(f"Total cells: {len(cells)}")

    # check for collisions between cells and food
    # TODO check cell fitness and spawn new cell if at 
    clone_indices = [] # cells to clone because they ate a food
    for j, cell in enumerate(cells):
        purge_indexes = []
        for i in range(len(foods)): 
            if is_collision(cell, foods[i]):
                # cell.eat(foods[i])
                clone_indices.append(j)
                purge_indexes.append(i)    
        # delete consumed foods
        for i in sorted(purge_indexes, reverse=True):
            foods[i].self_destruct(canvas)
            del foods[i]
    
    # TODO create cell clones
    for i in clone_indices:
        # x, y = int(screen_size/2), int(screen_size/2)
        x, y = generate_cell_coordinate(100)
        parentCell = cells[i]
        childCell = Cell(canvas, x, y, color=parentCell.color)
        childCell.generation = parentCell.generation + 1
        # initialize with parent 
        childCell.set_nn_weights(parentCell.get_nn_weights())
        # add slight mutation to child weights
        childCell.mutate_weights(0.25)
        childCell.clear_brain_memory() # delete recurrent memory from parent
        cells.append(childCell)

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

    # move cells based on fov, only keep cells with fitness above 0 to continue
    cells = [x for x in cells if x.advance(canvas, screen_size, screen_size)]
    
    # create cells if not at minimum cells
    if len(cells) < num_cells:
        for i in range(0, num_cells-len(cells)):
            x, y = int(screen_size/2), int(screen_size/2)
            cells.append(Cell(canvas, x, y, color=generate_hex_color()))
    
    # create foods if not at minimum foods
    if len(foods) < num_foods:
        for i in range(0, num_foods-len(foods)):
            x,y = generate_cell_coordinate(10)
            foods.append(Food(canvas, x, y))

    time_steps += 1
    window.after(frame_rate, move)

print("Starting movement...")
move()
window.mainloop()
