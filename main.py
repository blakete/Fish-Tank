import json
from tkinter import *
import numpy as np
import time
import math
from random import randint
import random
from cell import Cell
from food import Food


class SimulationConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def generate_blob_coordinate(config, r):
    '''
    Generates a coordinate pair within the window
    '''
    return randint(0+r, config.screen_size-r), randint(0+r, config.screen_size-r)

f = open("config.json", "r")
data = json.load(f)
config = SimulationConfig(**data)

time_steps = 0
generation = 0
cells = []
best_cells = {}
best_fitness = {}
foods = []

window = Tk()
window.title("Fish Tank")
window.geometry(f"{config.screen_size}x{config.screen_size}")
window.resizable(False, False)
canvas = Canvas(window, width=config.screen_size, height=config.screen_size)
canvas.pack()

# init foods
if config.food_init_pattern == "corners":
    foods.append(Food(canvas, config.cell_vision_distance, config.cell_vision_distance))
    foods.append(Food(canvas, config.screen_size-config.cell_vision_distance, config.cell_vision_distance))
    foods.append(Food(canvas, config.cell_vision_distance, config.screen_size-config.cell_vision_distance))
    foods.append(Food(canvas, config.screen_size-config.cell_vision_distance, config.screen_size-config.cell_vision_distance))
elif config.food_init_pattern == "random":
    for i in range(config.n_foods):
        x, y = generate_blob_coordinate(config, 10)
        foods.append(Food(canvas, x, y))

# init cells 
if config.cell_init_pattern == "center":
    for i in range(0, config.n_cells):
        x,y = int(config.screen_size/2), int(config.screen_size/2)
        cells.append(Cell(canvas, x, y, id=i))
elif config.cell_init_pattern == "random":
    for i in range(0, config.n_cells):
        x, y = generate_blob_coordinate(config, 10)
        cells.append(Cell(canvas, x, y, id=i))

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

def evolve_species(eol_cell, best_cells):
    ''' 
    Take end of life cell and compare with best cell fitness
    if > fitness, replace that cell with the new one
    '''
    if eol_cell.id not in best_cells:
        best_cells[eol_cell.id] = eol_cell
    elif best_cells[eol_cell.id].fitness < eol_cell.fitness:
        best_cells[eol_cell.id] = eol_cell
    return best_cells

def move():
    global config
    global cells
    global foods
    global epoch
    global time_steps
    global frame_rate
    global best_cells
    global best_fitness
    global generation

    # if no cells left, time to restart with a new generation
    if len(cells) == 0:
        generation += 1
        if generation >= config.n_generations:
            print(f"completed {generation} generations")
            window.destroy()
            return
        print(f"GENERATION {generation}")
        for key in best_cells.keys():
            print(f"cell {key} best: {best_cells[key].fitness}")
        if config.cell_inheritance == "random":
            for i in range(config.n_cells):
                if config.cell_init_pattern == "center":
                    x, y = int(config.screen_size/2), int(config.screen_size/2)
                elif config.cell_init_pattern == "random":
                    x, y = generate_blob_coordinate(config, 10)
                cells.append(Cell(canvas, x, y, id=i))
        elif config.cell_inheritance == "simple":
            for species in best_cells.keys():
                print(f"Species {species} highest fitness: {best_cells[species].fitness}")
                # create new cell with same: id, color, neural netw
                x, y = int(config.screen_size/2), int(config.screen_size/2)
                parentCell = best_cells[species]
                childCell = Cell(canvas, x, y, parent_cell=parentCell)
                childCell.generation = parentCell.generation + 1
                # initialize with parent network
                childCell.set_nn_weights(parentCell.get_nn_weights())
                # add slight mutation to child weights
                mutation_stddev = random.uniform(0.01, 1.0)
                childCell.mutate_weights(mutation_stddev)
                print(f"current child mutation stddev: {mutation_stddev}")
                # TODO make sure mutation does not modify parents weights because pass by reference
                childCell.clear_brain_memory() # delete recurrent memory from parent
                cells.append(childCell)
        
        # clear and regen foods
        for food in foods:
            food.self_destruct(canvas)
            foods.remove(food)

        # init foods
        if config.food_init_pattern == "corners":
            foods.append(Food(canvas, config.cell_vision_distance, config.cell_vision_distance))
            foods.append(Food(canvas, config.screen_size-config.cell_vision_distance, config.cell_vision_distance))
            foods.append(Food(canvas, config.cell_vision_distance, config.screen_size-config.cell_vision_distance))
            foods.append(Food(canvas, config.screen_size-config.cell_vision_distance, config.screen_size-config.cell_vision_distance))
        elif config.food_init_pattern == "random":
            for i in range(config.n_foods):
                x, y = generate_blob_coordinate(config, 10)
                foods.append(Food(canvas, x, y))
        
    # check for collisions between cells and food
    # TODO check cell fitness and spawn new cell if at 
    # clone_indices = [] # cells to clone because they ate a food

    for j, cell in enumerate(cells):
        purge_indexes = []
        for i in range(len(foods)): 
            if is_collision(cell, foods[i]):
                cell.eat(foods[i])
                # clone_indices.append(j)
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
        if cell_point[0] < config.cell_vision_distance:
            fov_walls[3] = 1
        elif cell_point[0] > config.screen_size - config.cell_vision_distance:
            fov_walls[2] = 1
        if cell_point[1] < config.cell_vision_distance:
            fov_walls[0] = 1
        elif cell_point[1] > config.screen_size - config.cell_vision_distance:
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

            if (h_eye_dist < 10 and h_eye_prox < config.cell_vision_distance):
                if cell_point[0] < food_point[0]:
                    fov[2] += 1
                else:
                    fov[3] += 1
            if (v_eye_dist < 10 and v_eye_prox < config.cell_vision_distance):
                if cell_point[1] > food_point[1]:
                    fov[0] += 1
                else:
                    fov[1] += 1
        fov.extend(fov_walls)
        cell.fov = np.asarray(fov)
        # print(f"cell fov: {cell.fov}")

    # move cells based on fov, only keep cells with fitness above 0 to continue
    for cell in cells:
        if not cell.advance(canvas, config.screen_size, config.screen_size):
            # print(f"Species {cell.id} died, fitness {cell.fitness}")
            best_cells = evolve_species(cell, best_cells)
            cells.remove(cell)

            # record best cell fitnesses 
            if cell.id not in best_fitness:
                best_fitness[cell.id] = [best_cells[cell.id].fitness]
            else:
                best_fitness[cell.id].append(best_cells[cell.id].fitness)

    time_steps += 1
    window.after(config.frame_delta, move)

print("Starting movement...")
move()
window.mainloop()
