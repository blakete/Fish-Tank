# imports every file form tkinter and tkinter.ttk
from tkinter import *
from tkinter.ttk import *
 
class GFG:
    def __init__(self, master = None):
        self.master = master
         
        # to take care movement in x direction
        self.x = 1
        # to take care movement in y direction
        self.y = 0
 
        # canvas object to create shape
        self.canvas = Canvas(master)
        # creating rectangle
        self.rectangle = self.canvas.create_rectangle(
                         5, 5, 25, 25, fill = "black")
        self.canvas.pack()
 
        # calling class's movement method to
        # move the rectangle
        self.movement()
     
    def movement(self):
        # This is where the move() method is called
        # This moves the rectangle to x, y coordinates
        self.canvas.move(self.rectangle, self.x, self.y)
 
        self.canvas.after(100, self.movement)

 
if __name__ == "__main__":
 
    # object of class Tk, responsible for creating
    # a tkinter toplevel window
    master = Tk()
    gfg = GFG(master)
     
    # Infinite loop breaks only by interrupt
    mainloop()