from random import randint 

class Food:
    def __init__(self, canvas, x, y, r=10, color="green", points=1, decay=0):
        self.x = x
        self.y = y
        self.r = r
        self.color = color
        self.points = points
        self.decay = decay
        self.init_body(canvas)

    def init_body(self, canvas):
        self.circle = canvas.create_oval(self.x-self.r, self.y-self.r, self.x+self.r, self.y+self.r, fill=self.color)

    def self_destruct(self, canvas):
        canvas.delete(self.circle)
    
    def get_coords(self):
        return self.x, self.y
    
    def end_epoch(self, canvas, new_x, new_y):
        self.self_destruct(canvas)
        self.x = new_x
        self.y = new_y
        self.init_body(canvas)