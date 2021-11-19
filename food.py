from random import randint 

class Food:
    def __init__(self, canvas, x, y, r=10, color="green", points=600, decay=1):
        self.x = x
        self.y = y
        self.r = r
        self.color = color
        self.points = points
        self.decay = decay
        self.circle = canvas.create_oval(x-r, y-r, x+r, y+r, fill=color)


    def advance(self, canvas):
        '''
        Advances food by decay factor and returns true if still alive.
        '''
        self.points -= self.decay
        if self.points <= 0:
            return False
        return True

    def self_destruct(self, canvas):
        canvas.delete(self.circle)
    
    def get_coords(self):
        return self.x, self.y