from helper_classes import Pair, Line

class Exit(Line):
    """Exit is a line with id, it represents an exit from the environment"""
    def __init__(self, start : Pair, end : Pair, id : int):
        super().__init__(start, end)
        self.points_arr = None
        self.id = id

    def points(self):
        if self.points_arr:
            return self.points_arr
        else:
            if self.start.y == self.end.y:
                self.points_arr = [Pair(x, self.start.y).round() for x in range(int(round(self.start.x)), int(round(self.end.x)) +1)]
            if self.start.x == self.end.x:
                self.points_arr = [Pair(self.start.x, y).round() for y in range(int(round(self.start.y)), int(round(self.end.y)) +1)]
        
        return self.points_arr
    
    def __eq__(self, other):
        return self.id == other.id
        
    
class ExitEx():
    """ExitEx is an extended exit, it has an id and a list of points"""
    def __init__(self, id : int, points : list = []):
        self.id = id
        self.points = points
    
    def add_point(self, point : Pair):
        self.points.append(point)

    def is_empty(self):
        return len(self.points) == 0

    def __eq__(self, other):
        return self.id == other.id

    def __str__(self):
        return f"ExitEx: {self.id} {self.points}"