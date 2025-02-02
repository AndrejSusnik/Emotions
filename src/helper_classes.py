import numpy as np

class Pair:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def __str__(self):
        return f"Pair({self.x}, {self.y})"

    def get(self):
        return self.x, self.y
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
     
    def __add__(self, other):
        return Pair(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Pair(self.x - other.x, self.y - other.y)
    
    # dot product
    def __mul__(self, other):
        return self.x * other.x + self.y * other.y
    
    def __repr__(self):
        return f"({round(self.x,1)}, {round(self.y,1)})"
    
    def __ge__(self, other):
        if self.x > other.x:
            return True
        if self.x < other.x:
            return False
        return self.y >= other.y
    
    def __lt__(self, other):
        return not self >= other
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def norm(self):
        return np.sqrt(self.x ** 2 + self.y ** 2)
    
    def round(self):
        return Pair(int(round(self.x)), int(round(self.y)))

    def scale(self, scalar):
        return Pair(self.x * scalar, self.y * scalar)
    
    def copy(self):
        return Pair(self.x, self.y)

class Line:
    def __init__(self, start: Pair, end: Pair):
        self.start = start
        self.end = end
        self.points_buf = None

    def __str__(self):
        return f"Line({self.start}, {self.end})"

    def norm(self, p: Pair):
        self.start.x = self.start.x / p.x
        self.start.y = self.start.y / p.y

        self.end.x = self.end.x / p.x
        self.end.y = self.end.y / p.y

        return self

    def scale(self, p: Pair):
        self.start.x = self.start.x * p.x
        self.start.y = self.start.y * p.y

        self.end.x = self.end.x * p.x
        self.end.y = self.end.y * p.y

        return self

    def points(self):
        if self.points_buf:
            return self.points_buf

        if self.start.y == self.end.y:
            self.points_buf = [Pair(x, self.start.y).round() for x in range(int(round(self.start.x)), int(round(self.end.x)) +1)]
        if self.start.x == self.end.x:
            self.points_buf =[Pair(self.start.x, y).round() for y in range(int(round(self.start.y)), int(round(self.end.y)) +1)]

        return self.points_buf
        
    
    def center(self):
        return Pair((self.start.x + self.end.x) / 2, (self.start.y + self.end.y) / 2)
    
    def intersection(self, other: 'Line'):
        x1, y1 = self.start.get()
        x2, y2 = self.end.get()
        x3, y3 = other.start.get()
        x4, y4 = other.end.get()
        
        # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
        d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if d == 0:
            return None
        x = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
        y = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
        inti = Pair(x / d, y / d)
        
        if inti.x < min(x1, x2) or inti.x > max(x1, x2) or inti.y < min(y1, y2) or inti.y > max(y1, y2):
            return None
        
        return inti
    
        


class Rect:
    def __init__(self, a: Line, b: Line, c: Line, d: Line):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
    
    def contains(self, p: Pair):
        return self.a.start.x <= p.x and p.x <= self.b.start.x and self.a.start.y <= p.y and p.y <= self.c.start.y

class OceanDistribution:
    def __init__(self, openness: Pair, conscientiousness: Pair, extroversion: Pair, agreeableness: Pair, neuroticism: Pair):
        self.openness_dist = openness
        self.conscientiousness_dist = conscientiousness
        self.extroversion_dist = extroversion
        self.agreeableness_dist = agreeableness
        self.neuroticism_dist = neuroticism

    def getDistArray(self):
        return [self.openness_dist, self.conscientiousness_dist, self.extroversion_dist, self.agreeableness_dist, self.neuroticism_dist]

class Ocean:
    """
    Personality trait model, emotion descriptor (orthogonal 5 dimensions)
    """
    def __init__(self, openness: float, conscientiousness: float, extroversion: float, agreeableness: float, neuroticism: float):
        self.openness = openness
        self.conscientiousness = conscientiousness
        self.extroversion = extroversion
        self.agreeableness = agreeableness
        self.neuroticism = neuroticism
        
    @staticmethod    
    def sample(oceanDistribution: OceanDistribution):
        """Sample OCEAN traits example, given the distribution for each dimension (as in the population)

        Args:
            *_params (Pair): pair of mu and sigma for each OCEAN trait

        Returns:
            Ocean: sampled OCEAN traits example
        """
        ocean_params = []
        for dist in oceanDistribution.getDistArray():
            mu, sig = dist.get()
            sig2 = sig ** 2
            assert 0 <= mu and mu <= 1
            assert -0.1 <= sig and sig <= 0.1
            s = np.random.normal(mu, sig2, 1)[0]
            ocean_params.append(s)
        return Ocean(*ocean_params)   

    @staticmethod
    def empty():
        return Ocean(0, 0, 0, 0, 0)
    
    def __str__(self):
        #round to k places
        k = 3
        return f"OCEAN({round(self.openness, k)}, {round(self.conscientiousness, k)}, {round(self.extroversion, k)}, {round(self.agreeableness, k)}, {round(self.neuroticism, k)})"
        
if __name__ == "__main__":
    p1 = Pair(0, 0)
    p2 = Pair(1, 1)
    p3 = Pair(0, 0)
    p4 = Pair(0, 1)
    l1 = Line(p1, p2)
    l2 = Line(p3, p4)
    print(l1.intersection(l2)) # Pair(0.5, 0.5)
    print(l2.intersection(l1)) # Pair(0.5, 0.5)