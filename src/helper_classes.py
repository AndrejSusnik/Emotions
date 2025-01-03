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

class Line:
    def __init__(self, start: Pair, end: Pair):
        self.start = start
        self.end = end

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
    
    def center(self):
        return Pair((self.start.x + self.end.x) / 2, (self.start.y + self.end.y) / 2)


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