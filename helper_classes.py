import numpy as np

class Pair:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Pair(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Pair(self.x - other.x, self.y - other.y)
    
    # dot product
    def __mul__(self, other):
        return self.x * other + self.y * other
    
    def norm(self):
        return np.sqrt(self.x ** 2 + self.y ** 2)

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
    def sample(openness_params: Pair, conscientiousness_params: Pair, extroversion_params: Pair, agreeableness_params: Pair, neuroticism_params: Pair):
        """Sample OCEAN traits example, given the distribution for each dimension (as in the population)

        Args:
            *_params (Pair): pair of mu and sigma for each OCEAN trait

        Returns:
            Ocean: sampled OCEAN traits example
        """
        ocean_params = []
        for mu, sig in [openness_params, conscientiousness_params, extroversion_params, agreeableness_params, neuroticism_params]:
            sig2 = sig ** 2
            assert 0 <= mu and mu <= 1
            assert -0.1 <= sig and sig <= 0.1
            s = np.random.normal(mu, sig2, 1)[0]
            ocean_params.append(s)
        return Ocean(*ocean_params)   
