import numpy as np

from agent import Agent
from helper_classes import Pair, Ocean

class Environment:
    """
    Environment class: environment initialization, emotion-to-path mapping, (local collision avoidance, crowd simulation)
    """
    def __init__(self):
        self.agents = []

        # Example initialization
        for i in range(10):
            a = Agent(i)
            a.traits = Ocean.sample(Pair(0.5, 0.1), Pair(0.5, 0.1), Pair(0.5, 0.1), Pair(0.5, 0.1), Pair(0.5, 0.1))
            a.position = Pair(np.random.randint(0, 100), np.random.randint(0, 100))
            a.velocity = Pair(np.random.randint(-5, 5), np.random.randint(-5, 5))
            self.agents.append(a)
            
        self.relationship_matrix = np.zeros((len(self.agents), len(self.agents)))
        for i, agent0 in enumerate(self.agents):
            for j, agent1 in enumerate(self.agents):
                self.relationship_matrix[i, j] = agent0.relationship(agent1)
                self.relationship_matrix[j, i] = self.relationship_matrix[i, j]
            
    def collective_density(self, agent0: Agent):
        """Calculate the collective density of the agent
            that is number of relationships with other agents ("degree of the node")
        """
        
        ro = np.sum(self.relationship_matrix[agent0.id,:])
        return ro