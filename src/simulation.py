import numpy as np

from agent import Agent
from environment import Environment
from helper_classes import Pair, Ocean, OceanDistribution

class SimulationParams:
    def __init__(self, num_agents: int, oceanDistribution: OceanDistribution, environment: Environment, simulation_time_in_seconds: int = 100, dt: float = 0.1):
        self.num_agents = num_agents
        self.oceanDistribution = oceanDistribution
        self.environment = environment
        self.simulation_time_in_seconds = simulation_time_in_seconds
        self.dt = dt

class Simulation:
    """
    Environment class: environment initialization, emotion-to-path mapping, (local collision avoidance, crowd simulation)
    """
    def __init__(self, params: SimulationParams):
        self.agents = []
        self.environment = params.environment

        valid_positions = self.environment.get_valid_positions()

        # Example initialization
        for i in range(params.num_agents):
            a = Agent(i)
            a.traits = Ocean.sample(params.oceanDistribution)
            while True:
                a.position = Pair(np.random.randint(self.environment.size[0]), np.random.randint(self.environment.size[1]))
                if a.position.get() in valid_positions:
                    valid_positions.remove(a.position.get())
                    break

            a.velocity = Pair(np.random.randint(-5, 5), np.random.randint(-5, 5))
            self.agents.append(a)
            
        self.relationship_matrix = np.zeros((len(self.agents), len(self.agents)))
        for i, agent0 in enumerate(self.agents):
            for j, agent1 in enumerate(self.agents):
                self.relationship_matrix[i, j] = agent0.relationship(agent1)
                self.relationship_matrix[j, i] = self.relationship_matrix[i, j]

    #Ali razširimo to funkcijo ali pa dodamo novo za clustering
            
    def collective_density(self, agent0: Agent):
        """Calculate the collective density of the agent
            that is number of relationships with other agents ("degree of the node")
        """
        
        ro = np.sum(self.relationship_matrix[agent0.id,:,2])
        return ro            

    def run(self):
        self.environment.plot(self.agents)
