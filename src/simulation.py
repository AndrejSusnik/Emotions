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
            
        # self.relationship_matrix = np.zeros((len(self.agents), len(self.agents)))
        # for i, agent0 in enumerate(self.agents):
        #     for j, agent1 in enumerate(self.agents):
        #         self.relationship_matrix[i, j] = agent0.relationship(agent1)
        #         self.relationship_matrix[j, i] = self.relationship_matrix[i, j]
                
        

    #Ali razširimo to funkcijo ali pa dodamo novo za clustering
            
    def collective_density(self, agent0: Agent) -> int:
        """Calculate the collective density of the agent
            that is number of relationships with other agents ("degree of the node")
        """
        
        # ro = np.sum(self.relationship_matrix[agent0.id,:])
        ro = sum([agent.relationship(agent0) for agent in self.agents])
        return ro
    
    def closest_denser_neighbour(self, agent0: Agent) -> Agent:
        """
        the nearest collective neighbor with higher densiti
        """
        neighbour, distance = -1, None
        for agent in self.agents:
            if agent0.relationship(agent) and\
                self.collective_density(agent0) < self.collective_density(agent):
                    distance_ = (agent0.position - agent.position).norm()
                    if distance is None or distance_ < distance:
                        neighbour = agent
                        distance = distance_
        return neighbour
        
    
    def clusters(self) -> list[int]:
        """For every agent, there is the number of its cluster ex. [9,0,3,3,9]
        """
        collective_densities = np.array([self.collective_density(agent) for agent in self.agents])
        agent_ids = np.flip(np.argsort(collective_densities)) # indexes of agents by escending density
        
        clusters_of_agents = [None] * len(self.agents)
        
        clusters_of_agents[agent_ids[0]] = int(agent_ids[0])
        for agent_id in agent_ids[1:]:
            agent = self.agents[agent_id]
            neighbour = self.closest_denser_neighbour(agent)
            if neighbour != -1:
                clusters_of_agents[agent.id] = clusters_of_agents[neighbour.id]
            else:
                clusters_of_agents[agent.id] = agent.id
        print(clusters_of_agents)
        # TODO does this work correctly?
        return clusters_of_agents
        
        # # Alternative implementation -> list[set[int]]
        # d = {}
        # for i, cluster_label in enumerate(clusters_of_agents):
        #     if cluster_label in d:
        #         d[cluster_label].add(i)
        #     else:
        #         d[cluster_label] = set([i])
        # return d.values()
        
        
    
        
        

    def run(self):
        clusters_of_agents = self.clusters()
        self.environment.plot(self.agents, clusters_of_agents)
