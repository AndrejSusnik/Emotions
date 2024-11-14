import numpy as np
import random

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

        # Example initialization
        for i in range(params.num_agents):
            a = Agent(i)
            a.traits = Ocean.sample(params.oceanDistribution)
            a.source = Pair(random.random() * self.environment.size[0], random.random() * self.environment.size[1])
            a.position = a.source
            a.destination = Pair(random.random() * self.environment.size[0], random.random() * self.environment.size[1])
 

            a.velocity = Pair(random.random() *10 -5, random.random() *10 -5) # random.random(-5, 5)
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
            # TODO maybe use different distribution for agent initialization than uniform
            if agent0.relationship(agent) == 1:
                if self.collective_density(agent0) < self.collective_density(agent):
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
        # TODO does this work correctly? it is a bit random, maybe combining more iterations the thing is in average stable
        return clusters_of_agents
    
    def labels_to_clusters(clusters_of_agents: list[int]) -> list[set[int]]:
        """Alternative representaion of clusters ex. [{0,3},{1,2,4}]"""
        d = {}
        for i, cluster_label in enumerate(clusters_of_agents):
            if cluster_label in d:
                d[cluster_label].add(i)
            else:
                d[cluster_label] = set([i])
        sets = d.values()
        return sets

    def contagion_of_emotion_preferences(self, clusters: list[set[int]]):
        """Update the emotion preferences (distance Pd, velocity Pv) of the agent based on the cluster he is in
        """
        # prev_Pds = [agent.distance_preference for agent in self.agents]
        # prev_Pvs = [agent.velocity_preference for agent in self.agents]
        
        # for cluster in clusters:
        #     cluster = list(cluster)
        #     if len(cluster) == 1:
        #         continue
        #     for i in cluster:
        #         agent = self.agents[i]
        #         # TODO implement
        #         # Pd = ...
        #         # Pv = ..
        #         # agent.distance_preference = Pd
        #         # agent.velocity_preference = Pv
        pass
        
    
        
        

    def run(self):
        # self.environment.plot(self.agents)
        clusters_of_agents = self.clusters()
        self.environment.plot(self.agents, clusters_of_agents, with_arrows=True)
