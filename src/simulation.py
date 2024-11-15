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
    def __init__(self, params: SimulationParams, mode = "uniform"):
        self.agents = []
        self.environment = params.environment

        if mode == "uniform":
            # Example initialization
            for i in range(params.num_agents):
                a = Agent(i)
                a.traits = Ocean.sample(params.oceanDistribution)
                a.source = Pair(random.random() * self.environment.size[0], random.random() * self.environment.size[1]).round()
                a.position = a.source
                a.destination = Pair(random.random() * self.environment.size[0], random.random() * self.environment.size[1]).round()
    

                a.velocity = Pair(random.random() *10 -5, random.random() *10 -5) # random.random(-5, 5)
                self.agents.append(a)
                
             
        elif mode == "multimodal":
            centers = [Pair(2,15),Pair(8,7)]  # Example centers

            for i in range(params.num_agents):
                a = Agent(i)
                a.traits = Ocean.sample(params.oceanDistribution)
                
                # Select a random center
                center = random.choice(centers)
                
                while True:
                
                    # Gaussian sampling around the selected center
                    x = np.random.normal(loc=center.x, scale=2)  # 10 is the standard deviation for x
                    y = np.random.normal(loc=center.y, scale=2)  # 10 is the standard deviation for y
                    p = Pair(x, y).round()
                    if self.environment.is_valid_position(p):
                        a.source = p
                        a.position = a.source
                        break


                # Destination sampled independently
                dest_center = random.choice(centers)
                while True:
                    dest_x = np.random.normal(loc=dest_center.x, scale=2)
                    dest_y = np.random.normal(loc=dest_center.y, scale=2)
                    p = Pair(dest_x, dest_y).round()
                    if self.environment.is_valid_position(p):
                        a.destination = p
                        break

                # Velocity remains uniformly random
                a.velocity = Pair(random.random() * 10 - 5, random.random() * 10 - 5)
                
                self.agents.append(a)
            
        
                
        
            
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
    
    @staticmethod
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
        prev_Pds = [agent.distance_preference for agent in self.agents]
        prev_Pvs = [agent.velocity_preference for agent in self.agents]
        
        for cluster in clusters:
            cluster = list(cluster)
            if len(cluster) == 1:
                continue
            for i in cluster:
                agent = self.agents[i]
                
                # (contagion inside cluster) + (contagion from contagious sources ex. fire)
                dPd = sum([np.exp((prev_Pds[j] - prev_Pds[i])/agent.d_xy(self.agents[j])) for j in cluster if j != i]) \
                    + sum([np.exp(prev_Pds[i]/((agent.position - contagious_source).norm())) for contagious_source in self.environment.contagious_sources])
                dPv = sum([np.exp((prev_Pvs[j] - prev_Pvs[i])/agent.d_xy(self.agents[j])) for j in cluster if j != i]) \
                    + sum([np.exp(prev_Pvs[i]/((agent.position - contagious_source).norm())) for contagious_source in self.environment.contagious_sources])
                
                # selective perception
                # (distance preseption ... - distance to destination)
                # (velocity perception ... - velocity)
                wd = np.exp(-0.05 * (agent.position - agent.destination).norm())
                wv = np.exp(-2 * agent.velocity.norm())
                
                Cd = agent.init_distance_preference * 0.1
                Cv = agent.init_velocity_preference * 0.1
                    
                # TODO suspicious, we get very high values (ex. 2 => 55), ok, if it will settle here after some iterations
                # TODO Above Pd there is a line in the article, but it is not clear what it means 
                agent.distance_preference = prev_Pds[i] + dPd * wd + Cd
                agent.velocity_preference = prev_Pvs[i] + dPv * wv + Cv

    
        
        

    def run(self):
        # self.environment.plot(self.agents)
        clusters_of_agents = self.clusters()
        self.environment.plot(self.agents, clusters_of_agents, with_arrows=True)
        print(str(self.agents[0]))
        self.contagion_of_emotion_preferences(Simulation.labels_to_clusters(clusters_of_agents))
        print(str(self.agents[0]))
