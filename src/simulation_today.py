import numpy as np
import random

from agent import Agent
from environment import Environment
from helper_classes import Pair, Ocean, OceanDistribution
from helper_classes import Point

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
        self.destinations = [(line.start.round(), line.center().round(), line.end.round()) for line in self.environment.exits]
        # TODO append all the points of the line

        
        # Example initialization
        for i in range(params.num_agents):
            a = Agent(i)
            a.traits = Ocean.sample(params.oceanDistribution)
            
            if mode == "uniform":
                a.source = Pair(random.random() * self.environment.size[0], random.random() * self.environment.size[1]).round()
                a.position = a.source
            # a.destination = Pair(random.random() * self.environment.size[0], random.random() * self.environment.size[1]).round()
            elif mode == "multimodal":
                
                centers = [Pair(2,15),Pair(8,7)]  # Example centers
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
                    
            
            # go to nearest exit
            # _, a.destination = min([((a.position - destination).norm(), destination) for destination in self.destinations])
            # TODO reasonable values
            a.velocity = Pair(random.random() *10, random.random() *10) # random.random(-5, 5)
            self.agents.append(a)
        
        self.navigation_graphs = self.init_navigation_graphs()
                
        
            
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
        agent_ids = np.flip(np.argsort(collective_densities)) # indexes of agents by descending density
        
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
                EPSI = 1e-11 # to avoid division by zero
                # (contagion inside cluster) + (contagion from contagious sources ex. fire)
                dPd = sum([np.exp((prev_Pds[j] - prev_Pds[i])/(agent.d_xy(self.agents[j])+ EPSI)) for j in cluster if j != i]) \
                    + sum([np.exp(prev_Pds[i]/((agent.position - contagious_source).norm() + EPSI)) for contagious_source in self.environment.contagious_sources])
                dPv = sum([np.exp((prev_Pvs[j] - prev_Pvs[i])/(agent.d_xy(self.agents[j])+ EPSI)) for j in cluster if j != i]) \
                    + sum([np.exp(prev_Pvs[i]/((agent.position - contagious_source).norm() + EPSI)) for contagious_source in self.environment.contagious_sources])
                
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

    def init_navigation_graphs(self):
        # maybe this gets more interesting when there are more rooms
        grids = dict()
        for destination in self.destinations:
            print(destination)
            # grid = np.zeros(self.environment.size + 1)
            grid = [[0] * (self.environment.size[1] + 1) for _ in range(self.environment.size[0] + 1)]
            # h = grid.shape[1] #hight
            h = len(grid[0])
            
            # start at destination and perform BFS to write names of the parents in the grid cells
            # grid[destination.x, h - destination.y] = -1
            queue = []
            for point in destination:
                grid[point.x][point.y] = None
                queue.append([point,0])
                
            while len(queue) > 0:
                current, length = queue.pop(0)
                for delta in [Pair(0,1), Pair(0,-1), Pair(1,0), Pair(-1,0)]:
                    new = current + delta
                    # print(new)
                    if self.environment.is_valid_position(new) and isinstance(grid[new.x][new.y], int):
                        grid[new.x][new.y] = (current, length)
                        queue.append((new, length + 1))
            for line in grid:
                print(line)
                
            grids[destination] = grid
                
       
        
        return grids
    
    def get_densities(self):
        densities = []
        for destination in self.destinations:
            # scan some radius back, and count the agents in the radius
            for agent in self.agents:
                # TODO what is the radius? 5? 10?
                # TODO Line ali pa vse tocke na liniji
                if (agent.position - destination[1]).norm() < 5:
                    pass
                    # TODO
             
        return densities
            
    
    def select_path(self):
        # if we choose a destination, the path is selected, as there is only one shortest path in the navigation graph of this destination
        # densities_of_destinations = [self.density_of_destination(destination) for destination in self.destinations]
        densities_of_destinations = self.get_densities()
        agent_destination_id = None
        max_score = None
        for agent in self.agents:        
            for i, destination in enumerate(self.destinations):
                grid = self.navigation_graphs[destination]
                current = agent.position
                length = grid[current.x][current.y][1]
                density = densities_of_destinations[i]
                
                desired_velocity_of_agent = Pair(5,5) # TODO what is this and init in init (i think i doesn not influence the argmin)
                
                # item exp is the adjustment factor for the desired velocity .
                # The larger den will lead the vel to get smaller, the crowded path will
                #   require more time to arrive at the destination
                
                score = length / (desired_velocity_of_agent.norm() * np.exp(-(density * (agent.velocity_preference+1)/ (agent.distance_preference+1))**2 ))
                if max_score is None or score > max_score:
                    max_score = score
                    agent_destination_id = i
                    
            # update the position according to the selected path
            speed = int(round(agent.velocity.norm())) # number of nodes to go in one step
            prev_position = Point(agent.position.x, agent.position.y)
            for i in range(speed):
                # CA
                agent.position = self.navigation_graphs[agent_destination_id][agent.position.x][agent.position.y][0]
                
            move = (agent.position - prev_position)
            agent.velocity = speed * (move / move.norm())
        
        
        # TODO what should be destination? a line maybe? refactor
        # TODO save the destination(id) to agent so that relationship works
        
        
        # TODO
        # collision avoidance
        # kdo ma prednost iti na neko polje ? sortiramo po agresivnosti
        
        
                
                
                
            
            
            
        
        
        

    def run(self):
        # self.environment.plot(self.agents)
        clusters_of_agents = self.clusters()
        self.environment.plot(self.agents, clusters_of_agents, with_arrows=True)
        self.environment.plot_discrete(self.agents)
        print(str(self.agents[0]))
        self.contagion_of_emotion_preferences(Simulation.labels_to_clusters(clusters_of_agents))
        print(str(self.agents[0]))
        
        self.init_navigation_graphs()