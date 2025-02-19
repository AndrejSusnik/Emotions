import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

from agent import Agent
from environment import Environment
from exit import Exit
from helper_classes import Pair, Ocean, OceanDistribution, Line
from grid_draw import plot_navigation_graph
from joblib import Parallel, delayed
import time
import os


class SimulationParams:
    def __init__(self, num_agents: int, oceanDistribution: OceanDistribution, environment: Environment, use_panic=False, create_gif=False, simulation_time_in_seconds: int = 1000, dt: float = 0.1):
        self.num_agents = num_agents
        self.oceanDistribution = oceanDistribution
        self.environment = environment
        self.simulation_time_in_seconds = simulation_time_in_seconds
        self.dt = dt
        self.create_gif = False
        self.use_panic = False


class Simulation:
    """
    Environment class: environment initialization, emotion-to-path mapping, (local collision avoidance, crowd simulation)
    """

    def __init__(self, params: SimulationParams, mode="uniform"):
        self.agents: list[Agent] = []
        self.agents_at_destination: list[Agent] = []
        self.environment = params.environment
        # self.destinations = [(line.start.round(), line.center().round(), line.end.round()) for line in self.environment.exits]
        self.exits = [Exit(l.start, l.end, i)
                      for i, l in enumerate(self.environment.exits)]
        # TODO append all the points of the line
        self.params = params

        random.seed(42)
        # Example initialization
        for i in range(params.num_agents):
            a = Agent(i)
            a.traits = Ocean.sample(params.oceanDistribution)
            a.calculate_panic_factor()

            if mode == "uniform":
                xOffset = params.environment.size[0] * 0.1
                yOffset = params.environment.size[1] * 0.1
                while True:
                    a.source = Pair(xOffset + random.random(
                    ) * (self.environment.size[0] - 2*xOffset), yOffset + random.random() * (self.environment.size[1] - 2*yOffset)).round()
                    a.position = a.source
                    
                    # if a.position in set([a2.position for a2 in self.agents]):
                    #     # print("Not unique")
                    #     continue
                    if a.position in set([a2.position + delta for a2 in self.agents for delta in [Pair(0,0),Pair(0, 1), Pair(1, 0), Pair(-1, 0), Pair(0,-1),
                                                                                                  Pair(1, 1), Pair(1, -1), Pair(-1, 1), Pair(-1, -1)]]):
                        # print("Not unique")
                        continue

                    if self.environment.is_valid_position(a.source):
                        break
            # a.destination = Pair(random.random() * self.environment.size[0], random.random() * self.environment.size[1]).round()
            elif mode == "multimodal":

                centers = [Pair(18, 18), Pair(1, 15)]  # Example centers
                # Select a random center
                center = random.choice(centers)

                while True:
                    # Gaussian sampling around the selected center
                    # 10 is the standard deviation for x
                    x = np.random.normal(loc=center.x, scale=2)
                    # 10 is the standard deviation for y
                    y = np.random.normal(loc=center.y, scale=2)
                    p = Pair(x, y).round()
                    
                    
                    
                    if not self.environment.is_valid_position(p):
                        continue
                    
                    a.source = p
                    a.position = a.source
                    
                    if a.position in set([a2.position for a2 in self.agents]):
                        # print("Not unique")
                        continue
                    
                    break
                    


            # go to nearest exit
            # _, a.destination = min([((a.position - destination).norm(), destination) for destination in self.destinations])
            # TODO reasonable values
            # random.random(3, 10)
            if a.velocity.x is None:
                a.velocity = Pair(random.random() * 7 +3, random.random() * 7 +3)
            # a.velocity = Pair(0, 0)
            self.agents.append(a)

        # a = Agent(len(self.agents))
        # a.traits = Ocean.sample(params.oceanDistribution)
        # a.source = Pair(2, 15)
        # a.position = a.source
        # a.distance_preference =  0.1
        # a.init_distance_preference = 0.1
        # a.velocity_preference = 100
        # a.init_velocity_preference = 100
        # a.velocity = Pair(5, 5)

        # self.agents.append(a)
        
        print("NUMBER OF AGENTS", len(self.agents))
        print("NUMBER OF POSITIONS", len(set([a.position for a in self.agents])))

        print("Initialized agents")
        print("Number of agents: ", len(self.agents))
        print("Initializing navigation graphs")
        self.navigation_graphs = self.init_navigation_graphs(plot=False)
        print("Initialized navigation graphs")
        
        self.agents_x = [agent for agent in self.agents]
        self.agents_x.sort(key=lambda x: x.position.x)
        
        self.agents_y = [agent for agent in self.agents]
        self.agents_y.sort(key=lambda x: x.position.y)
        
    def update_agents_x_y(self):
        self.agents_x.sort(key=lambda x: x.position.x)
        self.agents_y.sort(key=lambda x: x.position.y)

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
        for agent in [a for a in self.agents]:
            # TODO maybe use different distribution for agent initialization than uniform
            if agent0.relationship(agent) == 1:
                if self.collective_density(agent0) < self.collective_density(agent):
                    distance_ = (agent0.position - agent.position).norm()
                    if distance is None or distance_ < distance:
                        neighbour = agent
                        distance = distance_
        return neighbour

    # def clusters(self, mode="default") -> list[int]:
    # def clusters(self, mode="fast_label_propagation") -> list[int]:
    def clusters(self, mode="hierarchical_clustering") -> list[int]:
        """For every agent, there is the number of its cluster ex. [9,0,3,3,9]
        """
        if mode == "default":
            if len(self.agents) == 0:
                return []
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
            # print(clusters_of_agents)
            # TODO does this work correctly? it is a bit random, maybe combining more iterations the thing is in average stable
            return clusters_of_agents
        elif mode == "fast_label_propagation":
            G = nx.Graph()
            # for agent in self.agents:
            #     # G.add_node(agent.id, density=self.collective_density(agent))
            #     G.add_node(agent.id)
            for agent in self.agents:
                for agent2 in self.agents:
                    if agent.id == agent2.id:
                        break
                    dist = (agent.position - agent2.position).norm()
                    print("dist", dist)
                    try:
                        if dist is not None and dist < 10:
                            G.add_edge(agent.id, agent2.id, weight=dist)
                    except:
                        pass
                        
            sets_of_nodes = nx.community.fast_label_propagation_communities(G, weight="dist")
            # sets_of_nodes = nx.community.louvain_communities(G, weight="dist", resolution=1.1)
            # if G.number_of_nodes == 0:
            #     return []
            # try:
            #     sets_of_nodes = nx.community.kernighan_lin_bisection(G, weight="weight")
            # except:
            #     return [0] * len(self.agents)
            
            l = [None] * len(self.agents)
            for i, s in enumerate(sets_of_nodes):
                print(s)
                for node in s:
                    node_id = int(node)
                    l[node_id] = i
            return l
        elif mode == "hierarchical_clustering":
            dists = dict()
            for agent in self.agents:
                for agent2 in self.agents:
                    if agent.id == agent2.id:
                        break
                    dists[(agent.id, agent2.id)] = (agent.position - agent2.position).norm()
            # iterativly merge the closest clusters
            l = [i for i in range(len(self.agents))]
            distances = sorted(dists.items(), key=lambda x: x[1])
            for (a, b), dist in distances:
                if l[a] != l[b]:
                    l = [l[a] if x == l[b] else x for x in l]
                if len(set(l)) == 2:
                    break
            return l
                    

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
        
        if len(self.agents) != 0 and self.agents[0].destination is None:
            return 
        
        prev_Pds = [agent.distance_preference for agent in self.agents]
        prev_Pvs = [agent.velocity_preference for agent in self.agents]

        for cluster in clusters:
            y_prev_Pd = []
            y_new_Pd = []
            y_prev_Pv = []
            y_new_Pv = []
            cluster = list(cluster)
            if len(cluster) == 1:
                if self.params.use_panic:
                    agent = self.agents[0]
                    # depending on the panic_factor user should move to the cluster average
                    agent.current_panic = agent.current_panic * 0.05
                    # print("Agent panic factor: ", agent.current_panic)

                continue

            average_panic = sum(
                [self.agents[i].current_panic for i in cluster]) / len(self.agents)

            for i in cluster:
                agent = self.agents[i]
                if self.params.use_panic:
                    # depending on the panic_factor user should move to the cluster average
                    panic_diff = average_panic - agent.current_panic
                    ave_dist = 1 / ((sum([(agent.position - con_s).norm() for con_s in self.environment.contagious_sources]) / len(
                        self.environment.contagious_sources)) + 1e-11)

                    if agent.current_panic < 0.8 * agent.panic_factor:
                        agent.current_panic += ave_dist * agent.panic_factor * 10e-1
                    elif agent.current_panic > agent.panic_factor:
                        agent.current_panic -= (agent.current_panic -
                                                agent.panic_factor) * 0.3

                    agent.current_panic += agent.panic_factor * panic_diff * 0.05
                    agent.current_panic = max(0, min(1, agent.current_panic))

                EPSI = 1e-11  # to avoid division by zero
                # (contagion inside cluster) + (contagion from contagious sources ex. fire)
                dPd = sum([np.exp((prev_Pds[j] - prev_Pds[i])/(agent.d_xy(self.agents[j]) + EPSI)) for j in cluster if j != i]) \
                    + sum([np.exp(prev_Pds[i]/((agent.position - contagious_source).norm() + EPSI))
                          for contagious_source in self.environment.contagious_sources])
                dPv = sum([np.exp((prev_Pvs[j] - prev_Pvs[i])/(agent.d_xy(self.agents[j]) + EPSI)) for j in cluster if j != i]) \
                    + sum([np.exp(prev_Pvs[i]/((agent.position - contagious_source).norm() + EPSI))
                          for contagious_source in self.environment.contagious_sources])
                    
                dPd = min(1, dPd)
                dPv = min(1, dPv)

                # selective perception
                # (distance preseption ... - distance to destination)
                # (velocity perception ... - velocity)
                wd = np.exp(-0.05 * (agent.position -
                            agent.destination.center()).norm())
                wv = np.exp(-2 * agent.velocity.norm())

                Cd = agent.init_distance_preference * 0.1
                Cv = agent.init_velocity_preference * 0.1

                # TODO suspicious, we get very high values (ex. 2 => 55), ok, if it will settle here after some iterations
                # TODO Above Pd there is a line in the article, but it is not clear what it means
                y_prev_Pd.append(agent.distance_preference)
                y_prev_Pv.append(agent.velocity_preference)
                
                agent.distance_preference = prev_Pds[i] + dPd * wd + Cd
                agent.velocity_preference = prev_Pvs[i] + dPv * wv + Cv
                
                y_new_Pd.append(agent.distance_preference)
                y_new_Pv.append(agent.velocity_preference)
                
            
                # if np.isnan(agent.distance_preference) or np.isnan(agent.velocity_preference):
                #     pass
                
                # if np.isinf(agent.distance_preference) or np.isinf(agent.velocity_preference):
                #     pass
                
                # print("Agent", agent.id, "Pd", agent.distance_preference, "Pv", agent.velocity_preference)
            # plt.figure()
            # plt.plot(range(len(cluster)), y_prev_Pd, label="Pd_prev")
            # plt.plot(range(len(cluster)), y_new_Pd, label="Pd_new")
            # plt.legend()
            # plt.show()
            
            
            # plt.plot(range(len(cluster)), y_prev_Pv, label="Pv_prev")
            # plt.plot(range(len(cluster)), y_new_Pv, label="Pv_new")
            # plt.legend()
            # plt.show()
            
            

    def init_navigation_graphs(self, plot=False):
        # maybe this gets more interesting when there are more rooms

        os.makedirs("cache", exist_ok=True)

        filename = "cache/navigation_graphs_" + str(self.environment.size[0]) + "_" + str(self.environment.size[1]) + "_" + self.environment.filename + ".npy"
        check = os.path.exists(filename)

        if check:
            grids = np.load(filename, allow_pickle=True)
            self.navigation_graphs = grids.item()
            
            if plot:
                for grid in grids.tolist().values():
                    plot_navigation_graph(grid)
            return self.navigation_graphs


        grids = dict()
        for exit in self.exits:
            grid = [[0] * (self.environment.size[1] + 1)
                    for _ in range(self.environment.size[0] + 1)]
            # print(grid)
            # h = len(grid[0])

            deltas = [Pair(0, 1), Pair(0, -1), Pair(1, 0), Pair(-1, 0),
                      Pair(1, 1), Pair(1, -1), Pair(-1, 1), Pair(-1, -1),
                      Pair(1, 2), Pair(1, -2), Pair(-1, 2), Pair(-1, -2),
                      Pair(2, 1), Pair(2, -1), Pair(-2, 1), Pair(-2, -1)]
            queue = []

            for current in exit.points():
                grid[current.x][current.y] = None
                for delta in deltas:
                    new = current + delta
                    queue.append({
                        "position": new,
                        "length": 0 + delta.norm(),
                        "parent": current
                    })

            while len(queue) > 0:
                queue = sorted(queue, key=lambda x: x["length"])
                d = queue.pop(0)
                current = d["position"]
                length = d["length"]
                parent = d["parent"]

                if self.environment.is_valid_position(current) and isinstance(grid[current.x][current.y], int):
                    # win for current
                    grid[current.x][current.y] = (parent, length)

                    random.shuffle(deltas)
                    for delta in deltas:
                        new = current + delta
                        if self.environment.is_valid_position(new):
                            queue.append({
                                "position": new,
                                "length": length + delta.norm(),
                                "parent": current
                            })

            grids[exit.id] = grid

        self.navigation_graphs = grids

        np.save(filename, grids)

        if plot:
            for grid in grids.values():
                plot_navigation_graph(grid)

        return grids

    def get_densities(self):
        # scan in the radious that is 1,5 times the size of the destination
        densities = dict()
        for exit in self.exits:
            # scan some radius back, and count the agents in the radius
            radious_in_tiles = (exit.end - exit.start).scale(2.5)
            density = 0
            for agent in self.agents:
                # check if agent.position is inside the radious starting from destination[1]
                if (agent.position - exit.center()).norm() < radious_in_tiles.norm():
                    density += 1
            densities[exit.id] = density / len(self.agents)
            # densities.append(0.5)
        print(densities)
        return densities

    def calc_new_pos(self, agent, densities_of_exits):
        max_score = None
        for exit in self.exits:
            grid = self.navigation_graphs[exit.id]
            current = agent.position
            if grid[current.x][current.y] is None:
                # TODO: what to do if the agent is already at the destination
                length = 0
            else:
                length = grid[current.x][current.y][1]

            density = densities_of_exits[exit.id]

            # TODO what is this and init in init (i think i doesn not influence the argmin)
            desired_velocity_of_agent = Pair(5, 5)

            # item exp is the adjustment factor for the desired velocity .
            # The larger den will lead the vel to get smaller, the crowded path will
            #   require more time to arrive at the destination

            score = length / (desired_velocity_of_agent.norm() * np.exp(-(
                density * (agent.velocity_preference+1) / (agent.distance_preference+1))**2))
            # print("exit", exit.id, "score", score)
            # if np.isnan(score):
            #     pass
            if max_score is None or score < max_score:
                max_score = score
                # agent_destination_id = i
                agent.destination = exit
        # print("picked exit", agent.destination.id)
        # update the position according to the selected path
        # speed = max(int(round(agent.velocity.norm() * self.params.dt)), 1)
        speed = agent.velocity.norm() * self.params.dt
        prev_position = Pair(agent.position.x, agent.position.y)

        deltas = [Pair(0, 1), Pair(0, -1), Pair(1, 0), Pair(-1, 0),
                  Pair(1, 1), Pair(1, -1), Pair(-1, 1), Pair(-1, -1),
                  Pair(1, 2), Pair(1, -2), Pair(-1, 2), Pair(-1, -2),
                  Pair(2, 1), Pair(2, -1), Pair(-2, 1), Pair(-2, -1)]

        move_random = False

        if self.params.use_panic:
            num = random.random()
            if num < agent.current_panic:
                move_random = True

            if move_random:
                while True:
                    pos = agent.position
                    random.shuffle(deltas)
                    for delta in deltas:
                        new_pos = pos + delta
                        collision = False
                        for other_agent in self.agents:
                            if other_agent.position == agent.position and other_agent.id != agent.id:
                                collision = True
                        if self.environment.is_valid_position(new_pos) and not collision:
                            agent.position = new_pos
                            break
                    break
                if agent.position in agent.destination.points():
                    agent.arrivied = True
                    agent.history.append(agent.position)

        # for i in range(speed):
        while speed > 0 and (not move_random):
            # if agent position is in between the start and end of the line
            if agent.position in agent.destination.points():
                agent.arrivied = True
                agent.history.append(agent.position)
                break

            agent.history.append(agent.position)
            agent.position = self.navigation_graphs[agent.destination.id][agent.position.x][agent.position.y][0]

            # 1 check if any other agent occupies the position
            # collision = False
            # for other_agent in self.agents:
            #     if other_agent.position == agent.position and other_agent.id != agent.id:
            #         collision = True

            # if collision:
            #     agent.position = prev_position
            #     # move to random unoccupied position one step back

            #     pos = agent.position
            #     random.shuffle(deltas)
            #     for delta in deltas:
            #         new_pos = pos + delta
            #         if self.environment.is_valid_position(new_pos):
            #             agent.position = new_pos
            #             break

            #     agent.colided = True
            #     break

            speed = speed - (prev_position - agent.position).norm()

        move = (agent.position - prev_position)
        move = move if move.norm() == 0 else move.scale(1/move.norm())
        # if agent.colided:
        #     agent.colided = False
        #     agent.velocity = Pair(random.random() * 2, random.random() * 2)
        # else:
        #     agent.velocity = (move).scale(agent.velocity.norm())
        agent.velocity = (move).scale(agent.velocity.norm())
        
    # @staticmethod
    # def bresenham_line(line):
    #     """Returns a list of grid cells that a line crosses using Bresenham's algorithm."""
    #     x1 = line.start.x
    #     y1 = line.start.y
    #     x2 = line.end.x
    #     y2 = line.end.y
        
    #     cells = []
        
    #     dx = abs(x2 - x1)
    #     dy = abs(y2 - y1)
    #     sx = 1 if x2 > x1 else -1
    #     sy = 1 if y2 > y1 else -1
    #     err = dx - dy

    #     while True:
    #         cells.append(Pair(x1, y1))
    #         if x1 == x2 and y1 == y2:
    #             break
    #         e2 = 2 * err
    #         if e2 > -dy:
    #             err -= dy
    #             x1 += sx
    #         if e2 < dx:
    #             err += dx
    #             y1 += sy

    #     return cells
    
    # @staticmethod
    # def dda_line(line):
    #     """Returns a set of grid cells that a line touches using the DDA algorithm."""
    #     x1 = line.start.x
    #     y1 = line.start.y
    #     x2 = line.end.x
    #     y2 = line.end.y
        
        
    #     cells = set()  # Use a set to avoid duplicates
        
    #     dx = x2 - x1
    #     dy = y2 - y1
    #     steps = max(abs(dx), abs(dy))  # Determine the number of steps
    #     x_step = dx / steps
    #     y_step = dy / steps
        
    #     x, y = x1, y1  # Start from the first point

    #     for _ in range(steps + 1):  # Include last point
    #         cell_x = int(x)  # Get bottom-left x of the grid cell
    #         cell_y = int(y)  # Get bottom-left y of the grid cell
    #         cells.add((cell_x, cell_y))  # Add to set
            
    #         x += x_step  # Increment x
    #         y += y_step  # Increment y
            
    #         # Check if the line crosses into a new cell at a corner
    #         corner_x = int(x)
    #         corner_y = int(y)
    #         cells.add((corner_x, corner_y))  # Ensure corners are counted

    #     return sorted(cells)  # Return sorted list for readability

    
    
    # @staticmethod
    # def touched_cells(line):
        
    #     def cell_representation(line):
    #         """Returns a list of all faces (i, j) touched by the line from (x1, y1) to (x2, y2)"""
            
    #         x1 = line.start.x
    #         y1 = line.start.y
    #         x2 = line.end.x
    #         y2 = line.end.y
    #         faces = set()  # Use a set to store unique faces
            
    #         # Initialize stepping variables
    #         dx, dy = x2 - x1, y2 - y1
    #         sx = 1 if dx > 0 else -1
    #         sy = 1 if dy > 0 else -1
    #         dx, dy = abs(dx), abs(dy)
            
    #         # Current grid face
    #         x, y = x1, y1
    #         faces.add(Pair(x, y))

    #         # Initial error terms
    #         if dx > dy:
    #             err = dx / 2.0
    #             while x != x2:
    #                 x += sx
    #                 err -= dy
    #                 if err < 0:
    #                     y += sy
    #                     err += dx
    #                 faces.add(Pair(x, y))
    #         else:
    #             err = dy / 2.0
    #             while y != y2:
    #                 y += sy
    #                 err -= dx
    #                 if err < 0:
    #                     x += sx
    #                     err += dy
    #                 faces.add(Pair(x, y))

    #         return sorted(faces)
        
    #     cells = cell_representation(line)
        
    #     new_cells = []
    #     for i in range(len(cells) - 1):
    #         x1  = cells[i].x
    #         y1 = cells[i].y
    #         x2 = cells[i + 1].x
    #         y2 = cells[i + 1].y
    #         if x1 == x2 or y1 == y2:
    #             continue
    #         # new_cells.append(cells[i])
    #         if random.random() < 0.5:
    #             new_cells.append(Pair(x2, y1))
    #         else:
    #             new_cells.append(Pair(x1, y2))
        
    #     new_cells += [cells[i] - delta for delta in [Pair(0, 1), Pair(1, 0), Pair(1, 1)] for i in [0,-1]]
        
    #     return sorted(set(cells + new_cells))
        


    # if __name__ == "__main__":
    #     # Example: Line from (0,0) to (9,9) in a 10x10 grid
    #     # line = Line(Pair(1,1), Pair(2,3))
    #     line = Line(Pair(1,1),Pair(1,1))
    #     cells = touched_cells(line)
    #     print(cells)
        


    def select_path(self, collision_avoidance=True):
        # if we choose a destination, the path is selected, as there is only one shortest path in the navigation graph of this destination
        # densities_of_destinations = [self.density_of_destination(destination) for destination in self.destinations]
        densities_of_exits = self.get_densities()
        # agent_exit_id = None
        tmp_agents = self.agents.copy()

        # for agent in self.tmp_agents:
        #     self.calcNewPas(agent, densities_of_exits)

        t = time.time()
        _ = Parallel(n_jobs=-1, prefer="threads")(delayed(self.calc_new_pos)
                                                  (agent, densities_of_exits) for agent in tmp_agents)
        print(time.time() - t)

        self.agents = tmp_agents
        
        # resolve conflicts, if the paths intersect, one will not move
        # if collision_avoidance:
        #     for agent in self.agents:
        #         for agent2 in self.agents:
        #             if agent.id == agent2.id:
        #                 break
        #             line1 = Line(agent.history[-1], agent.position)
        #             line2 = Line(agent2.history[-1], agent2.position)
        #             intersection = line1.intersection(line2)
        #             if intersection is not None:
        #                 # the spot just got free (both can go)
        #                 if intersection == agent.position and intersection == agent2.history[-1] or \
        #                     intersection == agent2.position and intersection == agent.history[-1]:
        #                     continue
                        
        #                 # only one can go
        #                 if random.random() < 0.5:
        #                     agent.position = agent.history[-1].copy()
        #                     agent.arrivied = agent.position in agent.destination.points()
        #                 else:
        #                     agent2.position = agent2.history[-1].copy()
        #                     agent2.arrivied = agent2.position in agent.destination.points()
        #                 break
        
        # # linear time
        # if collision_avoidance:
        #     d = dict()
            
        #     for agent in self.agents:
        #         line = Line(agent.history[-1], agent.position)
        #         occupied_cells = Simulation.touched_cells(line)
        #         print(agent.id, occupied_cells)
        #         for cell in occupied_cells:
        #             if cell in d:
        #                 d[cell].append(agent.id)
        #             else:
        #                 d[cell] = [agent.id]
                        
        # resolve conflicts, if the paths intersect, one will not move
        if collision_avoidance:
            self.update_agents_x_y()
            agent_pairs = []
            k = 10 # number of neighbours to the right (up) to check against me
            for ordering in [self.agents_x, self.agents_y]:
                for i in range(len(ordering)-k-1):
                    for j in range(k):
                        agent_pairs.append((ordering[i], ordering[i+1+k]))
            
                        
            for agent, agent2 in agent_pairs:
                    line1 = Line(agent.history[-1], agent.position)
                    line2 = Line(agent2.history[-1], agent2.position)
                    intersection = line1.intersection(line2)
                    if intersection is not None:
                        # the spot just got free (both can go)
                        if intersection == agent.position and intersection == agent2.history[-1] or \
                            intersection == agent2.position and intersection == agent.history[-1]:
                            continue
                        
                        # only one can go
                        if random.random() < 0.5:
                            agent.position = agent.history[-1].copy()
                            agent.arrivied = agent.position in agent.destination.points()
                        else:
                            agent2.position = agent2.history[-1].copy()
                            agent2.arrivied = agent2.position in agent.destination.points()
                        break
            
            # print(d)
            # for cell, agents in d.items():
            #     if len(agents) > 1:
            #         print(cell, end=", ")
            #         for agent_id in agents:
            #             agent = [a for a in self.agents if a.id == agent_id][0]
            #             print(agent.org_id, end=", ")
            #         print()
            
            
            # stopped_agents = set()
            # for cell, agents in d.items():
            #     if cell == Pair(22,9):
            #         pass
                
            #     if cell == Pair(23,10):
            #         pass
            #     if cell in agent.destination.points():
            #         continue
            #     remaining_agents = [agent for agent in agents if agent not in stopped_agents]
            #     if len(remaining_agents) == 0:
            #         continue
                
            #     if len([1 for agent in stopped_agents if self.agents[agent].position == cell and self.agents[agent].history[-1] == cell]) > 0:
            #         approved_agent = None
            #     else:
            #         approved_agent = random.choice(remaining_agents)           
                    
            #     for agent_id in remaining_agents:
            #         if agent_id != approved_agent:
            #             # stop unapproved agents
            #             stopped_agents.add(agent_id)
            #             agent = self.agents[agent_id]
            #             agent.position = agent.history[-1].copy()
            #             agent.arrivied = agent.position in agent.destination.points()
                            
            # # TODO remove
            # for agent in self.agents:
            #     for agent2 in self.agents:
            #         if agent == agent2:
            #             break
            #         if agent.position == agent2.position:
            #             pass
                        
        for agent in self.agents:
            if agent.arrivied:
                self.agents_at_destination.append(agent)
        self.agents = [agent for agent in self.agents if not agent.arrivied]
        for i, agent in enumerate(self.agents):
            agent.id = i

    def run(self, clustering_mode, collision_avoidance, color_mode = "id"):
        # delete all the files in plots folder
        # if plots folder does not exist create it 
        if not os.path.exists("plots"):
            os.mkdir("plots")

        for file in os.listdir("plots"):
            os.remove(os.path.join("plots", file))

        print("Creating clusters")
        clusters_of_agents = self.clusters(mode=clustering_mode)
        print("Created clusters. Calculating contagion of emotion preferences")
        self.environment.plot(
            self.agents, clusters_of_agents, with_arrows=True)
        # self.environment.plot_discrete(self.agents)
        self.contagion_of_emotion_preferences(
            Simulation.labels_to_clusters(clusters_of_agents))
        print("Calculated contagion of emotion preferences. Calculating navigation graphs")

        num_steps = int(
            self.params.simulation_time_in_seconds / self.params.dt)

        for i in range(num_steps):
            print(f"Step {i}")
            print(f"Number of agents: {len(self.agents)}")
            if len(self.agents) == 0:
                break
            self.select_path(collision_avoidance=collision_avoidance)
            clusters_of_agents = self.clusters(mode=clustering_mode)
            self.contagion_of_emotion_preferences(
                Simulation.labels_to_clusters(clusters_of_agents))
            self.environment.plot(
                self.agents, clusters_of_agents, with_arrows=False, save=True, step=i,
                color_mode = color_mode)

        self.environment.plot(
            self.agents, clusters_of_agents, with_arrows=False, color_mode = color_mode)
        self.environment.plot_path(self.agents_at_destination, save=True)
        self.environment.create_gif()
