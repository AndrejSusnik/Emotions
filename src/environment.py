import numpy as np
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  # Import colors from matplotlib
from agent import Agent

from helper_classes import Pair, Line, Rect

env_map = {
    'w': 0,  # Wall
    'e': 1,  # Exit
    ' ': 2,  # Empty space
    'o': 3   # Obstacle
}

class Environment:
    def __init__(self, filename: str, size_in_meters: Pair, tile_size_in_meters: Pair):
        """
        Reads the environment from a file
        """
        load_dotenv()
        filepath = f"{os.getenv('ENVIRONMENTS_PATH')}/{filename}"
        with open(filepath, 'r') as file:
            self.environment = np.array(
                [np.array([env_map[char] for char in row.strip()]) for row in file.readlines()])

        self.size_in_meters = size_in_meters
        self.tile_size_in_meters = tile_size_in_meters

        #  extract the exits and walls in relative coordinates
        self.exits: list[Line] = []
        self.walls: list[Line] = []
        self.obstacles: list[Pair] = []

        for i in range(self.environment.shape[1]):
            wall_started = False
            wall_start = None

            exit_started = False
            exit_start = None
            for j in range(self.environment.shape[0]):
                if (self.environment[j, i] == 0) and not wall_started:
                    wall_started = True
                    wall_start = Pair(i, j)

                if wall_start and (self.environment[j, i] != 0):
                    if (not (wall_start.x == i and wall_start.y == j-1)):
                        self.walls.append(Line(wall_start, Pair(i, j - 1)))
                    wall_started = False
                    wall_start = None

                if self.environment[j, i] == 1 and not exit_started:
                    exit_started = True
                    exit_start = Pair(i, j)
                
                if exit_start and (self.environment[j, i] != 1):
                    if (not (exit_start.x == i and exit_start.y == j-1)):
                        self.exits.append(Line(Pair(exit_start.x, exit_start.y - 1), Pair(i, j)))
                    exit_started = False
                    exit_start = None

            if wall_started and wall_start and (not (wall_start.x == i and wall_start.y == self.environment.shape[0]-1)):
                self.walls.append(
                    Line(wall_start, Pair(i, self.environment.shape[0] - 1)))

        for i in range(self.environment.shape[0]):
            wall_started = False
            wall_start = None   
            exit_started = False
            exit_start = None
 
            for j in range(self.environment.shape[1]):
                if (self.environment[i, j] == 0) and not wall_started:
                    wall_started = True
                    wall_start = Pair(j, i)

                if wall_start and (self.environment[i, j] != 0):
                    if (not (wall_start.x == j-1 and wall_start.y == i)):
                        self.walls.append(Line(wall_start, Pair(j - 1, i)))
                    wall_started = False
                    wall_start = None

                if self.environment[i, j] == 1 and not exit_started:
                    exit_started = True
                    exit_start = Pair(j, i)

                if exit_start and (self.environment[i, j] != 1):
                    if (not (exit_start.x == j-1 and exit_start.y == i)):
                        self.exits.append(Line(Pair(exit_start.x - 1, exit_start.y), Pair(j, i)))
                    exit_started = False
                    exit_start = None

            if wall_started and wall_start and (not (wall_start.x == self.environment.shape[1]-1 and wall_start.y == i)):
                self.walls.append(
                    Line(wall_start, Pair(self.environment.shape[1] - 1, i)))

        for i in range(self.environment.shape[0]):
            for j in range(self.environment.shape[1]):
                if self.environment[i, j] == 3:
                    self.obstacles.append(Pair(j, i))

        size = Pair(round(size_in_meters.x / tile_size_in_meters.x), round(size_in_meters.y / tile_size_in_meters.y))

        tmp_obst = []
        # scale each obsticle 
        scale_factor_x = round((size_in_meters.x / tile_size_in_meters.x) / (self.environment.shape[1] - 1))
        scale_factor_y = round((size_in_meters.y / tile_size_in_meters.y) / (self.environment.shape[0] - 1))


        for obst in self.obstacles:
            # every obstacle becomes scale_factor_x*scale_factor_y obstacles in the scaled coordinate system
            for i in range(scale_factor_x):
                for j in range(scale_factor_y):
                    tmp_obst.append(Pair(obst.x * scale_factor_x + i, obst.y * scale_factor_y + j))
        
        self.obstacles = tmp_obst

        self.exits = list(map(lambda x: x.norm(Pair(
            self.environment.shape[1] - 1, self.environment.shape[0] - 1)).scale(size), self.exits))
        self.walls = list(map(lambda x: x.norm(Pair(
            self.environment.shape[1] - 1, self.environment.shape[0] - 1)).scale(size), self.walls))

        self.size = np.array([size.x, size.y])

        # self.size = self.environment.shape
        
        # self.contagious_sources = []
        self.contagious_sources = [Pair(0,0)] # ex. fire at position 0,0

    # def get_valid_positions(self) -> set[tuple[int, int]]:
    #     return set(zip(*np.where(self.environment == 2)))

    def scale_obstacles(self, scale: Pair):
        # 
        pass
    
    def is_valid_position(self, position: Pair) -> bool:
        xx, yy = self.size
        is_in_bounds = 0 <= position.x and position.x <= xx and 0 <= position.y and position.y <= yy
        is_not_a_wall = True
        # for wall in self.walls:
        #     is_not_a_wall = is_not_a_wall and Pair(position.x, position.y) not in wall.points()

        is_not_an_obstacle = True
        for obstacle in self.obstacles:
            is_not_an_obstacle = is_not_an_obstacle and (position.x != obstacle.x or position.y != obstacle.y)

        return is_in_bounds and is_not_a_wall and is_not_an_obstacle
    
    def print(self):
        print(self.environment)

    def plot_discrete(self, agents):
        agents_pos = np.array(
            [np.array([int(round(a.position.x)), int(round(a.position.y))]) for a in agents])

        # full_env = np.copy(self.environment)
        # print(self.environment) # TODO its the 10x10 one, not 100x100 ??
        
      
        full_env = np.zeros(self.size + 1)
        # print(full_env.shape)
        for pos in agents_pos:
            full_env[pos[0], pos[1]] = 3

        cmap = mcolors.ListedColormap(['black', 'green', 'white', 'red'])

        plt.imshow(full_env, cmap=cmap)
        plt.axis('off')
        plt.show()

    def plot_path(self, agents: list[Agent]):
        # 
        # agents[0].history
        # plot the history of all agents
        for exit in self.exits:
            plt.plot([exit.start.x, exit.end.x], [
                     exit.start.y, exit.end.y], 'r')

        for wall in self.walls:
            plt.plot([wall.start.x, wall.end.x], [
                     wall.start.y, wall.end.y], 'k')

        for obstacle in self.obstacles:
            plt.scatter(obstacle.x, obstacle.y, c='black', s=100)

        for agent in agents:
            history = agent.history
            history = np.array(
                [np.array([int(round(a.x)), int(round(a.y))]) for a in history])
            plt.plot(history[:, 0], history[:, 1])
        plt.show()


    def plot(self, agents, clusters_of_agents=None, with_arrows=False, arrow_scale=0.01):
        agents_pos = np.array(
            [np.array([a.position.x, a.position.y]) for a in agents]).reshape(-1,2)

        if with_arrows:
            plt.quiver(agents_pos[:, 0], agents_pos[:, 1], [a.velocity.x * arrow_scale for a in agents],
                       [a.velocity.y * arrow_scale for a in agents], color='blue')

        if clusters_of_agents is None:
            plt.scatter(agents_pos[:, 0], agents_pos[:, 1])
        else:
            _, colors = np.unique(clusters_of_agents, return_inverse=True)
            # print(colors)
            plt.scatter(agents_pos[:, 0], agents_pos[:,
                        1], c=colors+1, cmap="plasma")
            plt.colorbar()

        # plot the exits 
        for exit in self.exits:
            plt.plot([exit.start.x, exit.end.x], [
                     exit.start.y, exit.end.y], 'r')

        for wall in self.walls:
            plt.plot([wall.start.x, wall.end.x], [
                     wall.start.y, wall.end.y], 'k')
    
        for obstacle in self.obstacles:
            plt.scatter(obstacle.x, obstacle.y, c='black', s=100)

        # a, b = self.size
        # plt.plot([0, a], [0, 0], 'k')
        # plt.plot([0, a], [b, b], 'k')
        # plt.plot([0, 0], [0, b], 'k')
        # plt.plot([a, a], [0, b], 'k')

        plt.show()
