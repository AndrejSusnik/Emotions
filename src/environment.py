import numpy as np
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  # Import colors from matplotlib
from agent import Agent
import imageio
from bmp_parser import parse_bmp, write_bmp
from exit import ExitEx

from helper_classes import Pair, Line, Rect

env_map = {
    'w': 0,  # Wall
    'e': 1,  # Exit
    ' ': 2,  # Empty space
    'o': 3   # Obstacle
}


class Environment:
    def __init__(self, filename: str, size_in_meters: Pair, tile_size_in_meters: Pair, with_obstacles=False):
        """
        Reads the environment from a file
        """
        load_dotenv()
        path = os.getenv("ENVIRONMENTS_PATH")
        exits, walls, size, raw = parse_bmp(path  +"/"+ filename)

        self.raw_img = raw

        self.filename = filename
        self.size = size
        self.exits: list[ExitEx] = exits
        self.walls: list[Pair] = walls

        # self.full_env = np.zeros(size.get())

        # for wall in walls:
        #     self.full_env[wall.get()] = env_map['w']

        # for exit in exits:
        #     for point in exit.points:
        #         self.full_env[point.get()] = env_map['e']

        self.contagious_sources = []
        


    def is_valid_position(self, position: Pair) -> bool:
        # if it is in bounds and not a wall and not an exit
        if position.x < 0 or position.y < 0 or position.x >= self.size.x or position.y >= self.size.y:
            return False

        if position in self.walls:
            return False

        for exit in self.exits:
            if position in exit.points:
                return False
        
        return True

    def plot_path(self, agents: list[Agent], save = False):
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
            
        if save:
            plt.savefig(f"plots/path_plot.png")
            plt.show()
        else:
            plt.show()

    def plot(self, agents, clusters_of_agents=None, with_arrows=False, arrow_scale=0.01, save=False, step=None):
        plt.close()
        agents_pos = np.array(
            [np.array([a.position.x, a.position.y]) for a in agents]).reshape(-1, 2)

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
            for point in exit.points:
                plt.scatter(point.x, point.y, c='red', s=100)

        for wall in self.walls:
            plt.scatter(wall.x, wall.y, c='black', s=100)

            
        for contagious_source in self.contagious_sources:
            plt.scatter(contagious_source.x, contagious_source.y, c='red', s=100)

        if step:
            step = str(step + 1).zfill(4)

        if save:
            plt.savefig(f"plots/plot_{step}.png")
        else:
            plt.show()

    def create_gif(self):
        # create gif from plots in plots folder
        print("Creating gif")
        filenames = [filename for filename in os.listdir('plots')]
        filenames.sort()
        images = []
        for filename in filenames:
            images.append(imageio.imread(f'plots/{filename}'))

        imageio.mimsave('plots/test.gif', images, 'GIF', loop=1, duration=1, fps=1)
        print("Gif created")

    def draw_bmp(self, agents, clusters, step):
        filename = f"plots/plot_{str(step).zfill(4)}.bmp"

        tmp = np.copy(self.raw_img)

        for agent in agents:
            # agents are green dots
            tmp[agent.position.y, agent.position.x] = [0, 255, 0]



        write_bmp(tmp, filename)