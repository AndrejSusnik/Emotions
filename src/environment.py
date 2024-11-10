import numpy as np
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  # Import colors from matplotlib

env_map = {
    'w': 0,  # Wall
    'e': 1,  # Exit
    ' ': 2   # Empty space
}

class Environment:
    def __init__(self, filename: str):
        load_dotenv()
        filename = f'{os.getenv('ENVIRONMENTS_PATH')}/{filename}'
        with open(filename, 'r') as file:
            self.environment = np.array([np.array([env_map[char] for char in row.strip()]) for row in file.readlines()])

        self.size = self.environment.shape

    def get_valid_positions(self) -> set[tuple[int, int]]:
        return set(zip(*np.where(self.environment == 2)))
    
    def print(self):
        print(self.environment)

    def plot(self, agents):
        agents_pos = np.array([np.array([a.position.x, a.position.y]) for a in agents])

        full_env = np.copy(self.environment)
        for pos in agents_pos:
            full_env[pos[0], pos[1]] = 3

        cmap = mcolors.ListedColormap(['black', 'green', 'white', 'red'])

        plt.imshow(full_env, cmap=cmap)
        plt.axis('off')
        plt.show()


