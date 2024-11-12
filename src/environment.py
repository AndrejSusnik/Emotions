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
        """
        Reads the environment from a file
        """
        load_dotenv()
        filepath = f"{os.getenv('ENVIRONMENTS_PATH')}/{filename}"
        with open(filepath, 'r') as file:
            self.environment = np.array([np.array([env_map[char] for char in row.strip()]) for row in file.readlines()])

        self.size = self.environment.shape

    def get_valid_positions(self) -> set[tuple[int, int]]:
        return set(zip(*np.where(self.environment == 2)))
    
    def print(self):
        print(self.environment)

    def plot_discrete(self, agents):
        agents_pos = np.array([np.array([int(round(a.position.x)), int(round(a.position.y))]) for a in agents])

        full_env = np.copy(self.environment)
        for pos in agents_pos:
            full_env[pos[0], pos[1]] = 3

        cmap = mcolors.ListedColormap(['black', 'green', 'white', 'red'])

        plt.imshow(full_env, cmap=cmap)
        plt.axis('off')
        plt.show()
        
    def plot(self, agents, clusters_of_agents=None, with_arrows=False, arrow_scale=0.01):
        agents_pos = np.array([np.array([a.position.x, a.position.y]) for a in agents])
        
        if with_arrows:
            plt.quiver(agents_pos[:, 0], agents_pos[:, 1], [a.velocity.x * arrow_scale for a in agents],
                    [a.velocity.y * arrow_scale for a in agents], color='blue')

        if clusters_of_agents is None:
            plt.scatter(agents_pos[:, 0], agents_pos[:, 1])
        else:
            _, colors = np.unique(clusters_of_agents, return_inverse=True)
            plt.scatter(agents_pos[:, 0], agents_pos[:, 1], c=colors, cmap="plasma")
        
        
        
        a,b = self.size
        plt.plot([0,a],[0,0],'k')
        plt.plot([0,a],[b,b],'k')
        plt.plot([0,0],[0,b],'k')
        plt.plot([a,a],[0,b],'k')

    
        plt.show()


