from simulation import Simulation, SimulationParams
from environment import Environment
from helper_classes import OceanDistribution, Pair


if __name__ == '__main__':
    oceanDistribution = OceanDistribution(Pair(0.5, 0.1), Pair(0.5, 0.1), Pair(0.5, 0.1), Pair(0.5, 0.1), Pair(0.5, 0.1))
    environment = Environment('test1.txt')

    sim_params = SimulationParams(num_agents=10, oceanDistribution=oceanDistribution, environment=environment)

    sim = Simulation(sim_params)
    sim.run()