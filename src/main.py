from simulation import Simulation, SimulationParams
from environment import Environment
from helper_classes import OceanDistribution, Pair


if __name__ == '__main__':
    oceanDistribution = OceanDistribution(Pair(0.5, 0.1), Pair(0.5, 0.1), Pair(0.5, 0.1), Pair(0.5, 0.1), Pair(0.5, 0.1))
    environment = Environment('test1.txt', size_in_meters=Pair(5,5), tile_size_in_meters=Pair(0.35, 0.35))

    sim_params = SimulationParams(num_agents=10, oceanDistribution=oceanDistribution, environment=environment)

    sim = Simulation(sim_params,mode="uniform")
    sim.run()