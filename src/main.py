from simulation import Simulation, SimulationParams
from environment import Environment
from helper_classes import OceanDistribution, Pair


if __name__ == '__main__':
    oceanDistribution = OceanDistribution(Pair(0.5, 0.1), Pair(0.5, 0.1), Pair(0.5, 0.1), Pair(0.5, 0.1), Pair(0.5, 0.1))
    print("Creating environment")
    environment = Environment('test1.txt', size_in_meters=Pair(30,30), tile_size_in_meters=Pair(0.5, 0.5))
    print("Creating simulation")
    sim_params = SimulationParams(num_agents=50, oceanDistribution=oceanDistribution, environment=environment)
    sim = Simulation(sim_params,mode="uniform")

    print("Running simulation")
    sim.run()