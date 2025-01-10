from simulation import Simulation, SimulationParams
from environment import Environment
from helper_classes import OceanDistribution, Pair


if __name__ == '__main__':
    oceanDistribution = OceanDistribution(Pair(0.5, 0.1), Pair(0.5, 0.1), Pair(0.5, 0.1), Pair(0.5, 0.1), Pair(0.5, 0.1))
    print("Creating environment")
    environment = Environment('test2.txt', size_in_meters=Pair(30,30), tile_size_in_meters=Pair(1, 1), with_obstacles=True)
    print("Creating simulation")
    sim_params = SimulationParams(num_agents=20, oceanDistribution=oceanDistribution, environment=environment, create_gif=True, simulation_time_in_seconds=100)
    sim = Simulation(sim_params,mode="multimodal")

    print("Running simulation")
    # sim.run(clustering_mode="hierarchical_clustering")
    sim.run(clustering_mode="default")