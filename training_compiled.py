import matplotlib.pyplot as plt


from training_sim import Simulating
from trafficgen import TrafficGenerator
from memory import Memory
from Loop_Modelling import Modelling
from Model_Handling import model_train_path, set_sumo

path = model_train_path("model")

sumoBinary = "/path/to/sumo-gui"
config_cmd = set_sumo(False, "configuration.sumocfg", 5000)

Memory = Memory(30000, 100)
Model = Modelling(80, 2, 0.001, 6, 100, 64, Memory, 0.7)
Sim_Traffic = TrafficGenerator(5000, 600)

Simulation = Simulating(Model, Memory, Sim_Traffic, config_cmd, 0.7, 5000, 20, 6, 80, 2, 300)

episode = 0
total_episodes = 30
while episode < total_episodes:
    print(' Episode ', str(episode+1), 'of', str(total_episodes))
    epsilon = 1.0
    if episode > 1:
        epsilon = 1.0 - (episode / 30)**2  # set the epsilon for this episode according to epsilon-greedy policy
    simulation_time, training_time = Simulation.one_ep(episode, epsilon)
    if (episode+1) % 10 == 0:
        Model.save_model(path, episode)
    print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time + training_time, 1), 's')
    episode += 1


# plt.plot(episode, Simulating.__getattribute__(_), label="average rewards")
# plt.plot(episode, Simulating.cumulative_wait_store, label="total wait time")
# plt.plot(episode, Simulating.avg_queue_length_store, label="avg queue length")
# plt.legend(loc=4)
# plt.show()
