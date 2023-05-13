import matplotlib as plt


from training_sim import Simulating
from trafficgen import TrafficGenerator
from memory import Memory
from model2train import TrainingModel
from Model_Handling import set_train_path, set_sumo

path = set_train_path("model")

sumoBinary = "/path/to/sumo-gui"
config_cmd = set_sumo(False, "configuration.sumocfg", 200)

Model = TrainingModel(3, 50, 0.2, 60, 3, 200)

Memory = Memory(30000, 600)

Sim_Traffic = TrafficGenerator(200, 150)

Simulation = Simulating(Model, Memory, Sim_Traffic, config_cmd, 0.7, 200, 12, 5, 60, 3, 40)

episode = 0

while episode < 40:
    print(' Episode ', str(episode+1), 'of', str(40))
    epsilon = 1.0 - (episode / 200)  # set the epsilon for this episode according to epsilon-greedy policy
    simulation_time, training_time = Simulation.one_ep(episode, epsilon)  # run the simulation
    print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time+training_time, 1), 's')
    episode += 1

Model.save_model(path)

plt.plot(episode, Simulating.average_reward, label="average rewards")
plt.plot(episode, Simulating.cumulative_wait_store, label="total wait time")
plt.plot(episode, Simulating.avg_queue_length_store, label="avg queue length")
plt.legend(loc=4)
plt.show()
