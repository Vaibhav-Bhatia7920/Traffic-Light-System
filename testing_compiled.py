import tensorflow as tf
from testing_sim import Simulating
from trafficgen import TrafficGenerator
from Model_Handling import model_test_path, set_sumo

model_n = 3
req_ep = 39
path = model_test_path("model", model_n, req_ep)

sumoBinary = "/path/to/sumo-gui"
config_cmd = set_sumo(True, "configuration.sumocfg", 5000)

Model = tf.keras.models.load_model(path)
Sim_Traffic = TrafficGenerator(5000, 700)

Simulation = Simulating(Model, Sim_Traffic, config_cmd, 5000, 20, 6, 80, 2)

simulation_time = Simulation.one_ep()
print('Simulation time:', simulation_time, ' Reward:', Simulating.reward)

