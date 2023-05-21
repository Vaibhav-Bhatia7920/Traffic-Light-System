import traci
import numpy as np
import random
import timeit


Gp_1 = 0
Yp_1 = 1
Gp_2 = 2
Yp_2 = 3


class Simulating:
    def __init__(self, model, memory, traffic, sumo_cmd, gamma, max_steps, green_duration, yellow_duration,
                 num_states, num_actions, training_epochs):
        self.Model = model
        self.Memory = memory
        self.TrafficGen = traffic
        self.discount_rate = gamma
        self.no_of_step = 0
        self.sumo_cmd_line = sumo_cmd
        self.max_steps = max_steps
        self.green_duration = green_duration
        self.yellow_duration = yellow_duration
        self.no_of_states = num_states
        self.no_of_actions = num_actions
        self.no_of_train_epochs = training_epochs
        self.waiting_time = {}
        self.total_neg_reward = 0
        self.sum_queue_length = 0
        self.sum_waiting_time = 0

    def one_ep(self, episode, epsilon):

        start_time = timeit.default_timer()

        self.TrafficGen.generate_routefile(seed=episode)
        traci.start(self.sumo_cmd_line)
        print("Simulating...")

        self.no_of_step = 0
        self.waiting_time = {}
        self.total_neg_reward = 0
        self.sum_queue_length = 0
        self.sum_waiting_time = 0
        old_total_wait = 0
        old_state = -1
        old_action = -1

        while self.no_of_step < self.max_steps:
            present_state = self.get_state()
            present_waiting_time = self.total_waiting_time()
            reward = old_total_wait - present_waiting_time

            if self.no_of_step != 0:
                self.Memory.add_sample((old_state, old_action, reward, present_state))

            action = self.action_selection(epsilon, present_state)

            if self.no_of_step != 0 & action != old_action:
                self.set_yellow_state(old_action)
                self.simulate(self.yellow_duration)

            self.set_green_state(action)
            self.simulate(self.green_duration)

            old_state = present_state
            old_action = action
            old_total_wait = present_waiting_time

            if reward < 0:
                self.total_neg_reward += reward

        print("Total reward:", self.total_neg_reward, "- Epsilon:", round(epsilon, 2))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print("Training...")
        start_time = timeit.default_timer()

        for a in range(self.no_of_train_epochs):
            self.Model.loop()
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time

    def simulate(self, steps):
        if (self.no_of_step + steps) >= self.max_steps:
            steps = self.max_steps - self.no_of_step

        while steps > 0:
            traci.simulationStep()
            self.no_of_step += 1
            steps -= 1
            queue_length = self.queue_length()
            self.sum_queue_length += queue_length
            self.sum_waiting_time += queue_length

    def action_selection(self, epsilon, state):
        if random.random() < epsilon:
            print("random")
            return random.randint(0, 1)
        else:
            print("max")
            return np.argmax(self.Model.predict_one(state))

    def set_yellow_state(self, old_action):
        print("Yellow")
        state = 2*old_action + 1
        traci.trafficlight.setPhase("n4", state)

    def set_green_state(self, action):
        print("Green")
        traci.trafficlight.setPhase("n4", 2*action)

    def total_waiting_time(self):
        incoming_roads = ["1to4", "2to4", "3to4", "5to4"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id) 
            if road_id in incoming_roads:  
                self.waiting_time[car_id] = wait_time
            else:
                if car_id in self.waiting_time:  
                    del self.waiting_time[car_id]
        total_waiting_time = sum(self.waiting_time.values())
        return total_waiting_time


    def queue_length(self):
        n_wait = traci.edge.getLastStepHaltingNumber("2to4")
        w_wait = traci.edge.getLastStepHaltingNumber("1to4")
        e_wait = traci.edge.getLastStepHaltingNumber("3to4")
        s_wait = traci.edge.getLastStepHaltingNumber("5to4")
        queue_length = n_wait + e_wait + w_wait + s_wait
        return queue_length

    def get_state(self):
        state = np.zeros(self.no_of_states)
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = 750 - lane_pos

            if lane_pos < 7:
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 40:
                lane_cell = 4
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 100:
                lane_cell = 6
            elif lane_pos < 160:
                lane_cell = 7
            elif lane_pos < 400:
                lane_cell = 8
            elif lane_pos <= 750:
                lane_cell = 9

            if lane_id == "1to4_0":
                lane_group = 0
            elif lane_id == "1to4_1":
                lane_group = 1
            elif lane_id == "2to4_0":
                lane_group = 2
            elif lane_id == "2to4_1":
                lane_group = 3
            elif lane_id == "3to4_0":
                lane_group = 4
            elif lane_id == "3to4_1":
                lane_group = 5
            elif lane_id == "5to4_0":
                lane_group = 6
            elif lane_id == "5to4_1":
                lane_group = 7
            else:
                lane_group = -1

            if 7 >= lane_group >= 1:
                car_position = int(str(lane_group) + str(lane_cell))
                valid_car = True
            elif lane_group == 0:
                car_position = lane_cell
                valid_car = True
            else:
                valid_car = False

            if valid_car:
                state[car_position] = 1

        return state







