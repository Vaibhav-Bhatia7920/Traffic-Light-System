import traci
import numpy as np
import random
import timeit
import os
from simu_props import Props

Gp_1 = 0
Yp_1 = 1
Gp_2 = 2
Yp_2 = 3
Gp_3 = 4
Yp_3 = 5


class Simulating:
    def __init__(self, model, memory, traffic, sumo_cmd, gamma, max_steps, green_duration, yellow_duration,
                 num_states, num_actions, training_epochs):
        self._Model = model
        self._Memory = memory
        self._TrafficGen = traffic
        self._gamma = gamma
        self._no_of_step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._training_epochs = training_epochs
        self._waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0

    def one_ep(self, episode, epsilon):

        start_time = timeit.default_timer()

        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        self._no_of_step = 0
        self._waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        old_total_wait = 0
        old_state = -1
        old_action = -1

        while self._no_of_step < self._max_steps:
            present_state = Props._get_state()
            present_waiting_time = self._get_total_waitingtime()
        reward = old_total_wait - present_waiting_time

        if self._no_of_step != 0:
            self._Memory.new_sample((old_state, old_action, reward, present_state))

        action = self._choose_action(present_state, epsilon)

        if self._no_of_step != 0 & action != old_action:
            self.set_yellowstate(old_action)
            self.simulate(self._yellow_duration)

        self.set_greenstate(action)
        self.simulate(self._green_duration)

        # For next episode

        old_state = present_state
        old_action = action
        old_total_wait = present_waiting_time

        if reward < 0:
            self._total_neg_reward += reward  # total reward of an episode

        self.episode_stats()
        print("Total reward:", self._total_neg_reward, "- Epsilon:", round(epsilon, 2))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print("Training...")
        start_time = timeit.default_timer()
        for _ in range(self._training_epochs):
            self._replay()
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time

    def simulate(self, steps):
        if (self._no_of_step + steps) >= self._max_steps:
            steps = self._max_steps - self._no_of_step

        while steps > 0:
            traci.simulationStep()
            self._no_of_step += 1
            steps += -1
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length

    def _choose_action(self, epsilon, state):
        if random.random() < epsilon:
            return random.randint(0, 3)
        else:
            return np.argmax(self._Model.predict_one(state))

    def set_yellowstate(self, old_action):
        state = 2*old_action + 1
        traci.trafficlight.setPhase("n4", state)

    def set_greenstate(self,action):
        traci.trafficlight.setPhase("n4", 2*action)


    def _replay(self):

        batch = self._Memory.get_samples(self._Model.batch_size)

        if len(batch) > 0:  # if the memory is full enough
            states = np.array([val[0] for val in batch])  # extract states from the batch
            next_states = np.array([val[3] for val in batch])  # extract next states from the batch

            # prediction
            q_s_a = self._Model.predict_batch(states)  # predict Q(state), for every sample
            q_s_a_d = self._Model.predict_batch(next_states)  # predict Q(next_state), for every sample

            # setup training arrays
            x = np.zeros((len(batch), self._num_states))
            y = np.zeros((len(batch), self._num_actions))

            for i, b in enumerate(batch):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value

            self._Model.train_batch(x, y)  # train the NN

    def _get_total_waitingtime(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["1to4", "2to4", "3to4"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times:  # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id]
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time

    def episode_stats(self):

        self._reward_store.append(self._total_neg_reward)
        self._average_reward.append(sum(self._reward_store[-10:])/10)
        self._cumulative_wait_store.append(
            self._sum_waiting_time)  # total number of seconds waited by cars in this episode
        self._avg_queue_length_store.append(
            self._sum_queue_length / self._max_steps)  # average number of queued cars per step, in this episode

    def _get_queue_length(self):
        halt_n = traci.edge.getLastStepHaltingNumber("n2ton4")
        halt_w = traci.edge.getLastStepHaltingNumber("n1ton4")
        halt_e = traci.edge.getLastStepHaltingNumber("n3ton4")
        queue_length = halt_n + halt_e + halt_w
        return queue_length

    def _get_state(self):

        state = np.zeros(self._num_states)
        car_list = traci.vehicle.getIDList()

        for car in car_list:
            lane_pos = traci.vehicle.getLanePosition(car)
            lane_id = traci.vehicle.getLaneID(car)
            lane_pos = 750 - lane_pos

            if lane_pos < 9:
                lane_cell = 0
            elif lane_pos < 18:
                lane_cell = 1
            elif lane_pos < 27:
                lane_cell = 2
            elif lane_pos < 36:
                lane_cell = 3
            elif lane_pos < 45:
                lane_cell = 4
            elif lane_pos < 54:
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
            else:
                lane_group = -1

            if 5 >= lane_group >= 1:
                car_position = int(str(lane_group) + str(
                    lane_cell))
                valid_car = True
            elif lane_group == 0:
                car_position = lane_cell
                valid_car = True
            else:
                valid_car = False

            if valid_car:
                state[car_position] = 1

        return state
    @property
    def reward_store(self):
        return self._reward_store

    @property
    def average_reward(self):
        return self._average_reward

    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store

    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store






