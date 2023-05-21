import traci
import numpy as np
import timeit

Gp_1 = 0
Yp_1 = 1
Gp_2 = 2
Yp_2 = 3


class Simulating:
    def __init__(self, model, traffic, sumo_cmd, max_steps, green_duration, yellow_duration,
                 num_states, num_actions):
        self.Model = model
        self.TrafficGen = traffic
        self.no_of_step = 0
        self.sumo_cmd_line = sumo_cmd
        self.max_steps = max_steps
        self.green_duration = green_duration
        self.yellow_duration = yellow_duration
        self.no_of_states = num_states
        self.no_of_actions = num_actions
        self.waiting_times = {}
        self.total_reward = 0
        self.sum_waiting_time = 0
        self.input_dim = num_states

    def one_ep(self):

        start_time = timeit.default_timer()

        self.TrafficGen.generate_routefile(seed = 1200)
        traci.start(self.sumo_cmd_line)
        print("Simulating...")

        self.no_of_step = 0
        self.waiting_times = {}
        self.total_reward = 0
        old_total_wait = 0
        old_action = -1

        while self.no_of_step < self.max_steps:
            present_state = self.get_state()
            present_waiting_time = self.total_waiting_time()
            reward = old_total_wait - present_waiting_time

            action = self.action_selection(present_state)

            if self.no_of_step != 0 & action != old_action:
                self.set_yellow_state(old_action)
                self.simulate(self.yellow_duration)

            self.set_green_state(action)
            self.simulate(self.green_duration)

            old_action = action
            old_total_wait = present_waiting_time
            print(reward)
            self.total_reward += reward

        print("Total reward:", self.total_reward)
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time

    def simulate(self, steps):
        if (self.no_of_step + steps) >= self.max_steps:
            steps = self.max_steps - self.no_of_step

        while steps > 0:
            traci.simulationStep()
            self.no_of_step += 1
            steps -= 1
            queue_length = self.queue_length()
            self.sum_waiting_time += queue_length

    def action_selection(self, state):
        state = np.reshape(state, [1, self.input_dim])
        return np.argmax(self.Model.predict(state))

    def set_yellow_state(self, old_action):
        print("YEllow")
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
                self.waiting_times[car_id] = wait_time
            else:
                if car_id in self.waiting_times:
                    del self.waiting_times[car_id]
        total_waiting_time = sum(self.waiting_times.values())
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

    @property
    def reward(self):
        return self.total_reward
