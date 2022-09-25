import os
import random
import numpy as np
import matplotlib.pyplot as plt

# 每一台机器
    #当前的计算耗时
    #当前的通信耗时
    #当前的内存
    #当前的其他变量统计


class Visualization:
    def __init__(self, data_path="data", num_machines=20, pause_time=0.5):
        self.pause_time =pause_time
        self.data_path = data_path
        self.count = 0
        self.num_machines = num_machines
        self.machines_id = np.zeros(self.num_machines)
        self.machines_cal_cost = np.zeros(self.num_machines)
        self.machines_comm_cost = np.zeros(self.num_machines)

    def get_file_path(self):
        current_file_name = "%06d.txt" % self.count
        return os.path.join(self.data_path, current_file_name)

    def get_current_state(self, file_name):
        with open(file_name, "r") as file:
            for i in range(self.num_machines):
                machine_id, cal_cost, comm_cost = map(int, file.readline().split(' '))
                self.machines_cal_cost[machine_id] = cal_cost
                self.machines_comm_cost[machine_id] = comm_cost
                self.machines_id[machine_id] = machine_id

    def vis(self, file_name):
        self.get_current_state(file_name)
        plt.clf()
        high = max(self.machines_cal_cost) * 1.1
        filling = high - self.machines_cal_cost
        plt.bar(self.machines_id, self.machines_cal_cost)
        plt.bar(self.machines_id, filling, bottom=self.machines_cal_cost, color="#FFFFFF")
        plt.bar(self.machines_id, self.machines_comm_cost, bottom=filling + self.machines_cal_cost)
        plt.pause(self.pause_time)
        plt.ioff()

    def play(self):
        while os.path.isfile(self.get_file_path()):
            self.vis(self.get_file_path())
            self.count += 1

if __name__ == '__main__':
    Visual = Visualization()
    Visual.play()
