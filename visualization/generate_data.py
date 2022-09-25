import random
import numpy as np


def generate(file_name):
    with open(file_name, "w") as file:
        num_machines = 20
        for i in range(num_machines):
            machine_id = i
            cal_cost = random.randint(0, 100)
            comm_cost = random.randint(0, 100)
            file.write(str(machine_id)+' '+str(cal_cost)+' '+str(comm_cost)+'\n')

def main():
    count = 0
    while count < 10:
        generate(file_name="data/%06d.txt" % count)
        count += 1

if __name__ == '__main__':
    main()