import numpy as np
from scipy import mean
from scipy.stats import sem, t
import matplotlib.pyplot as plt
import random as rdm
import math
from collections import defaultdict
import time


# This class is used to read all the given files to supply parameters the simulation needs
class FileReader:
    def __init__(self, _file_index):
        self.file_index = _file_index
        self.lamda = 0
        self.arrival_time_list = []
        self.service_time_list = []
        self.latency_list = []
        self.mode = ''
        # fogTimeLimit
        self.FTL = 0
        # fogTimeTOCloudTime
        self.FT2CT = 0
        # time_end
        self.time_end = 0
        self.alpha_1 = 0
        self.alpha_2 = 0
        self.beta = 0
        self.v_1 = 0
        self.v_2 = 0

    def set_mode(self):
        with open(f"mode_{self.file_index}.txt") as mode_file:
            self.mode = mode_file.read()
            return self.mode

    def get_configuration(self):
        # para_*.txt,arrival_*.txt,service_*.txt,network_*.txt
        if self.mode == 'random':
            # Read para_*.txt file.
            with open(f"para_{self.file_index}.txt", 'r') as para_file:
                para_list = para_file.readlines()
                self.FTL = float(para_list[0])
                self.FT2CT = float(para_list[1])
                self.time_end = float(para_list[2])
            with open(f"arrival_{self.file_index}.txt", 'r') as arrival_file:
                # The file for random mode contains the value for lambda
                self.lamda = float(arrival_file.read())
            with open(f"service_{self.file_index}.txt", 'r') as service_file:
                service_para_list = service_file.readlines()
                self.alpha_1 = float(service_para_list[0])
                self.alpha_2 = float(service_para_list[1])
                self.beta = float(service_para_list[2])
            with open(f"network_{self.file_index}.txt", 'r') as network_file:
                network_para_list = network_file.readlines()
                self.v_1 = float(network_para_list[0])
                self.v_2 = float(network_para_list[1])
        elif self.mode == 'trace':
            with open(f"para_{self.file_index}.txt", 'r') as para_file:
                para_list = para_file.readlines()
                self.FTL = float(para_list[0])
                self.FT2CT = float(para_list[1])
            with open(f"arrival_{self.file_index}.txt", 'r') as arrival_file:
                self.arrival_time_list = [float(i) for i in arrival_file]
            with open(f"service_{self.file_index}.txt", 'r') as service_file:
                self.service_time_list = [float(i) for i in service_file]
            with open(f"network_{self.file_index}.txt", 'r') as network_file:
                self.latency_list = [float(i) for i in network_file]


def write_output(_mrt, _fog_dep, _net_dep, _cloud_dep, test_index):
    # mrt_*.txt,fog_dep_*.txt,network_dep_*.txt,cloud_dep_*.txt,
    with open(f"mrt_{test_index}.txt", 'w') as mrt_file:
        mrt_file.write("{:.4f}".format(_mrt))
    with open(f"fog_dep_{test_index}.txt", 'w') as fog_file:
        for i in _fog_dep:
            fog_file.write("{:.4f}  {:.4f}\n".format(i[0], i[1]))
    with open(f"net_dep_{test_index}.txt", 'w') as net_file:
        for i in _net_dep:
            net_file.write("{:.4f}  {:.4f}\n".format(i[0], i[1]))
    with open(f"cloud_dep_{test_index}.txt", 'w') as cloud_file:
        for i in _cloud_dep:
            cloud_file.write("{:.4f}  {:.4f}\n".format(i[0], i[1]))


def min_service_time(dict_type):
    min_time = min([dict_type[i][0] for i in dict_type])
    arrival_time = 0
    for i in dict_type:
        if dict_type[i][0] == min_time:
            arrival_time = i
    return min_time, arrival_time


def update_service_time(dict_type, time_interval):
    if dict_type:
        service_time_each_request_received = time_interval / len(dict_type)
    else:
        service_time_each_request_received = 0
    for event in dict_type:
        dict_type[event][0] -= service_time_each_request_received
    return dict_type


def next_event_time_and_type(arrival, departure, last_event, dict_type, queue_type):
    # last event is the time when it arrived at fog.
    if arrival < departure and last_event not in dict_type:
        return arrival, f'arrival_{queue_type}'
    else:
        return departure, f'departure_{queue_type}'


def confidence_interval(data, confidence):
    n = len(data)
    # m = mean(data)
    m = np.mean(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)

    return m, m - h, m + h


def replication(FTL_list, rep_time, length_sim, _lamda, _service_para_list, _network_para_list, _FT2CT):
    mrt_replication = defaultdict(list)
    seed_list = list(range(1, rep_time + 1, 1))
    for _FTL in FTL_list:
        for i in range(rep_time):
            rdm.seed(seed_list[i])
            _whole_rec, _, _, _ = simulation("random", _lamda, _service_para_list, _network_para_list, _FTL, _FT2CT,
                                             length_sim)
            # steady_state_rec = _whole_rec[24999:]
            steady_state_rec = _whole_rec[15000:]

            _mrt = mean([depart - arrival for arrival, depart in steady_state_rec])
            mrt_replication[_FTL].append(_mrt)
    return mrt_replication


def simulation(mode, arrival, service, network, fogTimeLimit, fogTimeToCloudTIme, time_end):
    # [(arrival_time_at_fog, departure_time_from_system)]
    whole_record = []
    # fog={arrival_time_at_fog:service_time_remaining_in_fog, service_time_in_cloud, network_latency}
    fog = {}
    cloud_arrival = {}
    cloud = {}
    fog_depart = []
    net_depart = []
    cloud_depart = []
    # random mode
    if mode == 'random':
        # G(t) = gamma/(1-beta)(t^(1-beta) - alpha_1^(1-beta)), use inverse transform method to generate random numbers
        # with specific probability distribution.
        _alpha_1 = service[0]
        _alpha_2 = service[1]
        _beta = service[2]
        b_1 = 1 - _beta
        _v_1, _v_2 = network[0], network[1]

        next_arrival_time_fog = rdm.expovariate(arrival)
        next_depart_time_fog = math.inf
        arrival_time_fog_next_fog_depart = 0
        service_time_next_arrival = (rdm.random()*(_alpha_2 ** b_1 - _alpha_1 ** b_1) + _alpha_1 ** b_1) ** (1 / b_1)
        network_latency = rdm.uniform(_v_1, _v_2)
        # Simulate fog first
        MC_fog = 0
        while MC_fog < time_end:
            next_event_time, next_event_type = next_event_time_and_type(next_arrival_time_fog, next_depart_time_fog,
                                                                        next_arrival_time_fog, fog, 'fog')
            # Updates the service time still needed, each request in the fog received 1/(the number of requests
            # in the fog) of the service.
            time_elapsed = next_event_time - MC_fog
            fog = update_service_time(fog, time_elapsed)
            # Advance Master clock for fog
            MC_fog = next_event_time

            if next_event_type == 'arrival_fog':
                if service_time_next_arrival > fogTimeLimit:
                    cloud_service_time = (service_time_next_arrival - fogTimeLimit) * fogTimeToCloudTIme
                    fog[next_arrival_time_fog] = [fogTimeLimit, network_latency, cloud_service_time]
                else:
                    fog[next_arrival_time_fog] = [service_time_next_arrival, 0, 0]
                min_service_time_fog, arrival_time_fog_next_fog_depart = min_service_time(fog)
                next_depart_time_fog = MC_fog + min_service_time_fog * len(fog)
                # Generate new arrival event.
                next_arrival_time_fog = MC_fog + rdm.expovariate(arrival)
                service_time_next_arrival = (rdm.random() * (_alpha_2 ** b_1 - _alpha_1 ** b_1)
                                             + _alpha_1 ** b_1) ** (1 / b_1)
            elif next_event_type == 'departure_fog':
                fog_depart.append((arrival_time_fog_next_fog_depart, MC_fog))
                # the service time in fog unit of the request is more than that fog can offer, the remaining work has
                # to be served by the cloud
                depart_time = MC_fog + fog[arrival_time_fog_next_fog_depart][1]
                if fog[arrival_time_fog_next_fog_depart][2] > 0:
                    net_depart.append((arrival_time_fog_next_fog_depart, depart_time))
                    cloud_arrival[arrival_time_fog_next_fog_depart] = fog[arrival_time_fog_next_fog_depart][2]
                # the request leaves the fog permanently
                else:
                    whole_record.append((arrival_time_fog_next_fog_depart, MC_fog))
                fog.pop(arrival_time_fog_next_fog_depart)
                if fog:
                    min_service_time_fog, arrival_time_fog_next_fog_depart = min_service_time(fog)
                    next_depart_time_fog = MC_fog + min_service_time_fog * len(fog)
                else:
                    arrival_time_fog_next_fog_depart = 0
                    next_depart_time_fog = math.inf

        # simulate cloud
        event = 0
        # network_depart = [arrival at the fog, depart from the network(arrival at cloud)]
        net_depart = sorted(net_depart, key=lambda x: x[1])
        next_arrival_time_cloud = net_depart[event][1]
        arrival_time_fog_next_cloud_depart = net_depart[event][0]
        service_time_next_arrival = cloud_arrival[arrival_time_fog_next_cloud_depart]
        next_depart_time_cloud = math.inf
        MC_cloud = 0

        while MC_cloud < time_end:
            next_event_time, next_event_type = next_event_time_and_type(next_arrival_time_cloud,
                                                                        next_depart_time_cloud, net_depart[-1][0],
                                                                        cloud, 'cloud')
            # Updates the service time still needed in cloud.
            time_elapsed = next_event_time - MC_cloud
            cloud = update_service_time(cloud, time_elapsed)
            # Advance Master clock for cloud
            MC_cloud = next_event_time
            if next_event_type == 'arrival_cloud':
                cloud[net_depart[event][0]] = [service_time_next_arrival]
                min_service_time_cloud, arrival_time_fog_next_cloud_depart = min_service_time(cloud)
                next_depart_time_cloud = MC_cloud + min_service_time_cloud * len(cloud)
                if event != len(net_depart) - 1:
                    event += 1
                    next_arrival_time_cloud = net_depart[event][1]
                    service_time_next_arrival = cloud_arrival[net_depart[event][0]]
            elif next_event_type == 'departure_cloud':
                cloud_depart.append((arrival_time_fog_next_cloud_depart, MC_cloud))
                cloud.pop(arrival_time_fog_next_cloud_depart)
                whole_record.append((arrival_time_fog_next_cloud_depart, MC_cloud))
                if arrival_time_fog_next_cloud_depart == net_depart[-1][0]:
                    break
                # there are requests in the cloud still waiting to be served.
                if cloud:
                    min_service_time_cloud, arrival_time_fog_next_cloud_depart = min_service_time(cloud)
                    next_depart_time_cloud = MC_cloud + min_service_time_cloud * len(cloud)
                # the cloud is empty for now and waiting for new arriving requests
                else:
                    arrival_time_fog_next_cloud_depart = net_depart[event][0]
                    next_depart_time_cloud = math.inf

    # trace mode
    elif mode == 'trace':
        # MC:master_clock
        MC_fog = 0
        # arrival is a list of the time at which each arrival occurs.
        # service is a list of the service time each arrival needs.
        # fog = {arrival_time:[service_remaining_in_fog, network_latency, service_time_in_cloud],...}
        next_depart_time_fog = math.inf
        event = 0
        # Simulate fog, if no arrival at fog, then return.
        arrival_time_fog_next_fog_depart = 0
        next_arrival_time_fog = arrival[event]
        service_time_next_arrival = service[event]
        while True:
            # If the last arrival is already in the fog, then no arrival will occur
            next_event_time, next_event_type = next_event_time_and_type(next_arrival_time_fog, next_depart_time_fog,
                                                                        arrival[-1], fog, 'fog')
            # The request in the PS queue each get 1/n of the service, n is the number of requests in the queue.
            time_elapsed = next_event_time - MC_fog
            fog = update_service_time(fog, time_elapsed)
            # Advance the MC to the next consecutive event
            MC_fog = next_event_time
            if next_event_type == 'arrival_fog':
                # Add arrival into fog
                if service_time_next_arrival > fogTimeLimit:
                    service_time_in_cloud = (service_time_next_arrival - fogTimeLimit)*fogTimeToCloudTIme
                    fog[next_arrival_time_fog] = [fogTimeLimit, network[event], service_time_in_cloud]
                else:
                    fog[next_arrival_time_fog] = [service_time_next_arrival, 0, 0]
                # Determine the next departure and the arrival time of it.
                min_service_time_fog, arrival_time_fog_next_fog_depart = min_service_time(fog)
                next_depart_time_fog = MC_fog + min_service_time_fog * len(fog)
                # Find the next arrival and its service time in the fog.
                if event != len(arrival) - 1:
                    event += 1
                    next_arrival_time_fog = arrival[event]
                    service_time_next_arrival = service[event]
            elif next_event_type == 'departure_fog':
                fog_depart.append((arrival_time_fog_next_fog_depart, MC_fog))
                depart_time = MC_fog + fog[arrival_time_fog_next_fog_depart][1]
                if fog[arrival_time_fog_next_fog_depart][2] > 0:
                    net_depart.append((arrival_time_fog_next_fog_depart, depart_time))
                    cloud_arrival[arrival_time_fog_next_fog_depart] = fog[arrival_time_fog_next_fog_depart][2]
                else:
                    whole_record.append((arrival_time_fog_next_fog_depart, MC_fog))
                fog.pop(arrival_time_fog_next_fog_depart)
                # If the last arrival departs, then end simulation for fog.
                if arrival_time_fog_next_fog_depart == arrival[-1]:
                    break
                if fog:
                    min_service_time_fog, arrival_time_fog_next_fog_depart = min_service_time(fog)
                    next_depart_time_fog = MC_fog + min_service_time_fog * len(fog)
                else:
                    arrival_time_fog_next_fog_depart = 0
                    next_depart_time_fog = math.inf
        # Simulate cloud.
        event = 0
        net_depart = sorted(net_depart, key=lambda x: x[1])
        # print(network_depart)
        next_arrival_time_cloud = net_depart[event][1]
        arrival_time_fog_next_cloud_depart = net_depart[event][0]
        service_time_next_arrival = cloud_arrival[arrival_time_fog_next_cloud_depart]
        next_depart_time_cloud = math.inf
        MC_cloud = 0
        # print(cloud_arrival)
        while True:
            next_event_time, next_event_type = next_event_time_and_type(next_arrival_time_cloud,
                                                                        next_depart_time_cloud,
                                                                        net_depart[-1][0], cloud,
                                                                        'cloud')
            # Updates the service time still needed in cloud.
            time_elapsed = next_event_time - MC_cloud
            cloud = update_service_time(cloud, time_elapsed)
            MC_cloud = next_event_time
            if next_event_type == 'arrival_cloud':
                cloud[net_depart[event][0]] = [service_time_next_arrival]
                min_service_time_cloud, arrival_time_fog_next_cloud_depart = min_service_time(cloud)
                next_depart_time_cloud = MC_cloud + min_service_time_cloud * len(cloud)
                if event != len(net_depart) - 1:
                    event += 1
                    next_arrival_time_cloud = net_depart[event][1]
                    service_time_next_arrival = cloud_arrival[net_depart[event][0]]
            elif next_event_type == 'departure_cloud':
                cloud_depart.append((arrival_time_fog_next_cloud_depart, MC_cloud))
                cloud.pop(arrival_time_fog_next_cloud_depart)
                whole_record.append((arrival_time_fog_next_cloud_depart, MC_cloud))
                if arrival_time_fog_next_cloud_depart == net_depart[-1][0]:
                    break
                if cloud:
                    min_service_time_cloud, arrival_time_fog_next_cloud_depart = min_service_time(cloud)
                    next_depart_time_cloud = MC_cloud + min_service_time_cloud * len(cloud)
                else:
                    arrival_time_fog_next_cloud_depart = net_depart[event][0]
                    next_depart_time_cloud = math.inf
    # sort in the order of the arrival time at fog
    fog_depart.sort()
    net_depart.sort()
    cloud_depart.sort()
    whole_record.sort()
    return whole_record, fog_depart, net_depart, cloud_depart


if __name__ == "__main__":
    # This part is to test reproducibility
    file_index = int(input("Input the file number of which you want to test: "))
    seed_for_repro = int(input("Input a integer as seed for random number generator: "))
    if file_index:
        test_object = FileReader(file_index)
        try:
            test_object.set_mode()
            test_object.get_configuration()
            service_para = [test_object.alpha_1, test_object.alpha_2, test_object.beta]
            network_para = [test_object.v_1, test_object.v_2]
            # (5, 10), (6, 20), (7, 40)
            rdm.seed(seed_for_repro)
            whole_rec, f_dep, n_dep, cloud_dep = simulation(test_object.mode, test_object.lamda, service_para,
                                                            network_para,
                                                            test_object.FTL, test_object.FT2CT, test_object.time_end)
            mrt = mean([depart - arrival for arrival, depart in whole_rec])
            write_output(mrt, f_dep, n_dep, cloud_dep, file_index)
        except FileNotFoundError:
            print("There is no such file.")

    design_flag = int(input("Input '1' to enable design simulation, '0' otherwise: "))
    # This part is used to determine a suitable value of fogTimeLimit
    if design_flag:
        lamda = 9.72
        alpha_1 = 0.01
        alpha_2 = 0.4
        beta = 0.86
        v_1 = 1.2
        v_2 = 1.47
        # fogTimeToCloudTime
        FT2CT = 0.6
        service_para_list = [alpha_1, alpha_2, beta]
        network_para_list = [v_1, v_2]
        number_of_rep = int(input("Input the number of replications you want to run(positive integer): "))

        # this part is to determine a narrower range of fogTimeLimit
        # range_of_FTL = list(np.arange(0.01, 0.4, 0.01))
        # for FTL in range_of_FTL:
        #     running_mrt_rec = list()
        #     mrt_sum = 0
        #     rdm.seed(rdm.random())
        #     whole_rec, f_dep, n_dep, cloud_dep = simulation("random", lamda, service_para_list, network_para_list,
        #                                                     FTL, FT2CT, 5000)
        #     for rec in range(len(whole_rec)):
        #         mrt_sum += whole_rec[rec][1] - whole_rec[rec][0]
        #         running_mrt = mrt_sum / (rec + 1)
        #         running_mrt_rec.append(running_mrt)
        #     x = range(1, len(whole_rec) + 1)
        #     plt.plot(x, running_mrt_rec)
        #     plt.ylabel('Running mean response time')
        #     plt.title(f'fogTimeLimit = {FTL}')
        #     plt.show()

        # narrow_FTL_range = [0.08, 0.09, 0.1, 0.11, 0.12, 0.13]
        # steady state mrt
        # start_time = time.time()
        # steady_mrt = replication(narrow_FTL_range, number_of_rep, 10000, lamda, service_para_list, network_para_list,
        #                          FT2CT)
        # for i in steady_mrt:
        #     # print(steady_mrt[i])
        #     m, l, h = confidence_interval(steady_mrt[i], 0.95)
        #     print("when the fogTimeLimit = {} and n = {},there is a 95% chance "
        #           "that mrt lies in [{:.4f},{:.4f}]".format(i, number_of_rep, l, h))

        # This part is to compare two different systems
        # narrow_FTL_range_new = [0.1, 0.11]
        # mrt_2_sys = replication(narrow_FTL_range_new, number_of_rep, 5000, lamda, service_para_list, network_para_list,
        #                         FT2CT)
        # diff_mrt = [mrt_2_sys[0.1][i] - mrt_2_sys[0.11][i] for i in range(number_of_rep)]
        # m, l, h = confidence_interval(diff_mrt, 0.95)
        # print("the 95% confidence interval of the difference of mrt of 2 systems is [{:.4f}, {:.4f}]".format(l, h))
        # end_time = time.time()
        # print(end_time - start_time, 's')

        # This part is for visual inspection to determine transient state
        # for FTL in narrow_FTL_range:
        #     for s in range(2):
        #         mrt_sum = 0
        #         running_mrt_rec = list()
        #         rdm.seed(rdm.random())
        #         whole_rec, f_dep, n_dep, cloud_dep = simulation("random", lamda, service_para_list, network_para_list,
        #                                                         FTL, FT2CT, 10000)
        #         for rec in range(len(whole_rec)):
        #             mrt_sum += whole_rec[rec][1] - whole_rec[rec][0]
        #             running_mrt = mrt_sum / (rec + 1)
        #             running_mrt_rec.append(running_mrt)
        #
        #         plt.plot(running_mrt_rec)
        #         plt.ylabel('Running mean response time')
        #         plt.title(f'fogTimeLimit = {FTL}')
        #         plt.show()

        # This part is for verification of service time
        # seed_list = list(range(1, 300, 50))
        # k = 0
        # for i in range(6):
        #     rdm.seed(seed_list[k])
        #     s = [(rdm.random()*(alpha_2**(1-beta) - alpha_1**(1-beta)) + alpha_1**(1-beta))**(1/(1-beta))
        #          for j in range(1000)]
        #     plt.plot(s, 'o')
        #     plt.show()
        #     k += 1


