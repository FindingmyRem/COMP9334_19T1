from sim import *


if __name__ == '__main__':
    # Test a batch of testing files.
    try:
        with open("num_tests.txt", 'r') as num_file:
            # Read the number of tests in the txt file.
            num_tests = int(num_file.read())
            # All the files are indexed from 1
            for test_index in range(1, num_tests + 1):
                # Initialize all the parameters for proper simulation
                test_obj = FileReader(test_index)
                test_obj.set_mode()
                test_obj.get_configuration()
                # call simulation function
                if test_obj.mode == 'trace':
                    whole_rec, fog_dep, net_dep, cloud_dep = simulation(test_obj.mode, test_obj.arrival_time_list,
                                                                        test_obj.service_time_list,
                                                                        test_obj.latency_list,
                                                                        test_obj.FTL, test_obj.FT2CT,
                                                                        test_obj.time_end)
                else:
                    service_para = [test_obj.alpha_1, test_obj.alpha_2, test_obj.beta]
                    network_para = [test_obj.v_1, test_obj.v_2]
                    whole_rec, fog_dep, net_dep, cloud_dep = simulation(test_obj.mode, test_obj.lamda,service_para,
                                                                        network_para, test_obj.FTL, test_obj.FT2CT,
                                                                        test_obj.time_end)
                # mrt = sum([i[1] - i[0] for i in whole_rec]) / len(whole_rec)
                mrt = mean([depart - arrival for arrival, depart in whole_rec])
                write_output(mrt, fog_dep, net_dep, cloud_dep, test_obj.file_index)
    except FileNotFoundError:
        print("There is no such file called num_tests.txt.")
