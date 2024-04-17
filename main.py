import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils

num_exp = 5

if __name__ == "__main__":

    for ind in range(num_exp):
        # sometimes u need disgusting code for disgusting results
        try:
            os.mkdir("./experiment"+str(ind))
        except:
            ""
        try:
            os.mkdir("./experiment" + str(ind) +"/x")
        except:
            ""
        try:
            os.mkdir("./experiment" + str(ind) +"/y")
        except:
            ""
        try:
            os.mkdir("./experiment" + str(ind) +"/z")
        except:
            ""
        try:
            os.mkdir("./experiment" + str(ind) +"/abs")
        except:
            ""

    testing = []
    for n in range(num_exp - 2):
        testing.append([])

    # ok we are hardcoding variables, which is wrong, but whatever.
    # UPDATE: WE ARE NO LONGER HARDCODING VARIABLES. YIPPEEE!!

    # each training[0..n] is a training set
    training = []
    

    for bandwidth_pct in range(100):
        bd_testing, bd_training = utils.get_training_and_testing_for_bandwidth(bandwidth_pct, num_exp)
        for n in range(num_exp - 2):
            testing[n].append(bd_testing[n])
        training.append(bd_training)

    for test_ind, test_set in enumerate(testing):
        with open("test_"+str(test_ind)+"_likely.txt","w+") as output:
            for ind, test_entry in enumerate(test_set):
                distances = []
                for train_entry in training:
                    # sqrt ( train - test)
                    distances.append(float((((train_entry[0] - test_entry[0])**2).sum()**0.5).item()))
                distances = pd.Series(distances)
                mins = []
                d_max = distances.max()
                for n in range(3): # find 3 bottom mins
                    min_ind = distances.idxmin()
                    mins.append(min_ind)

                    if min_ind < len(distances): # bro it is returning an index that is not a valid index. ?????
                        distances[min_ind] = d_max
                output.write(str(ind) + "\t" + str(mins) + "\n")
