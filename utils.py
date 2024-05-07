import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

############################################################
### MAIN PARSE

def read_and_split_file(file_in: str):
    print("file in", file_in)
    input_csv = np.genfromtxt("./csv/"+file_in, delimiter=",", skip_header=1).T
    in_x_axis = input_csv[0]

    # in_y_axis = input_csv[1]
    # in_y_axis = input_csv[2]
    # in_y_axis = input_csv[3]
    in_y_axis = input_csv[4]

    # plt.plot(in_x_axis, in_y_axis, linewidth=2, linestyle="-", c="b")
    # plt.show()

    # [[x1, x2, x3]
    #  [y1, y2, y2]]
    arr_in = np.vstack((in_x_axis, in_y_axis)).T

    # [[x1, y1]
    #  [x2, y2]
    #  [x3, y3]]

    # slice arry_in into proper windows

    arr_in = arr_in[300:]
    sliced_arr = slice_arr(arr_in)
    # [ [[x1, y1]
    #    [x2, y2]],
    #   [[x3, y3]
    #    [x4, y4]]  ]

    return sliced_arr

def filter_data(sliced_arr: np.ndarray):
    

    ma_out_x = np.array([])
    ma_out_y = np.array([])

    filtered_out_x = np.array([])
    filtered_out_y = np.array([])

    combined_out_y = np.array([])
    for arr in sliced_arr:
        window_size = 25
        iter_arr = trim_arr_mod(arr, window_size)
        iter_arr_transpose = iter_arr.T

        # print(iter_arr_transpose.shape)
        ma_out_x = np.append(ma_out_x, moving_avg(arr_in=iter_arr_transpose[0], window=window_size))
        
        y_ma = moving_avg(arr_in=iter_arr_transpose[1], window=window_size)
        ma_out_y = np.append(ma_out_y, y_ma)

        filtered_out_x = np.append(filtered_out_x, iter_arr_transpose[0])
        filtered_out_y = np.append(filtered_out_y, utils_sav_filter(arr_in=iter_arr_transpose[1], window=window_size))

        combined_out_y = np.append(combined_out_y, utils_sav_filter(arr_in=y_ma, window=5))
    
    # ma filter trials
    # plt.plot(ma_out_x, ma_out_y, linewidth=2, linestyle="-", c="b")
    # plt.show()

    # filtered trials
    # plt.plot(filtered_out_x, filtered_out_y, linewidth=2, linestyle="-", c="b")
    # plt.show()

    # plt.plot(ma_out_x, combined_out_y, linewidth=2, linestyle="-", c="b")
    # plt.show()


############################################################
### UTILS

def euclidian_distance():
    return

def pad_arr(arr_in: np.ndarray[np.float64], window: int = 20):
    return np.pad(
            arr_in,
            (0, window - arr_in.size%window),
            mode = 'constant',
            constant_values = np.NaN
        )

def trim_arr_mod(arr_in: np.ndarray[np.float64], window: int = 20):
    return arr_in[:arr_in.size-(arr_in.size%window)]

############################################################
### FILTERS

def utils_sav_filter(arr_in: np.ndarray[np.float64], window = 100):
    return savgol_filter(arr_in, window, 2) # quadratic polynomial is very generous for fitting a linear 

def moving_avg(arr_in: np.ndarray[np.float64], window: int = 20):
    arr = np.mean(arr_in.reshape(-1, window), axis=1)
    return arr

############################################################
### SLICE

def slice_arr(
        arr_in: np.ndarray[(np.float64, np.float64)],

        entry_len: int = 10,
        sleep_duration = 3,

        head_pad = 1,
        tail_pad = 1,
        granularity= 100,
        trials = 100
    ):
    '''
    returns [100] [n][x,y] where each window corresponds to a bandwidth
    '''

    trial_total_len = (entry_len + sleep_duration) * granularity # granularity is 100Hz
    head_slice = entry_len * granularity
    # sleep_duration = sleep_duration * granularity # granularity is 100Hz
    head_pad = head_pad * granularity # granularity is 100Hz
    tail_pad = tail_pad * granularity # granularity is 100Hz

    head_width = np.int_(head_slice-head_pad-tail_pad)

    arr_in = arr_in[:trials * trial_total_len]
    ret_arr = np.ndarray(shape=(100,head_width,2))
    for i in range(0,trials):
        base_offset = i*trial_total_len
        arr_slice = arr_in[(base_offset + head_pad) : np.int_(base_offset + (head_slice - tail_pad))]
        # print("arr slice", arr_slice)
        try:
            ret_arr[i] = arr_slice
        except:
            print("trial {} error", i)
    
    # print("returned nd arr: ", ret_arr)

    return ret_arr
    
############################################################
### REGRESSION UTILS

def slice_data(data_in, num_slices, window_len, bandwidth_num, col_name):
    data_out = []

    for exp_ind in range(num_slices):
        out_dat = data_in[window_len * exp_ind : window_len*(exp_ind+1)]

        ########
        # to df

        out_df = pd.DataFrame(out_dat)

        # drop old index column
        
        out_df.drop(out_df.columns[[0]], axis=1, inplace=True)

        ########
        # append this window to our total num of trials
        data_out.append(out_df)

        out_df.to_csv("./experiment"+str(exp_ind)+"/"+col_name+"/trial"+str(bandwidth_num)+".csv")

    return data_out

def get_training_and_testing_for_bandwidth(bandwidth_num, num_experiments):
    file = "./out/trendent_increasing/trial"+str(bandwidth_num)+"/"

    #### read csv from file and parse as numpy array

    x_input_csv = np.genfromtxt(file+"data_x.csv", delimiter=",", skip_header=1)
    y_input_csv = np.genfromtxt(file+"data_y.csv", delimiter=",", skip_header=1)
    z_input_csv = np.genfromtxt(file+"data_z.csv", delimiter=",", skip_header=1)
    abs_input_csv = np.genfromtxt(file+"data.csv", delimiter=",", skip_header=1)

    exp_len = int(len(x_input_csv) / num_experiments) # len of each window

    # [[window 1],[window 2],[window 3]...]
    x_trial = slice_data(x_input_csv, num_experiments, exp_len, bandwidth_num, "x")
    y_trial = slice_data(y_input_csv, num_experiments, exp_len, bandwidth_num, "y")
    z_trial = slice_data(z_input_csv, num_experiments, exp_len, bandwidth_num, "z")
    abs_trial = slice_data(abs_input_csv, num_experiments, exp_len, bandwidth_num, "abs")

    testing = []

    training = []
    
    # fill testing set

    for test_ind in range(num_experiments-2):
        testing.append([])

        testing[test_ind].append(pd.Series([
            x_trial[test_ind][[2]].mean(),
            y_trial[test_ind][[2]].mean(),
            z_trial[test_ind][[2]].mean(),
            abs_trial[test_ind][[2]].mean(),

            x_trial[test_ind][[2]].min(),
            y_trial[test_ind][[2]].min(),
            z_trial[test_ind][[2]].min(),
            abs_trial[test_ind][[2]].min(),

            x_trial[test_ind][[2]].max(),
            y_trial[test_ind][[2]].max(),
            z_trial[test_ind][[2]].max(),
            abs_trial[test_ind][[2]].max(),
        ]))

    
    # populate training array
    
    training.append(pd.Series([
        (x_trial[0][[2]].mean() + x_trial[1][[2]].mean()) / 2,
        (y_trial[0][[2]].mean() + y_trial[1][[2]].mean()) / 2,
        (z_trial[0][[2]].mean() + z_trial[1][[2]].mean()) / 2,
        (abs_trial[0][[2]].mean() + abs_trial[1][[2]].mean()) / 2,

        (x_trial[0][[2]].min() + x_trial[1][[2]].min()) / 2,
        (y_trial[0][[2]].min() + y_trial[1][[2]].min()) / 2,
        (z_trial[0][[2]].min() + z_trial[1][[2]].min()) / 2,
        (abs_trial[0][[2]].min() + abs_trial[1][[2]].min()) / 2,

        (x_trial[0][[2]].max() + x_trial[1][[2]].max()) / 2,
        (y_trial[0][[2]].max() + y_trial[1][[2]].max()) / 2,
        (z_trial[0][[2]].max() + z_trial[1][[2]].max()) / 2,
        (abs_trial[0][[2]].max() + abs_trial[1][[2]].max()) / 2
    ]))
    

    return (testing, training)

#################################################################
### MAIN

num_exp = 5

def main_run():
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
        bd_testing, bd_training = get_training_and_testing_for_bandwidth(bandwidth_pct, num_exp)
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