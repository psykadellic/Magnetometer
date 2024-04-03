import numpy as np
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
    
