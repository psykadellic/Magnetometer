import numpy as np
from scipy.signal import savgol_filter

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

        head_pad = 2,
        tail_pad = 2,
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
        ret_arr[i] = arr_slice
    
    # print("returned nd arr: ", ret_arr)

    return ret_arr
    
