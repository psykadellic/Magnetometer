import numpy as np
import matplotlib.pyplot as plt
import utils

##########################
# cleaned input should be array of tuples [](x,y)
# use zip and unzip when modifying accordingly
#
#
##########################



if __name__ == "__main__":
    input_csv = np.genfromtxt("./csv/Raw Data.csv", delimiter=",", skip_header=1).T
    in_x_axis = input_csv[0]
    in_y_axis = input_csv[4]

    plt.plot(in_x_axis, in_y_axis, linewidth=2, linestyle="-", c="b")
    plt.show()

    # [[x1, x2, x3]
    #  [y1, y2, y2]]
    arr_in = np.vstack((in_x_axis, in_y_axis)).T

    # [[x1, y1]
    #  [x2, y2]
    #  [x3, y3]]

    # slice arry_in into proper windows

    sliced_arr = utils.slice_arr(arr_in)
    # [ [[x1, y1]
    #    [x2, y2]],
    #   [[x3, y3]
    #    [x4, y4]]  ]

    ma_out_x = np.array([])
    ma_out_y = np.array([])

    filtered_out_x = np.array([])
    filtered_out_y = np.array([])

    combined_out_y = np.array([])
    for arr in sliced_arr:
        window_size = 25
        iter_arr = utils.trim_arr_mod(arr, window_size)
        iter_arr_transpose = iter_arr.T

        # print(iter_arr_transpose.shape)
        
        ma_out_x = np.append(ma_out_x, utils.moving_avg(arr_in=iter_arr_transpose[0], window=window_size))
        
        y_ma = utils.moving_avg(arr_in=iter_arr_transpose[1], window=window_size)
        ma_out_y = np.append(ma_out_y, y_ma)

        filtered_out_x = np.append(filtered_out_x, iter_arr_transpose[0])
        filtered_out_y = np.append(filtered_out_y, utils.utils_sav_filter(arr_in=iter_arr_transpose[1], window=window_size))

        combined_out_y = np.append(combined_out_y, utils.utils_sav_filter(arr_in=y_ma, window=5))
    
    # ma filter trials
    plt.plot(ma_out_x, ma_out_y, linewidth=2, linestyle="-", c="b")
    plt.show()

    # filtered trials
    plt.plot(filtered_out_x, filtered_out_y, linewidth=2, linestyle="-", c="b")
    plt.show()

    plt.plot(ma_out_x, combined_out_y, linewidth=2, linestyle="-", c="b")
    plt.show()