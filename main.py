import os
import pandas as pd
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

    for root, dir, files in os.walk("./csv"):
        for file in files:
            if ".DS_Store" in file:
                    continue
            ret_arr = utils.read_and_split_file(file)
            try:
                os.mkdir("./out/"+file[0:-4])
            except:
                ""

            averages = []
            
            for ind, arr in enumerate(ret_arr):
                try:
                    os.mkdir("./out/"+file[0:-4]+"/trial"+str(ind))
                except:
                    ""
                df = pd.DataFrame(arr)
                df.to_csv("./out/"+file[0:-4]+"/trial"+str(ind)+"/data.csv")
                iter_arr_transpose = arr.T
                plt.plot(iter_arr_transpose[0], iter_arr_transpose[1], linewidth=2, linestyle="-", c="b")
                plt.savefig("./out/"+file[0:-4]+"/trial"+str(ind)+"/plot.png")
                plt.cla()
                # plt.show()

                averages.append(df[[1]].mean())
            
            average_df = pd.DataFrame(averages)
            average_df.to_csv("./out/"+file[0:-4]+"/average_data.csv")
            plt.plot(average_df.index, average_df.values)
            plt.savefig("./out/"+file[0:-4]+"/average_plot.png")