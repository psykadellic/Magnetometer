import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils


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
            mins = []
            maxes = []
            std_dev = []
            
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
                mins.append(df[[1]].min())
                maxes.append(df[[1]].max())
                std_dev.append(df[[1]].std())
            
            # average_df = pd.DataFrame(averages)
            # average_df.to_csv("./out/"+file[0:-4]+"/average/data_avg_abs.csv")
            # plt.plot(average_df.index, average_df.values)
            # plt.savefig("./out/"+file[0:-4]+"/average/plot_avg_abs.png")
            # plt.cla()

            # average_df = pd.DataFrame(mins)
            # average_df.to_csv("./out/"+file[0:-4]+"/average/data_min_abs.csv")
            # plt.plot(average_df.index, average_df.values)
            # plt.savefig("./out/"+file[0:-4]+"/average/plot_min_abs.png")
            # plt.cla()

            # average_df = pd.DataFrame(maxes)
            # average_df.to_csv("./out/"+file[0:-4]+"/average/data_max_abs.csv")
            # plt.plot(average_df.index, average_df.values)
            # plt.savefig("./out/"+file[0:-4]+"/average/plot_max_abs.png")
            # plt.cla()

            # average_df = pd.DataFrame(std_dev)
            # average_df.to_csv("./out/"+file[0:-4]+"/average/data_std_abs.csv")
            # plt.plot(average_df.index, average_df.values)
            # plt.savefig("./out/"+file[0:-4]+"/average/plot_std_abs.png")