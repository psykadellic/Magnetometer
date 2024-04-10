import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils

num_exp = 5

if __name__ == "__main__":

    for ind in range(num_exp):
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

    testing_1 = []
    testing_2 = []
    testing_3 = []

    # ok we are hardcoding variables, which is wrong, but whatever.

    training = []
    

    for i in range(100):
        file = "./out/trendent_increasing/trial"+str(i)+"/"

        x_input_csv = np.genfromtxt(file+"data_x.csv", delimiter=",", skip_header=1)
        y_input_csv = np.genfromtxt(file+"data_y.csv", delimiter=",", skip_header=1)
        z_input_csv = np.genfromtxt(file+"data_z.csv", delimiter=",", skip_header=1)
        abs_input_csv = np.genfromtxt(file+"data.csv", delimiter=",", skip_header=1)


        exp_len = int(len(x_input_csv) / num_exp) # n experiments

        x_trial = []
        y_trial = []
        z_trial = []
        abs_trial = []

        
        for j in range(num_exp):
            x_dat = x_input_csv[exp_len * j : exp_len*(j+1)]
            y_dat = y_input_csv[exp_len * j : exp_len*(j+1)]
            z_dat = z_input_csv[exp_len * j : exp_len*(j+1)]
            abs_dat = abs_input_csv[exp_len * j : exp_len*(j+1)]

            x_df = pd.DataFrame(x_dat)
            y_df = pd.DataFrame(y_dat)
            z_df = pd.DataFrame(z_dat)
            abs_df = pd.DataFrame(abs_dat)

            # drop old index column
            
            x_df.drop(x_df.columns[[0]], axis=1, inplace=True)
            y_df.drop(y_df.columns[[0]], axis=1, inplace=True)
            z_df.drop(z_df.columns[[0]], axis=1, inplace=True)
            abs_df.drop(abs_df.columns[[0]], axis=1, inplace=True)
            
            x_trial.append(x_df)
            y_trial.append(y_df)
            z_trial.append(z_df)
            abs_trial.append(abs_df)

            x_df.to_csv("./experiment"+str(j)+"/x/trial"+str(i)+".csv")
            y_df.to_csv("./experiment"+str(j)+"/y/trial"+str(i)+".csv")
            z_df.to_csv("./experiment"+str(j)+"/z/trial"+str(i)+".csv")
            abs_df.to_csv("./experiment"+str(j)+"/abs/trial"+str(i)+".csv")

        # testing 1

        testing_1.append(pd.Series([
            x_trial[2][[1]].mean(),
            y_trial[2][[1]].mean(),
            z_trial[2][[1]].mean(),
            abs_trial[2][[1]].mean(),

            x_trial[2][[1]].min(),
            y_trial[2][[1]].min(),
            z_trial[2][[1]].min(),
            abs_trial[2][[1]].min(),

            x_trial[2][[1]].max(),
            y_trial[2][[1]].max(),
            z_trial[2][[1]].max(),
            abs_trial[2][[1]].max(),
        ]))

        # testing 2
        
        testing_2.append(pd.Series([
            x_trial[3][[1]].mean(),
            y_trial[3][[1]].mean(),
            z_trial[3][[1]].mean(),
            abs_trial[3][[1]].mean(),

            x_trial[3][[1]].min(),
            y_trial[3][[1]].min(),
            z_trial[3][[1]].min(),
            abs_trial[3][[1]].min(),

            x_trial[3][[1]].max(),
            y_trial[3][[1]].max(),
            z_trial[3][[1]].max(),
            abs_trial[3][[1]].max(),
        ]))
        # testing 3
        
        testing_3.append(pd.Series([
            x_trial[4][[1]].mean(),
            y_trial[4][[1]].mean(),
            z_trial[4][[1]].mean(),
            abs_trial[4][[1]].mean(),

            x_trial[4][[1]].min(),
            y_trial[4][[1]].min(),
            z_trial[4][[1]].min(),
            abs_trial[4][[1]].min(),

            x_trial[4][[1]].max(),
            y_trial[4][[1]].max(),
            z_trial[4][[1]].max(),
            abs_trial[4][[1]].max(),
        ]))
        
        # populate training array
        
        training.append(pd.Series([
            (x_trial[0][[1]].mean() + x_trial[1][[1]].mean()) / 2,
            (y_trial[0][[1]].mean() + y_trial[1][[1]].mean()) / 2,
            (z_trial[0][[1]].mean() + z_trial[1][[1]].mean()) / 2,
            (abs_trial[0][[1]].mean() + abs_trial[1][[1]].mean()) / 2,

            (x_trial[0][[1]].min() + x_trial[1][[1]].min()) / 2,
            (y_trial[0][[1]].min() + y_trial[1][[1]].min()) / 2,
            (z_trial[0][[1]].min() + z_trial[1][[1]].min()) / 2,
            (abs_trial[0][[1]].min() + abs_trial[1][[1]].min()) / 2,

            (x_trial[0][[1]].max() + x_trial[1][[1]].max()) / 2,
            (y_trial[0][[1]].max() + y_trial[1][[1]].max()) / 2,
            (z_trial[0][[1]].max() + z_trial[1][[1]].max()) / 2,
            (abs_trial[0][[1]].max() + abs_trial[1][[1]].max()) / 2,
        ]))

    with open("test_1_likely.txt","w+") as output:
        for ind, entry in enumerate(testing_1):
            distances = pd.Series(list(map(lambda s2: float((((entry - s2)**2).sum()**0.5).item()), training))) # vector of 100
            
            mins = []
            d_max = distances.max()
            for n in range(3): # find 3 bottom mins
                min_ind = distances.idxmin()
                mins.append(min_ind)

                if min_ind < len(distances): # bro it is returning an index that is not a valid index. ?????
                    distances[min_ind] = d_max
            output.write(str(ind) + "\t" + mins + "\n")

    with open("test_2_likely.txt","w+") as output:
        for ind, entry in enumerate(testing_2):
            distances = pd.Series(list(map(lambda s2: float((((entry - s2)**2).sum()**0.5).item()), training))) # vector of 100
            mins = []
            d_max = distances.max()
            for n in range(3): # find 3 bottom mins
                min_ind = distances.idxmin()
                mins.append(min_ind)

                if min_ind < len(distances): # bro it is returning an index that is not a valid index. ?????
                    distances[min_ind] = d_max
            output.write(str(ind) + "\t" + mins + "\n")


    with open("test_3_likely.txt","w+") as output:
        for ind, entry in enumerate(testing_3):
            distances = pd.Series(list(map(lambda s2: float((((entry - s2)**2).sum()**0.5).item()), training))) # vector of 100
            
            mins = []
            d_max = distances.max()
            for n in range(3): # find 3 bottom mins
                min_ind = distances.idxmin()
                mins.append(min_ind)

                if min_ind < len(distances): # bro it is returning an index that is not a valid index. ?????
                    distances[min_ind] = d_max
            output.write(str(ind) + "\t" + mins + "\n")

    print(test_3_likely)