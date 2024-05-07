import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
import re

if __name__ == "__main__":
    for i in range(3):
        with open("./test_"+str(i)+"_likely.txt", "r") as test_file:
            
            in_one = 0
            in_two = 0
            in_three = 0

            distance_from = {}

            for line in test_file.readlines():
                matches = list(map(lambda x: int(x), re.findall("\d+", line)))
                actual, predicted = matches[0], matches[1:]

                if actual == predicted[0]:
                    in_one += 1
                    in_two += 1
                    in_three += 1
                elif actual == predicted[1]:
                    in_two += 1
                    in_three += 1
                elif actual == predicted[2]:
                    in_three += 1
                else:
                    distance_from[actual] = min([abs(actual-predicted[0]),abs(actual-predicted[1]),abs(actual-predicted[2])])
                
            print("TEST %d %d%% %d%% %d%%"%(i, in_one, in_two, in_three))
            print(distance_from)