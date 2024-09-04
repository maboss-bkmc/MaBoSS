import sys, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


if len(sys.argv) < 2:
    print("Usage: plot_dist.py <file> <line>")
    exit(1)

dist_file = sys.argv[1]
first_index = -1
with open(dist_file, 'r') as f:
    for i, line in enumerate(f):
        if i == 0:
            first_index = next(i for i, col in enumerate(line.strip("\n").split("\t")) if col == "State")
        if line.startswith(sys.argv[2]):
            pop_sizes = [int(x) for x in line.strip("\n").split("\t")[first_index::3]]
            pop_probas = line.strip("\n").split("\t")[first_index+1::3]
            res = pd.Series(pop_probas, index=pop_sizes).astype(float)
            res.sort_index(inplace=True)
            res.plot()
            plt.show() 
            break
    
