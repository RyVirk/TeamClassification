import csv 
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

#removed unwanted columns and joined 4 datasets seen in folder separately outside of main file

Handle = open("JoinedStats.csv",encoding='latin-1')

df = pd.read_csv('JoinedStats.csv')
print(df)
    