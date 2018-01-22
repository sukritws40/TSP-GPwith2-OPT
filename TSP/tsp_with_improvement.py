import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from time import time
from random import shuffle, randrange, randint
#number of generation
n = 10

def main():
    
    #Data preparation
    df = pd.read_csv('test_200.csv', header = 0)
    df = df.drop('city', 1)
    df_drop = df.drop_duplicates(subset = 'i', keep = 'first')
    #df -> array
    dataset = df_drop.as_matrix(columns = None)
    
    start = time()
    #Swap algorithm
    path, length = swap(dataset)
    print(path)
    
    tottime = time() - start
    print("time = ",tottime)
    print("total_distance", length)

    list_plot = []
    for x in range(0, len(path)):
        list_plot.append(dataset[path[x]])
    
    list_plot.append(list_plot[0])
    plt.gca().invert_yaxis()
    plt.scatter(*zip(*dataset))
    plt.plot(*zip(*list_plot))
    
    plt.show()

################################################################################
#Swap algorithm
def swap(dataset):
    best_order = []
    best_length = float('inf')
    
#    order =  list(range(dataset.shape[0]))
#    shuffle(order)
    
    for i in list(range(n)):
        order =  list(range(dataset.shape[0]))
#Create random solution        
        shuffle(order)
        length = calc_length(dataset, order)
        print("generation: ", 1+i, " [f(s): ", best_length,"]")

        changed = True
        while changed:

            changed = False

            for a in range(0, dataset.shape[0]):


                for b in range(a+1, dataset.shape[0]):

                    new_order = order[:a] + order[a:b][::-1] + order[b:]
                    new_length = calc_length(dataset, new_order)

                    if new_length < length:
                        length = new_length
                        order = new_order
                        changed = True
            
            
        if length < best_length:
            best_length = length
            best_order = order
            
    return best_order, best_length
################################################################################
#Calculate length
def calc_length(dataset, path):
    length = 0
    for i in list(range(0, len(path))):
        length += distance(dataset[path[i-1]], dataset[path[i]])
#        length += tsp_cost(path[i-1], path[i], dataset)
    return length

#Distance square
def distance(c1, c2):
    t1 = c2[0] - c1[0]
    t2 = c2[1] - c1[1]

    return math.sqrt(t1**2 + t2**2)
##################################################################################
def tsp_cost(i, j, dataset):
    pi = 3.14159265358979323846264
    
    lat_i = (pi*dataset[i][0])/180
    lat_j = (pi*dataset[j][0])/180
    long_i = (pi*dataset[i][1])/180
    long_j = (pi*dataset[j][1])/180
    
    q1 = math.cos(lat_j)*math.sin(long_i-long_j)
    q3 = math.sin((long_i-long_j)/2.0)
    q4 = math.cos((long_i-long_j)/2.0)
    q2 = (math.sin(lat_i+lat_j)*q3*q3) - (math.sin(lat_i-lat_j)*q4*q4)
    q5 = (math.cos(lat_i-lat_j)*q4*q4) - (math.cos(lat_i+lat_j)*q3*q3)
    
    return int(6378388.0*math.atan2(math.sqrt((q1*q1)+(q2*q2)),q5)+1.0)
##################################################################################
main()

