import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt

list_of_cities = []

##########################    Parameters    ###################################
mut_prob = 0.4

# Number of generations to run for
generations = 1000

# Population size of 1 generation (RoutePop)
pop_size = 10

# Size of the tournament selection. 
tournament_size = 2

# Dataset resize into 10 cities to ful list of 980 cities
#####################
# File name
# test_10.csv  generation = 10, popsize = 2000, tour_size = 10
# test_20.csv 
# test_50.csv
# test_100.csv
# test_200.csv
# test_450.csv
# test_980.csv
#####################
csv_name = 'test_10.csv'

df = pd.read_csv(csv_name, header = 0)
df = df.set_index(df['city'])
df = df.drop('city', 1)
df_drop = df.drop_duplicates(subset = 'i', keep = 'first')

dataset = df_drop.as_matrix(columns = None)
###############################################################################
def main():

#    print(df.head())
#    print(dataset)

    print("######################################################################")
    print("No. of Random Solution = %a" %pop_size, "\n")
    
    soln = []
    fitness = []
   
    for a in range(0, pop_size):   
        order =  list(range(dataset.shape[0]))
        random.shuffle(order)
        soln.append(order)
        length = calc_length(dataset, order)
        fitness.append(length)
    
#        print(a, soln[a], "{f(s) = ", fitness[a], "}")
        
    print("######################################################################") 
    for x in range(0, generations):        
#        print("######################################################################")  
#        print("Tournament\n")
        Tour_result = []
        Tour_fitness = []    
        Tour_result = random_subset(soln, tournament_size)

        for a in range(0, len(Tour_result)):
            Tour_fit = calc_length(dataset, Tour_result[a])
            Tour_fitness.append(Tour_fit)

        mutate_soln = []
        high_f, high_index = h_fit(Tour_fitness)
#        print(Tour_result[high_index], "{f(s) = ", Tour_fitness[high_index], "}")
    
        mutate_soln, mutate_length = mutate(high_index, Tour_result, dataset)

    
        low_fit, low_index = l_fit(fitness)

        if fitness[low_index] > mutate_length:          
            soln[low_index] = mutate_soln
            fitness[low_index] = mutate_length
    
        best_fit, best_index = h_fit(fitness)
    
#        print("\nBest_Solution")
        print("Gen",x+1,soln[best_index], "{f(s) = ", best_fit, "}")
          
    return soln[best_index], fitness[best_index]

###############################################################################
#GA Stuff
#1-point Crossover
def Crossover_1P(i, j, soln, fitness, dataset):
    """
    i: parent 1
    j: parent 2
    """
    child_soln = []
    x = random.randint(0,len(soln[i])-1)

    for m in range(0, x):
        child_soln.append(soln[i][m]) # set the values to eachother
    
    for n in range(x, 10):
        child_soln.append(soln[j][n]) # set the values to eachother

    print("Crossover position = %x" %x)            
    child_length = calc_length(dataset, child_soln)
#    print(child_soln, "{f(s) = ", child_length, "}")
    return child_soln, child_length

###############################################################################
#Mutation
def mutate(i, soln, dataset):
    '''
    Route() --> Route()
    Swaps two random indexes in route_to_mut.route. Runs k_mut_prob*100 % of the time
    '''
#    x = "no"
    mut_soln = soln[i]
    mut_1 = []
    mut_2 = []
    mut_pos1 = random.randint(0,len(soln[i])-1)
    mut_pos2 = random.randint(0,len(soln[i])-1)
#    chance = random.random()
#    set to always mutate
    chance = 1
    
    while mut_pos1 == mut_pos2:
        mut_pos2 = random.randint(0,len(soln[i])-1) 
    
    
    # k_mut_prob %
    if chance > mut_prob:
        if mut_pos1 == mut_pos2:
#            x = "no"

            mut_length = calc_length(dataset, soln[i])
            mut_soln = soln[i]
#            print("mutate = ", x)
#            print(mut_soln, "{f(s) = ", mut_length, "}")
            return mut_soln, mut_length

    # Otherwise swap them:
        else:
#            x = "yes"
            mut_soln = soln[i]
            mut_1 = soln[i][mut_pos2]
            mut_2 = soln[i][mut_pos1]

            mut_soln[mut_pos1] = mut_1
            mut_soln[mut_pos2] = mut_2

    mut_length = calc_length(dataset, mut_soln)
    return mut_soln, mut_length

###############################################################################
#fittest
#highest
def h_fit(fitness):
    h_f = min(fitness)
    index_hf = fitness.index(min(fitness))
    
    return h_f, index_hf

#lowest
def l_fit(fitness):
    l_f = max(fitness)
    index_lf = fitness.index(max(fitness))
    
    return l_f, index_lf

###############################################################################
#Tournament
def random_subset(soln, tournament_size):
    result = []
    N = 0

    for item in soln:
        N += 1
        if len(result) < tournament_size:
            result.append(item)
        else:
            s = int(random.random() * N)
            if s < tournament_size:
                result[s] = item

    return result

###############################################################################
#Calculate length
def calc_length(dataset, path):
    length = 0
    for i in list(range(len(path))):
        length += distance(dataset[path[i-1]], dataset[path[i]])
#        length += tsp_cost(path[i-1], path[i], dataset)
    return length

#Distance square
def distance(c1, c2):
    t1 = c2[0] - c1[0]
    t2 = c2[1] - c1[1]

    return math.sqrt(t1**2 + t2**2)

###############################################################################
def tsp_cost(i, j, dataset):
    pi = 3.14159265358979323846264
    
    lat_i = (pi*dataset[i][0])/180
    lat_j = (pi*dataset[j][0])/180
    long_i = (pi*dataset[i][1])/180
    long_j = (pi*dataset[j][1])/180
    
    q1 = math.cos(lat_j)*math.sin(long_i-long_j)
    q3 = math.sin((long_i-long_j)/2.0)
    q4 = math.cos((long_i-long_j)/2.0)
    q2 = math.sin(lat_i+lat_j)*q3*q3 - math.sin(lat_i-lat_j)*q4*q4
    q5 = math.cos(lat_i-lat_j)*q4*q4 - math.cos(lat_i+lat_j)*q3*q3
    
    return (6378388.0*math.atan2(math.sqrt((q1*q1)+(q2*q2)),q5)+1.0)

#####################################################
#run everything nicely

list_plot = []
b_soln, b_fitness = main()
print("######################################################################")
print("\nBest Solution")
print(b_soln, "{f(s) = ", b_fitness, "}")

for a in range(0, len(b_soln)):
    list_plot.append(dataset[b_soln[a]])
    
list_plot.append(list_plot[0])

plt.gca().invert_yaxis()
plt.scatter(*zip(*dataset))
plt.plot(*zip(*list_plot))
plt.show()
print("######################################################################")














