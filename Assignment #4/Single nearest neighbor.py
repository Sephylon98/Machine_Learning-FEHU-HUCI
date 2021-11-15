# Written by: Mohamed Ashraf Mohamed Shebl , Student ID: 19037

import numpy as np
import pandas as pd
import os
import math

os.chdir(r'C:\Users\Midox\Documents\Python Scripts\Machine_Learning-FEHU-HUCI\Assignment #4') #remove this for your code

x = np.array([1.8 , -0.56 , 1.51]).reshape(3,1) #feature vector of unknown pattern
print("feature vector of unknown pattern: ",x)
print("--------------------")

#print(x[0][0]) 1.8
#print(x[1][0]) -0.56
#print(x[2][0]) 1.51

class_1 = pd.read_csv('class 1.csv')
class_2 = pd.read_csv('class 2.csv')
class_3 = pd.read_csv('class 3.csv')

print("C1:")
print(class_1)
print("--------------------")
print("C2:")
print(class_2)
print("--------------------")
print("C3:")
print(class_3)
print("--------------------")


data_class_1 = np.array([class_1['x1'],class_1['x2'],class_1['x3']]).reshape(10,3)
data_class_2 = np.array([class_2['x1'],class_2['x2'],class_2['x3']]).reshape(10,3)
data_class_3 = np.array([class_3['x1'],class_3['x2'],class_3['x3']]).reshape(10,3)

print(data_class_1)
print("--------------------")
print(data_class_2)
print("--------------------")
print(data_class_3)
print("--------------------")

#print(data_class_3[0][0],data_class_3[0][1],data_class_3[0][2]) #-1.54 5.41 1.55
#--------------------------------------------------------------
#print(data_class_3[9][0],data_class_3[9][1],data_class_3[9][2]) #1.97 -0.99 0.08

def Euclidean_distance(training_data,feature_vector):
    euclidean_distances = []
    for row in range(10):
           euclidean_distances.append(math.sqrt((feature_vector[0][0] - training_data[row][0])**2 + (feature_vector[1][0] - training_data[row][1])**2 +(feature_vector[2][0] - training_data[row][2])**2))
    return euclidean_distances   

dclass1 = Euclidean_distance(data_class_1,x)
dclass2 = Euclidean_distance(data_class_2,x)
dclass3 = Euclidean_distance(data_class_3,x)

dclass1.sort()
dclass2.sort()
dclass3.sort()

print("Distances for class 1: ",dclass1)
print("--------------------")
print("Distances for class 2: ",dclass2)
print("--------------------")
print("Distances for class 3: ",dclass3)
print("--------------------")

Single_nearest_classifier_dict = {'Class 1':dclass1[0],'Class 2':dclass2[0],'Class 3':dclass3[0]}
print("The unknown pattern from: ",min(Single_nearest_classifier_dict, key=Single_nearest_classifier_dict.get))