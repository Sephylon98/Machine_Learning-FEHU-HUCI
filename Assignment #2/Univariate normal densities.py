# Written by: Mohamed Ashraf Mohamed Shebl , Student ID: 19037

import numpy as np
import pandas as pd
import os

os.chdir(r'C:\Users\Midox\Documents\Python Scripts\Machine_Learning-FEHU-HUCI\Assignment #2') #remove this for your code

x = np.array([1.8]) #feature vector of unknown pattern
print("feature vector of unknown pattern: ",x)
print("--------------------")

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

u1 = np.array([class_1['x'].mean()])
u2 = np.array([class_2['x'].mean()])
u3 = np.array([class_3['x'].mean()])

print("u1: ",u1)
print("--------------------")
print("u2: ",u2)
print("--------------------")
print("u3: ",u3)
print("--------------------")

data_class_1 = np.array([class_1['x']])
variance1 = np.cov(data_class_1,bias=True)
print('Variance1: ',variance1)
print("--------------------")

data_class_2 = np.array([class_2['x']])
variance2 = np.cov(data_class_2,bias=True)
print('Variance2: ',variance2)
print("--------------------")

data_class_3 = np.array([class_3['x']])
variance3 = np.cov(data_class_3,bias=True)
print('Variance3: ',variance3)
print("--------------------")

def discriminant_fun(variance,feature_vector,u):
    return  (-(1/2)*np.log(variance)) - (feature_vector[0] - u[0])**2/(2*variance)  + np.log(1/3)

g1_x = discriminant_fun(variance1,x,u1)
g2_x = discriminant_fun(variance2,x,u2)
g3_x = discriminant_fun(variance3,x,u3)

print('g1(x): ',g1_x)
print('g2(x): ',g2_x)
print('g3(x): ',g3_x)
print("--------------------")

Bayes_classifier_dict = {'Class 1':g1_x,'Class 2':g2_x,'Class 3':g3_x}
print("The unknown pattern from: ",max(Bayes_classifier_dict, key=Bayes_classifier_dict.get))