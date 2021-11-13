# Written by: Mohamed Ashraf Mohamed Shebl , Student ID: 19037

import numpy as np
from numpy.linalg import inv
import pandas as pd
import os
import seaborn as sn
import matplotlib.pyplot as plt

os.chdir(r'C:\Users\Midox\Documents\Python Scripts\Machine_Learning-FEHU-HUCI\Assignment #2') #remove this for your code

x = np.array([1.8 , -0.56 , 1.51]).reshape(3,1) #feature vector of unknown pattern
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

u1 = np.array([class_1['x1'].mean(), class_1['x2'].mean(), class_1['x3'].mean()]).reshape(3,1)
u2 = np.array([class_2['x1'].mean(), class_2['x2'].mean(), class_2['x3'].mean()]).reshape(3,1)
u3 = np.array([class_3['x1'].mean(), class_3['x2'].mean(), class_3['x3'].mean()]).reshape(3,1)

print("u1: ",u1)
print("--------------------")
print("u2: ",u2)
print("--------------------")
print("u3: ",u3)
print("--------------------")

data_class_1 = np.array([class_1['x1'],class_1['x2'],class_1['x3']])
covMatrix1 = np.cov(data_class_1,bias=True)
print('covMatrix1: ',covMatrix1)
print("--------------------")
sn.heatmap(covMatrix1, annot=True, fmt='g')
plt.show()

data_class_2 = np.array([class_2['x1'],class_2['x2'],class_2['x3']])
covMatrix2 = np.cov(data_class_2,bias=True)
print('covMatrix2:', covMatrix2)
print("--------------------")
sn.heatmap(covMatrix2, annot=True, fmt='g')
plt.show()

data_class_3 = np.array([class_3['x1'],class_3['x2'],class_3['x3']])
covMatrix3 = np.cov(data_class_3,bias=True)
print('covMatrix3: ',covMatrix3)
print("--------------------")
sn.heatmap(covMatrix3, annot=True, fmt='g')
plt.show()

def discriminant_fun(covMatrix,feature_vector,u):
    ainv_covMatrix = np.linalg.inv(covMatrix)
    row_vector = (-1/2)*np.log(np.linalg.det(covMatrix))- (1/2)*(feature_vector.reshape(1,3) - u1.reshape(1,3))
    col_vector = np.dot(ainv_covMatrix,(feature_vector-u)) + np.log(1/3)
    return np.dot(row_vector,col_vector) # 1x3 dot product 3x1 = 1x1 matrix

g1_x = discriminant_fun(covMatrix1,x,u1)
g2_x = discriminant_fun(covMatrix2,x,u2)
g3_x = discriminant_fun(covMatrix3,x,u3)

print('g1(x): ',g1_x[0][0])
print('g2(x): ',g2_x[0][0])
print('g3(x): ',g3_x[0][0])
print("--------------------")

Bayes_classifier_dict = {'Class 1':g1_x[0][0],'Class 2':g2_x[0][0],'Class 3':g3_x[0][0]}
print("The unknown pattern from: ",max(Bayes_classifier_dict, key=Bayes_classifier_dict.get))


