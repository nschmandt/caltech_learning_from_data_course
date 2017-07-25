import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import os

os.chdir('/home/nick/caltech_data/')

in_data=pd.read_csv('in_sample_data.txt', header=None, sep='\s+')
out_data=pd.read_csv('out_sample_data.txt', header=None, sep='\s+')

in_data_matrix=np.ones((len(in_data), 8))
out_data_matrix=np.ones((len(in_data), 8))

#create the data matrices as specified in the assignment

for i in range(0, len(in_data)):
    in_data_matrix[i, 1]=in_data[0][i]
    in_data_matrix[i, 2]=in_data[1][i]
    in_data_matrix[i, 3]=in_data[0][i] ** 2
    in_data_matrix[i, 4]=in_data[1][i] ** 2
    in_data_matrix[i, 5]=in_data[0][i] * in_data[1][i]
    in_data_matrix[i, 6]=np.abs(in_data[0][i] - in_data[1][i])
    in_data_matrix[i, 7]=np.abs(in_data[0][i] + in_data[1][i])

for i in range(0, len(out_data)):
    out_data_matrix[i, 1]=out_data[0][i]
    out_data_matrix[i, 2]=out_data[1][i]
    out_data_matrix[i, 3]=out_data[0][i] ** 2
    out_data_matrix[i, 4]=out_data[1][i] ** 2
    out_data_matrix[i, 5]=out_data[0][i] * out_data[1][i]
    out_data_matrix[i, 6]=np.abs(out_data[0][i] - out_data[1][i])
    out_data_matrix[i, 7]=np.abs(out_data[0][i] + out_data[1][i])

#create the linear regression matrix

regression_matrix = np.dot(np.dot(inv(np.dot(in_data_matrix.transpose(), in_data_matrix)), \
                                  in_data_matrix.transpose()), in_data[2])

in_data_prediction=np.sign(np.sum(regression_matrix*in_data_matrix, axis=1))

print(np.sum(data_prediction != in_data[2])/len(in_data[2])) #in-sample error

out_data_prediction=np.sign(np.sum(regression_matrix*out_data_matrix, axis=1))

print(np.sum(data_prediction != out_data[2])/len(out_data[2])) #out-sample error

#this time, with regression

lamb=.001

regression_reg_matrix = np.dot(np.dot(inv(np.dot(in_data_matrix.transpose(), in_data_matrix)+ \
                                      lamb*np.identity(out_data_matrix.shape[1])), in_data_matrix.transpose()), in_data[2])

out_data_reg_prediction=np.sign(np.sum(regression_reg_matrix*out_data_matrix, axis=1))

print(np.sum(out_data_reg_prediction != out_data[2])/len(out_data[2]))