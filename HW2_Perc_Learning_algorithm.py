import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

#set up the secret correct function, starting with two random points and creating a classifying line

average_iterations=[]

for i in range(0,1000):

    hidden_f=(np.random.rand(2,2)-.5)*2
    m=(hidden_f[1,1]-hidden_f[0,1])/(hidden_f[1,0]-hidden_f[0,0])
    b=-m*hidden_f[0,0]+hidden_f[0,1]
    number_of_points_to_classify=10
    correct_percep=np.array([-m, 1, -b])

    #set up points to be classified

    points_to_classify=(np.random.rand(number_of_points_to_classify,2)-.5)*2
    points_to_classify=np.append(points_to_classify, np.ones((number_of_points_to_classify,1)), axis=1)

    # uncomment these lines to see a scatter plot of the classified points
    # plt.figure()
    # plt.scatter(points_to_classify[:,0], points_to_classify[:,1])
    # plt.show()

    #set up correct prediction

    correct_pred=np.sign(np.sum(points_to_classify*correct_percep, axis=1))

    #classify the weights as zeros (comment out second line) or by the linear regression classification

    weights=np.array([0,0,0])

    weights=np.dot(np.dot(inv(np.dot(points_to_classify.transpose(),points_to_classify)), \
                                           points_to_classify.transpose()),correct_pred)

    # first initial classification

    prediction = np.sign((points_to_classify * weights).sum(axis=1))
    error_list = points_to_classify[prediction != correct_pred]
    error_pred = correct_pred[prediction != correct_pred]

    #Repeat until correct, save the number of iterations required

    number_of_iterations=[]
    while len(error_list)>0:
        temp = np.random.randint(0, len(error_list))
        weights = weights + (error_pred[temp]) * error_list[temp]
        prediction = np.sign((points_to_classify * weights).sum(axis=1))
        error_list = points_to_classify[prediction != correct_pred]
        error_pred = correct_pred[prediction != correct_pred]
        number_of_iterations.append(len(error_list))
        if len(number_of_iterations)>10000: #If it hasn't converged in 10000 tried, something is wrong
            break
            print("perceptron failed to converge!")
    average_iterations.append(len(number_of_iterations))


print(np.mean(average_iterations))

