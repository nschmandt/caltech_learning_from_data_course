import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

def calc_change_v(w, points_to_classify, correct_pred):
    total_value=0
    for i in np.arange(0, len(points_to_classify)):
        total_value+=np.log(1+np.exp(-correct_pred[i]*w*points_to_classify[i]))
    total_value=total_value/len(points_to_classify)

average_error=[]
for i in range(0, 1000):

    #Create a hidden correct classifying line, from two random points

    hidden_f=(np.random.rand(2,2)-.5)*2
    m=(hidden_f[1,1]-hidden_f[0,1])/(hidden_f[1,0]-hidden_f[0,0])
    b=-m*hidden_f[0,0]+hidden_f[0,1]
    number_of_points_to_classify=100
    target_function=np.array([-m, 1, -b])

    #generate a number of points that will be classified

    points_to_classify=(np.random.rand(number_of_points_to_classify,2)-.5)*2
    points_to_classify=np.append(points_to_classify, np.ones((number_of_points_to_classify,1)), axis=1)

    #identify what the correct predictions should be and classify them

    correct_pred=np.sign(np.sum(points_to_classify*target_function, axis=1))

    w=[1,1,1]
    prev_w=[0,0,0]

    while np.linalg.norm(w-prev_w)>.01:
        prev_w=w
        change_v=calc_change_v(w, points_to_classify, correct_pred)
        w=prev_w + .01 * change_v







    #perform linear regression on the data set

    # linear_regression_matrix=np.dot(np.dot(inv(np.dot(points_to_classify.transpose(),points_to_classify)), \
    #                                        points_to_classify.transpose()),correct_pred)
    #
    # linear_prediction=np.sign((linear_regression_matrix*points_to_classify).sum(axis=1))
    #
    # #generate and additional 1000 points as the data set and classify those
    #
    # number_of_additional_test_points=1000
    #
    # test_points_to_classify = (np.random.rand(number_of_additional_test_points, 2) - .5) * 2
    # test_points_to_classify = np.append(test_points_to_classify, np.ones((number_of_additional_test_points, 1)), axis=1)
    #
    # test_prediction = np.sign((linear_regression_matrix * test_points_to_classify).sum(axis=1))
    #
    # correct_test_results=np.sign(np.sum(test_points_to_classify*target_function, axis=1))
    #
    # # calculate the error
    #
    # average_error.append(np.sum(correct_test_results != test_prediction))

#print(np.mean(average_error))