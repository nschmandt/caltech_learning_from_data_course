import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

def hidden_f(x, y):
    return np.sign(x**2+y**2-.6)

#set up the secret correct function, starting with two random points and creating a classifying line

average_error=[]
average_test_error=[]
weight_error=[]

weight_linear_regression_matrix_av=np.zeros((6, 1000))

for j in range(0,1000):

    #set up points to be classified

    number_of_points_to_classify=1000

    points_to_classify=(np.random.rand(number_of_points_to_classify,2)-.5)*2

    # uncomment these lines to see a scatter plot of the classified points
    # plt.figure()
    # plt.scatter(points_to_classify[:,0], points_to_classify[:,1])
    # plt.show()

    #set up correct prediction, note that 10% are misclassified as "noise"

    correct_pred=[]
    noise_generation=list(range(len(points_to_classify)))
    np.random.shuffle(noise_generation)
    for i in range(0, len(points_to_classify)):
        correct_pred.append(hidden_f(points_to_classify[i,0], points_to_classify[i,1]))
        if noise_generation[i]<(len(points_to_classify)/10):
            correct_pred[i]=correct_pred[i]*-1

    #Calculate a linear regression of the data and in-sample error

    linear_regression_matrix = np.dot(np.dot(inv(np.dot(points_to_classify.transpose(), points_to_classify)), \
                                             points_to_classify.transpose()), correct_pred)

    linear_prediction = np.sign((linear_regression_matrix * points_to_classify).sum(axis=1))

    average_error.append(np.sum(correct_pred != linear_prediction))

    #Construct a matrix for nonlinear transformation, of (1, x1, x2, x1x2, x1**2, x2**2)

    weight_matrix = np.ones((len(points_to_classify), 6))
    for i in range(0, len(weight_matrix)):
        weight_matrix[i, 1] = points_to_classify[i, 0]
        weight_matrix[i, 2] = points_to_classify[i, 1]
        weight_matrix[i, 3] = points_to_classify[i, 0] * points_to_classify[i, 1]
        weight_matrix[i, 4] = points_to_classify[i, 0] ** 2
        weight_matrix[i, 5] = points_to_classify[i, 1] ** 2

    #Perform linear regression on transformed data

    weight_linear_regression_matrix = np.dot(np.dot(inv(np.dot(weight_matrix.transpose(), weight_matrix)), \
                                             weight_matrix.transpose()), correct_pred)

    weight_prediction = np.sign((weight_linear_regression_matrix * weight_matrix).sum(axis=1))

    weight_error.append(np.sum(correct_pred != weight_prediction))

    weight_linear_regression_matrix_av[:, j]=(weight_linear_regression_matrix)

    # generate and additional 1000 points as the data set and classify those

    number_of_additional_test_points = 1000

    test_points_to_classify = (np.random.rand(number_of_additional_test_points, 2) - .5) * 2

    test_weight_matrix = np.ones((len(test_points_to_classify), 6))
    for i in range(0, len(weight_matrix)):
        test_weight_matrix[i, 1] = test_points_to_classify[i, 0]
        test_weight_matrix[i, 2] = test_points_to_classify[i, 1]
        test_weight_matrix[i, 3] = test_points_to_classify[i, 0] * test_points_to_classify[i, 1]
        test_weight_matrix[i, 4] = test_points_to_classify[i, 0] ** 2
        test_weight_matrix[i, 5] = test_points_to_classify[i, 1] ** 2

    test_prediction = np.sign((weight_linear_regression_matrix * test_weight_matrix).sum(axis=1))

    correct_test_results = []
    test_noise_generation = list(range(len(test_points_to_classify)))
    np.random.shuffle(test_noise_generation)
    for i in range(0, len(test_points_to_classify)):
        correct_test_results.append(hidden_f(test_points_to_classify[i, 0], test_points_to_classify[i, 1]))
        if test_noise_generation[i] < (len(test_points_to_classify) / 10):
            correct_test_results[i] = correct_test_results[i] * -1

    # calculate the error

    average_test_error.append(np.sum(correct_test_results != test_prediction))

print(np.mean(average_error)/number_of_points_to_classify)
print(np.mean(weight_linear_regression_matrix_av, axis=1))

print(np.mean(average_test_error)/1000)