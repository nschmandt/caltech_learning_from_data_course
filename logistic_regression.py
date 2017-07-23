import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

def calc_change_v(w, points_to_classify, correct_pred):
    total_value=0
    for i in np.arange(0, len(points_to_classify)):
        #total_value+=np.log(1+np.exp(-correct_pred[i]*w*points_to_classify[i]))
        total_value+=correct_pred[i]*points_to_classify[i]/(1+ np.exp(correct_pred[i]*w*points_to_classify[i]))

    total_value=total_value/len(points_to_classify)
    return total_value

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

w=np.array([1,1,1])
prev_w=np.array([0,0,0])

while np.linalg.norm(w-prev_w)>.01:
    prev_w=w
    change_v=calc_change_v(w, points_to_classify, correct_pred)
    w=prev_w - .01 * change_v

print(w)
print(target_function)
print(np.dot(target_function,w))
prediction=np.sign(np.dot(points_to_classify, w))
print(np.sum(prediction==correct_pred))
