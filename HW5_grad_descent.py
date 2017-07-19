import numpy as np
import matplotlib.pyplot as plt

def func_e(u, v):
    return (u * np.exp( v ) - 2 * v * np.exp( - u))**2

def deriv_u(u, v):
    return 2 * (np.exp(v) + 2 * v * np.exp(-u))*(u*np.exp(v) - 2 * v * np.exp(-u))

def deriv_v(u, v):
    return 2 * (u*np.exp(v) - 2 * v * np.exp(-u)) * (u * np.exp(v) - 2 * np.exp(-u))


u=1
v=1
prev_u=1
prev_v=1
error_values=[]
threshold=10**-14
counter=0
while(func_e(u,v)>threshold):
    error_values.append(func_e(u,v))
    u-=.1*deriv_u(prev_u, prev_v)
    v-=.1*deriv_v(prev_u, prev_v)
    prev_u=u
    prev_v=v
    counter+=1

print(counter)
plt.plot(error_values)
plt.show()