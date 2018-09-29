# encoding=utf-8
# Date: 2018-09-29
# Author: MJUZY


import numpy as np


def test(syn0, syn1, X):

    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    print("The test result : ")
    print(l2)


def nonlin(x, deriv=False):
    """

    :param x:
    :param deriv: if deriv==False, then the function will return the result of sigmoid
                    if deriv=True, then the function will return the derivative of the result
                     and the result is out * (1 - out)
                Attention:
                    The indication process please see the D:\CNN Learning-Process\
    :return:
    """
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

"""Description:

    input dataset
"""
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

"""Description:

    output dataset (label)
"""
y = np.array([[0],
              [1],
              [1],
              [0]])

"""Description:

    seed random numbers to make calculation deterministic
"""
np.random.seed(1)

"""Description:

    initialize weights randomly with mean 0
"""
syn0 = 2 * np.random.random((3, 4)) - 1
syn1 = 2 * np.random.random((4, 1)) - 1

for j in range(100000):

    """Description:
    
        Feed forward through layers 0, 1 and 2
        named Forward Propagation
    """
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    """Description:
        
        restore the temporay loss
    """
    l2_error = y - l2

    """Description:
    
        show the mean loss of the output layer
        every 10000 steps
    """
    if (j % 10000 == 0):
        print("Error : ", str(np.mean(np.abs(l2_error))))

    """Description:
        
        The direction of the target value
    """
    l2_delta = l2_error * nonlin(l2, deriv=True)

    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1, deriv=True)

    """Description
    
        refresh the weights of the l1 and l2
    """
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)


X_test = np.array([[0, 1, 1],
              [0, 0, 1],
              [1, 1, 1],
              [1, 0, 1]])

test(syn0, syn1, X_test)