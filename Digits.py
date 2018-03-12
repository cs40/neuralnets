
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time

from scipy.misc import imread,imsave
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

import cPickle

import os
from scipy.io import loadmat

def part1():
    M = loadmat("mnist_all.mat")
    fig = plt.figure(figsize = (50, 50))
    z = 0
    for i in range(10):
        for j in range(10):
            z = z + 1
            fig.add_subplot(10, 10, z)
            plt.axis('off')
            plt.imshow(M["train" + str(i)][j].reshape((28, 28)), cmap=cm.gray)


    plt.show()
def cost_function(y, p):
    ''' y is a 10 x Z matrix
        p is a 10 x Z matrix
    '''
    m = sum(np.multiply(y, log(p)), axis=0)
    return-1*sum(m)
def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))

def part2(W, x, b):
    '''Return the output of the neural network where W is a 10x784 matrix containing the
    weights, x is a Zx784 matrix of flattened image vector and b is a 10xZ vector of
    biases'''
    output = np.dot(W, x.T) + b
    return softmax(output)

def gradient(p,y,x):
    grad = np.dot(p-y, x)
    return grad

def grad_descent(x, y, b, init_w, alpha, iterations):
    EPS = 1e-5
    prev_w = init_w-10*EPS
    w = init_w.copy()
    max_iter = iterations
    iter  = 0
    while norm(w - prev_w) >  EPS and iter < max_iter:
        prev_w = w.copy()
        p = part2(w, x, b)
        d = gradient(p, y, x)
        w -= alpha*d
        b -= alpha*b

        iter += 1
    return w, b

def grad_descent_mom(x, y, b, init_w, alpha, iterations, gamma = 0.9):
    EPS = 1e-5
    v = 0
    prev_w = init_w-10*EPS
    w = init_w.copy()
    max_iter = iterations
    iter  = 0
    while norm(w - prev_w) >  EPS and iter < max_iter:
        prev_w = w.copy()
        p = part2(w, x, b)
        d = gradient(p, y, x)
        v = gamma * v + alpha * d
        w -= v
        b -= alpha*b

        iter += 1
    return w, b

def get_sets(type, num_samples):
    M = loadmat("mnist_all.mat")
    y = np.zeros((10*num_samples, 10))
    if type == 'training':
        x_range = [i for i in range(5000)]
        random.shuffle(x_range)
        for i in range(10):
            for j in range(num_samples):
                if i == 0 and j == 0:
                    x = M["train0"][x_range[j]].reshape((1, 784))
                else:
                    x = vstack((x, M["train" + str(i)][x_range[j]].reshape((1, 784))))
    else:
        x_range = [i for i in range(800)]
        random.shuffle(x_range)
        for i in range(10):
            for j in range(num_samples):
                if i == 0 and j == 0:
                    x = M["test0"][x_range[j]].reshape((1, 784))
                else:
                    x = vstack((x, M["test" + str(i)][x_range[j]].reshape((1, 784))))
    j = 0
    k = 0
    for i in range(10):
        for z in range(num_samples):
            y[k][i] = 1
            k+=1
        j+=1

    x= x/255.0
    return x, y


def check_3b_correct():
    h = 0.000000000001
    snapshot = cPickle.load(open("snapshot50.pkl"))
    W0 = snapshot["W0"]
    W1 = snapshot["W1"]
    W0 = dot(W0, W1)
    W0 = W0.T
    b = np.zeros((10, 5000))
    x, y = get_sets('training', 500)
    p0 = part2(W0, x, b)
    G = gradient(p0, y.T, x)
    for i in range(390, 395):
        W1 = W0.copy()
        W1[3, i] = W0[3, i] + h
        p1 = part2(W1, x, b)
        finite_diff = G[3, i] - ((cost_function(y.T, p0) - cost_function(y.T, p1))/h)
        print("The difference between the gradient and approximation:" + str(finite_diff))
def check_predictions(pred, y):
    pred = pred.T
    predictions = []
    for i in range(len(pred)):
	max_idx =  argmax(pred[i])
        predictions.append(max_idx)
    num_correct = 0
    total = 0
    for i in range(len(predictions)):
        total += 1
        if y[i][predictions[i]] == 1:
            num_correct +=1
    return num_correct, total
def part4():
    training_size = [500, 1000, 2000, 3000, 4000, 5000]
    test_x, test_y =  get_sets('test', 500)
    performance_training = []
    performance_test = []
    snapshot = cPickle.load(open("snapshot50.pkl"))
    W0 = snapshot["W0"]
    W1 = snapshot["W1"]
    for size in training_size:
        training_x, training_y = get_sets('training', size)
        W2 = dot(W0, W1)
        b = np.zeros((10, 10 * size))
        W3, b = grad_descent(training_x, training_y.T, b, W2.T, 1E-7, 10000)
        b_test = b[:, 0:5000]
        pred_test = part2(W3, test_x, b_test)
        num_correct_test, total_test = check_predictions(pred_test, test_y)
        pred_training = part2(W3, training_x, b)
        num_correct_training, total_training = check_predictions(pred_training, training_y)
        percentage_correct_test = (float(num_correct_test)/float(total_test))*100
        percentage_correct_training = (float(num_correct_training)/float(total_training))*100
        performance_training.append(percentage_correct_training)
        performance_test.append(percentage_correct_test)
        print("The percentage of correct predictions on the training set of size "+ str(size) + " is:" + str(percentage_correct_training) + "%")
        print("The percentage of correct predictions on the test set after training on a set of size " + str(size) + " is:" + str(percentage_correct_test) + "%")
    plt.plot(training_size, performance_training, color = 'g', linewidth=1, marker = 'o', label = 'Training Set Performance')
    plt.plot(training_size, performance_test, color = 'r', linewidth=1, marker = 'o', label = 'Test Set Performance')
    plt.title('Training Size vs. Percentage correct predictions on some datasets')
    plt.ylim([80, 110])
    plt.xlabel('Training size')
    plt.ylabel('Percentage of correct predictions')
    plt.legend()
    plt.show()
    visualize_weights(W3)
def visualize_weights(W):
    
    fig = plt.figure(figsize=(100,100))
    for i in range(len(W)):
	   fig.add_subplot(2, 5, i + 1)
	   plt.imshow(W[i][:].reshape(28, 28), cm.coolwarm)
    plt.show()
def part5():
    #training_size = [500, 1000, 2000, 3000, 4000, 5000]
    training_size = [5000]
    test_x, test_y =  get_sets('test', 500)
    performance_training = []
    performance_test = []
    snapshot = cPickle.load(open("snapshot50.pkl"))
    W0 = snapshot["W0"]
    W1 = snapshot["W1"]
    for size in training_size:
        training_x, training_y = get_sets('training', size)
        W2 = dot(W0, W1)
        b = np.zeros((10, 10 * size))
        W3, b = grad_descent_mom(training_x, training_y.T, b, W2.T, 1E-7, 10000)
        
        b_test = b[:, 0:5000]
	
        pred_test = part2(W3, test_x, b_test)
    	num_correct_test, total_test = check_predictions(pred_test, test_y)
    	pred_training = part2(W3, training_x, b)
    	num_correct_training, total_training = check_predictions(pred_training, training_y)
    	percentage_correct_test = (float(num_correct_test)/float(total_test))*100
    	percentage_correct_training = (float(num_correct_training)/float(total_training))*100
        performance_training.append(percentage_correct_training)
    	performance_test.append(percentage_correct_test)
    	print("The percentage of correct predictions on the training set of size "+ str(size) + " is:" + str(percentage_correct_training) + "%")
    	print("The percentage of correct predictions on the test set after training on a set of size " + str(size) + " is:" + str(percentage_correct_test) + "%")
    plt.plot(training_size, performance_training, color = 'g', linewidth=1, marker = 'o', label = 'Training Set Performance')
    plt.plot(training_size, performance_test, color = 'r', linewidth=1, marker = 'o', label = 'Test Set Performance')
    plt.title('Training Size vs. Percentage correct predictions on some datasets')
    plt.ylim([80, 110])
    plt.xlabel('Training size')
    plt.ylabel('Percentage of correct predictions')
    plt.legend()
    plt.show()
    np.save("W_part5", W3)
    np.save("b_part5", b)
def part6a():
    weights = np.load("W_part5.npy")
    #snapshot = cPickle.load(open("snapshot50.pkl"))
    #W0 = snapshot["W0"]
    #W1 = snapshot["W1"]
    #weights = dot(W0, W1).T
    W = weights.copy()
    b = np.load("b_part5.npy")
    b = b[:, 0:5000]
    #b = np.zeros((10, 5000))
    test_x, test_y =  get_sets('test', 500)
    w1s = np.arange(-1, 1, 0.1)
    w2s = np.arange(-1, 1, 0.1)
    w1z, w2z = np.meshgrid(w1s, w2s)
    C = np.zeros([w1s.size, w2s.size])
    z = 0
    for i, w1 in enumerate(w1s):
        for j, w2 in enumerate(w2s):
            W[5, 372] = weights[5, 372] + w1s[i]
            W[5, 412] = weights[5, 412] + w2s[j]
            p1 = part2(W, test_x, b)
            C[i,j] = cost_function(test_y.T, p1)
            print z
            z+= 1
    plt.contour(w1z, w2z, C, 50)
    plt.ylabel('W1')
    plt.xlabel('W2')
    plt.show()
def part6bc():
    weights = np.load("W_part5.npy")
    W_van = weights.copy()
    W_mom = weights.copy()
    b = np.load("b_part5.npy")
    b0 = b[:, 0:5000]
    b1 = b[:, 0:5000]
    b2 = b[:, 0:5000]
    test_x, test_y =  get_sets('test', 500)
    W_van[5, 372] = W_van[5, 372] -2.5 
    W_van[5, 412] = W_van[5, 412] -2.5
    W_mom[5, 372] = W_mom[5, 372] -2.5
    W_mom[5, 412] = W_mom[5, 412] -2.5
    gd_traj = [(W_van[5, 372], W_van[5, 412])]
    mo_traj = [(W_mom[5, 372], W_mom[5, 412])]
    print W_van.shape
    for i in range(30):
        W_temp_1, b1 = grad_descent(test_x, test_y.T, b0, W_van, 1e-3, 20)
        W_temp_2, b2 = grad_descent_mom(test_x, test_y.T, b0, W_mom, 1e-4, 20)
        W_van[5, 372] = W_temp_1[5, 372] 
        W_van[5, 412] = W_temp_1[5, 412]
        W_mom[5, 372] = W_temp_2[5, 372] 
        W_mom[5, 412] = W_temp_2[5, 412] 
        gd_traj.append((W_van[5, 372], W_van[5, 412]))
        mo_traj.append((W_mom[5, 372], W_mom[5, 412]))
        print i
    W = weights.copy()
    w1s = np.arange(-3, 3, 0.3)
    w2s = np.arange(-3, 3, 0.3)
    w1z, w2z = np.meshgrid(w1s, w2s)
    C = np.zeros([w1s.size, w2s.size])
    for i, w1 in enumerate(w1s):
        for j, w2 in enumerate(w2s):
            W[5, 372] = weights[5, 372] + w1s[i]
            W[5, 412] = weights[5, 412] + w2s[j]
            p1 = part2(W, test_x, b0)
            C[i,j] = cost_function(test_y.T, p1)
    CS = plt.contour(w1z, w2z, C)
    plt.plot([a for a, b in gd_traj], [b for a,b in gd_traj], 'yo-', label="No Momentum")
    plt.plot([a for a, b in mo_traj], [b for a,b in mo_traj], 'go-', label="Momentum")
    plt.legend()
    plt.ylabel('W1')
    plt.xlabel('W2')
    plt.show()
def part6e():
    weights = np.load("W_part5.npy")
    W_van = weights.copy()
    W_mom = weights.copy()
    b = np.load("b_part5.npy")
    b0 = b[:, 0:5000]
    b1 = b[:, 0:5000]
    b2 = b[:, 0:5000]
    test_x, test_y =  get_sets('test', 500)
    W_van[5, 372] = W_van[5, 372] -2.5 
    W_van[5, 412] = W_van[5, 412] -2.5
    W_mom[5, 372] = W_mom[5, 372] -2.5
    W_mom[5, 412] = W_mom[5, 412] -2.5
    gd_traj = [(W_van[5, 372], W_van[5, 412])]
    mo_traj = [(W_mom[5, 372], W_mom[5, 412])]
    print W_van.shape
    for i in range(30):
        W_temp_1, b1 = grad_descent(test_x, test_y.T, b0, W_van, 1e-4, 40)
        W_temp_2, b2 = grad_descent_mom(test_x, test_y.T, b0, W_mom, 1e-4, 40)
        W_van[5, 372] = W_temp_1[5, 372] 
        W_van[5, 412] = W_temp_1[5, 412]
        W_mom[5, 372] = W_temp_2[5, 372] 
        W_mom[5, 412] = W_temp_2[5, 412] 
        gd_traj.append((W_van[5, 372], W_van[5, 412]))
        mo_traj.append((W_mom[5, 372], W_mom[5, 412]))
        print i
    W = weights.copy()
    w1s = np.arange(-3, 3, 0.3)
    w2s = np.arange(-3, 3, 0.3)
    w1z, w2z = np.meshgrid(w1s, w2s)
    C = np.zeros([w1s.size, w2s.size])
    for i, w1 in enumerate(w1s):
        for j, w2 in enumerate(w2s):
            W[5, 372] = weights[5, 372] + w1s[i]
            W[5, 412] = weights[5, 412] + w2s[j]
            p1 = part2(W, test_x, b0)
            C[i,j] = cost_function(test_y.T, p1)
    CS = plt.contour(w1z, w2z, C)
    plt.plot([a for a, b in gd_traj], [b for a,b in gd_traj], 'yo-', label="No Momentum")
    plt.plot([a for a, b in mo_traj], [b for a,b in mo_traj], 'go-', label="Momentum")
    plt.legend()
    plt.ylabel('W1')
    plt.xlabel('W2')
    plt.show()
if __name__ == "__main__":
    if sys.argv[1] == "check3bcorrect":
        check_3b_correct()
    if sys.argv[1] == "part4": 
        part4()
    if sys.argv[1] == "part5":
        part5()
    if sys.argv[1] == "part6a":
        part6a()
    if sys.argv[1] == "part6bc":
        part6bc()
    if sys.argv[1] == "part6e":
        part6e()