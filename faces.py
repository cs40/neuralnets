from torch.autograd import Variable
import torch
import numpy as np
import matplotlib.pyplot as plt

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
from Digits import check_predictions 

from scipy.io import loadmat




def get_sets_multiclass(actors, training_size, test_size, validation_size):
    X = np.empty([0, 1024])
    Y = np.empty([0, len(actors)])
    training_X = []
    test_X = []
    validation_X = []
    training_Y = []
    test_Y = []
    validation_Y = []
    label_idx = 0
    total_size = training_size + test_size + validation_size
    training_range = [i for i in range(60)]
    test_range = [i for i in range(70,90)]
    validation_range = [i for i in range(60,69)]
    random.shuffle(training_range)
    random.shuffle(test_range)    
    random.shuffle(validation_range)
    #print training_range

    for bin in actors:
        for act in bin:
            j = 0
            i = 0
            while (j < training_size):
        
                filename = act + str(training_range[i]) + '.jpg'
                filename1 = act + str(training_range[i]) + '.jpeg'
                filename2 = act + str(training_range[i]) + '.JPG'
                filename3 = act + str(training_range[i]) + '.png'
                files = [filename, filename1, filename2, filename3]
                for filename in files:
                    if os.path.isfile("training/" + filename):
                        imarray = imread("training/" + filename)
                        X = vstack((X, reshape(np.ndarray.flatten(imarray), [1,1024])))
                        break
                new_label = np.random.rand(1, len(actors))*(0)
                new_label[0][label_idx] = 1
                Y = vstack((Y, new_label))
                i+=1
                j+=1


            i = 0
            while (j < training_size + validation_size):
                filename = act + str(validation_range[i]) + '.jpg'
                filename1 = act + str(validation_range[i]) + '.jpeg'
                filename2 = act + str(validation_range[i]) + '.JPG'
                filename3 = act + str(validation_range[i]) + '.png'
                files = [filename, filename1, filename2, filename3]
                for filename in files:
                    if os.path.isfile("validation/" + filename):
                        imarray = imread("validation/" + filename)
                        X = vstack((X, reshape(np.ndarray.flatten(imarray), [1,1024])))
                        break
                new_label = np.random.rand(1, len(actors))*(0)
                new_label[0][label_idx] = 1
                Y = vstack((Y, new_label))
                i+=1
                j+=1

            i = 0
            while (j< training_size + test_size + validation_size):
                filename = act + str(test_range[i]) + '.jpg'
                filename1 = act + str(test_range[i]) + '.jpeg'
                filename2 = act + str(test_range[i]) + '.JPG'
                filename3 = act + str(test_range[i]) + '.png'
                files = [filename, filename1, filename2, filename3]
                for filename in files:
                    if os.path.isfile("test/" + filename):
                        imarray = imread("test/" + filename)
                        X = vstack((X, reshape(np.ndarray.flatten(imarray), [1,1024])))
                        break
                new_label = np.random.rand(1, len(actors))*(0)
                new_label[0][label_idx] = 1
                Y = vstack((Y, new_label))
                i+=1
                j+=1

        label_idx += 1

    X /=255
    for bin in actors:
        for i in range(len(bin)):
            training_X.extend(X[:training_size])
            validation_X.extend(X[training_size:validation_size+training_size])
            test_X.extend(X[validation_size+training_size:total_size])
            X = X[total_size:]

    for i in range(len(actors)):
        training_Y.extend(Y[i*total_size : i*total_size+training_size])
        validation_Y.extend(Y[i*total_size+training_size : i*total_size+training_size+validation_size])
        test_Y.extend(Y[i*total_size+training_size+validation_size : (i+1)*(total_size)])
    

    return np.array(training_X), np.array(test_X), np.array(validation_X), np.array(training_Y), np.array(test_Y), np.array(validation_Y)

def shuffle_training(training_x, training_y, actors):
    X = np.empty([0, 1024])
    Y = np.empty([0, len(actors)])

    training_range = [i for i in range(348)]

    random.shuffle(training_range)

    for i in training_range:

        X = vstack((X, training_x[i]))
        Y = vstack((Y, training_y[i]))
    return X,Y
def check_predictions(pred, y):
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


def part_8():
    act = [['butler'], ['vartan'], ['radcliffe'],['baldwin'], ['hader'], ['carell']]
    training_x, t_x, valid_x, training_y, t_y, valid_y = get_sets_multiclass(act, 60, 20, 9)
    dim_x = 32*32
    dim_h = 20
    dim_out = 6

    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor

    model = torch.nn.Sequential(
    torch.nn.Linear(dim_x, dim_h),
    torch.nn.ReLU(),
    torch.nn.Linear(dim_h, dim_out),
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    minibatch_size = 30
    learning_rate = 1e-4    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch = []
    validation_set = []
    training_set = []

    for t in range(100):
        train_x, train_y = shuffle_training(training_x, training_y, act)
        for i in range(0, train_x.shape[0], minibatch_size):
            x = Variable(torch.from_numpy(train_x[i:i + minibatch_size]), requires_grad=False).type(dtype_float)
            y_classes = Variable(torch.from_numpy(np.argmax(train_y[i:i + minibatch_size], 1)), requires_grad=False).type(dtype_long)
    	    for j in range(1000):
        		y_pred = model(x)
        		loss = loss_fn(y_pred, y_classes)	    
        		model.zero_grad()  # Zero out the previous gradient computation
        		loss.backward()    # Compute the gradientprint "this is"
        		optimizer.step()
                
                # print model[0].weight.data.numpy().shape
                  # Use the gradient information to 
        		                       # make a step
        #track epoch number
        epoch.append(t)
        print t
        #performance on training
        train_x = Variable(torch.from_numpy(train_x), requires_grad=False).type(dtype_float)
        y_pred_training = model(train_x).data.numpy()
        num_correct, total = check_predictions(y_pred_training, train_y)
        train_pct = float(num_correct)/ float(total) * 100
        training_set.append(train_pct)
        
        #performance on validation
        validate_x = Variable(torch.from_numpy(valid_x), requires_grad=False).type(dtype_float)
        y_pred_validation = model(validate_x).data.numpy()
        valid_correct, valid_total = check_predictions(y_pred_validation, valid_y)
        valid_pct = float(valid_correct)/ float(valid_total) * 100
        validation_set.append(valid_pct)

    #performance on test set after all the epochs completed
    testing_x = Variable(torch.from_numpy(t_x), requires_grad=False).type(dtype_float)
    y_pred_test = model(testing_x).data.numpy()
    test_correct, test_total = check_predictions(y_pred_test, t_y)
    test_pct = float(test_correct)/ float(test_total) * 100
    print "The performance on the test set is: " + str(test_pct) + "%."
    
    #plot
    plt.plot(epoch, training_set, color = 'g', linewidth=1, marker = 'o', label = 'Training Set Performance')
    plt.plot(epoch, validation_set, color = 'r', linewidth=1, marker = 'o', label = 'Validation Set Performance')
    plt.title('Number of Epochs vs. Percentage correct predictions on some datasets')
    plt.ylim([50, 110])
    plt.xlabel('Number of Epochs')
    plt.ylabel('Percentage of correct predictions')
    plt.legend()
    plt.show()

    W = model[0].weight.data.numpy()
    b = model[0].bias.data.numpy()
    actors = ["butler", "radcliffe"]
    dict = {"butler": [], "radcliffe": []}
    for i in range(20):
        dict["butler"].append(0)
        dict["radcliffe"].append(0)
    for act in actors:
        for i in range(60):
            filename = act + str(i) + '.jpg'
            filename1 = act + str(i) + '.jpeg'
            filename2 = act + str(i) + '.JPG'
            filename3 = act + str(i) + '.png'
            files = [filename, filename1, filename2, filename3]
            for filename in files:
                if os.path.isfile("training/" + filename):
                    img = imread("training/" + filename)
                    img = reshape(np.ndarray.flatten(img), [1,1024])
                    img=img/255.0
                    for j in range(len(W)):
                        o = np.dot(W[j], img.T) + b[j]
                        if o[0] > 0:
                            dict[act][j]+=o[0]

    
    max_neuron1 = argmax(dict["butler"])
    max_neuron2 = argmax(dict["radcliffe"])
    
    print max_neuron1
    print max_neuron2
                
    fig = plt.figure(figsize = (50, 50))
    fig.add_subplot(1, 2, 1)
    plt.imshow(W[max_neuron1].reshape((32, 32)), cmap=plt.cm.coolwarm)
    fig.add_subplot(1, 2, 2)
    plt.imshow(W[max_neuron2].reshape((32, 32)), cmap=plt.cm.coolwarm)
    plt.show()

    
    #plt.imshow(model[0].weight.data.numpy()[12, :].reshape((32, 32)), cmap=plt.cm.coolwarm)
    #plt.show()





part_8()




