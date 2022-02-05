# classify.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set

return - a list containing predicted labels for dev_set
"""
import math
import numpy as np

def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # return the trained weight and bias parameters
    #(len(train_labels))
    # list of y_hat's, size is the same as train_labels (7500)
    len_trainset = len(train_set)       # 7500
    len_image = len(train_set[0])       # length of one image which is 3072 (32x32x3)
    w = np.zeros((len_image)+1)               # weight array of size 3072 + 1 for the bias value
    alpha = learning_rate
    iter_num = max_iter                 # initiate iter_num to max number of iteration allowed
    
    #while iter_num != 0:                # iterate training until iter_num == 0
    for iter in range(max_iter):    
        for img_num in range(len_trainset):         # loop through each image in train_set
            x = train_set[img_num]      # set x to be current image with length 3072
            x = np.append(x,1)          # add element for bias, increasing the length to 3073
            
            if train_labels[img_num] == True:
                y = 1
            else:
                y = -1
            
            y_hat = np.dot(w,x)         # get the sum of all w0x0+w1x1+...+wnxn
            y_hat = int(np.sign(y_hat)) # convert to integer -1 or 1, to avoid nan values when subtracting with y (which is only 1 or -1)
            if (y == 1 and np.sign(y_hat) ==1 ) or ( y == -1 and np.sign(y_hat) <= 0):                  # if correct label is positive/True
                #if y == np.sign(y_hat): # if correct label and predicted label are both positive
                #    pass                # dont do anything
                pass
                # else:                   # if correct label is negative
                #     coef_x = (y-y_hat)*(alpha*0.5)*x   # if correct label is positive but predicted label is negative, divide by 2 to compesate for 2 and -2
                #     w = np.add(w, coef_x)              # update weight as necessary
                
            else:                                               # if correct label is negative
                # if np.sign(y_hat) == 0 or np.sign(y_hat) == -1: # if predicted label is negative or 0  
                #     pass                                        # dont do anything if theyre the same
                # else:                                           # if correct label and predicted label is positive
                coef_x = ((y)*(alpha))*x           # update weight,  ## divide by 2 to compesate for 2 and -2
                w = np.add(w, coef_x)
        
        #iter_num -= 1   # decrement iter_num
        
    b = float(w[len_image])   # extract b from the end of the w matrix
    w = np.delete(w, len_image) # delete the matrix location of b, to cut the dimension back to 3072
    #print(w, b)
    return w, b

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train perceptron model and return predicted labels of development set
    w,b = trainPerceptron(train_set, train_labels, learning_rate, max_iter)
    
    dev_length = len(dev_set)
    labels = []

    for img_num in range(dev_length):
        x = dev_set[img_num]

        y_hat = np.dot(w,x) + b
        #print(y_hat)
        if np.sign(y_hat) > 0 :
            labels.append(1)
        else:
            labels.append(0)
            
    #print(labels)
    #print(b)
    return labels
    

def classifyKNN(train_set, train_labels, dev_set, k):
    # TODO: Write your code here
    #print(k)
    len_devset = len(dev_set)
    len_trainset = len(train_set)
    #print(len_trainset)
    
    pred_label = []
    for img in range(len_devset):
        dist_dict = {}
        labels_list = []
        for img_num in range(len_trainset):
            dist_dict[img_num] = np.linalg.norm(train_set[img_num] - dev_set[img])
        
        sorted_dict = dict(sorted(dist_dict.items(), key=lambda item: item[1]))
        k_neighbors = list(sorted_dict)[:k]
        
        for index in k_neighbors:
            labels_list.append(train_labels[index])
        #fake = [True, False, False, True]
        max_label = max(set(labels_list), key = labels_list.count)
        pred_label.append(max_label)
    
    return pred_label
