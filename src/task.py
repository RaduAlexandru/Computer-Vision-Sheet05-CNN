#!/usr/bin/env python

import sys
import os
import time
import gzip

import numpy as np
import theano
import theano.tensor as T

import lasagne


# load_dataset loads the data from the file specified with filename
# @return a numpy 4-D array with the shape (examples, channels, rows, columns)
#         and a numpy 1-D array with the labels
# for USPS, images are 16x16 and grayscale, so there is only one channel
# for train, data has the shape (7291, 1, 16, 16)
# for test,  data has the shape (2007, 1, 16, 16)
def load_dataset(filename):

    # TODO: load dataset into data/labels
    #       hint: the data and labels container should be initialized like this:
    #             data = np.zeros((examples, 1, 16, 16), dtype=np.float32
    #             labels = np.zeros(examples, dtype=np.uint8)
    #             where examples is the number of training/test examples, respectively


    examples=7291+2007

    data = np.zeros((0, 1, 16, 16), dtype=np.float32)
    labels = np.zeros(0, dtype=np.uint8)
    number = np.zeros((1, 0, 16), dtype=np.float32)
    
    
    '''
    same_number=1
    index_global=0
    counter=0
    read_label=1
    with gzip.open(filename,'r') as fin:
        for line in fin:
            #print "got line", line
            
            index_global=index_global+1
            
            if read_label:
                #print "got label", line
                #Do stuff with label
                labels=np.append(labels,int (line))    
                
                #We will afterwards read a new number so we clear everything and append the current one
                if index_global!=1:
                    print "adding number to data with label",line
                    same_number=0
                    number=number.reshape(1,1,16,16)
                    data= np.append(data,number,axis=0)
                    
                print  number
                
                
                
                counter =counter +1
                read_label=0
                continue
            else:
                
                #do stuff with the line which is a part of the number
                if same_number==0:
                    #number.fill(0)
                    number = np.zeros((1, 0, 16), dtype=np.float32)

                    
                print "got line",counter , line
                    
                
                    
                #print "number_line strn format",line
                number_line= np.fromstring(line,sep=' ')
                number_line=number_line.reshape(1,1,16)
                #print "number_line array is",number_line
                #print "number line has shape" ,number_line.shape
                #print "number has shape" ,number.shape
                number=np.append(number,number_line, axis=1)
                
                
            
                counter =counter +1
                if counter==17:
                    counter=0
                    read_label=1
                continue
            
            
        
    '''
    number = np.zeros((1, 0, 16), dtype=np.float32)   
    
    
    start_new_number=1    
    counter=0
    with gzip.open(filename,'r') as fin:
        for line in fin:
            
            if counter %17 ==0:
                #print "got label" ,line
                labels=np.append(labels,int (line))    
                
            else:
                if start_new_number:
                    #print "number stored is", number
                    #print "we start a new number"
                    number = np.zeros((1, 0, 16), dtype=np.float32)   
                    start_new_number=0
                    
                    
                #print "got number_line", line
                
                number_line= np.fromstring(line,sep=' ')
                number_line=number_line.reshape(1,1,16)
                number=np.append(number,number_line, axis=1)
                
                #if the number is finished, add it to data
                if number.shape[1]==16:
                    #we finished with the number, put it in data
                    start_new_number=1
                    number=number.reshape(1, 1, 16, 16)
                    data= np.append(data,number,axis=0)
                
            
            counter=counter + 1
            
            
            
            
                        
#            if counter > 100:
#                break
    
 

        
    return data, labels


# @data the data to be normalized
# @return the normalized data
def normalize(data):
    
    # TODO: mean and variance normalization of data
    
    mean=np.mean (data)
    var=np.var (data)
    
#    print "mean of data is" , mean
#    print "var of data is" , var
    
    data=(data-mean)/ (np.sqrt(var))
    
#    mean=np.mean (data)
#    var=np.var (data)
#    
#    print "after mean of data is" , mean
#    print "after var of data is" , var

    return data


# build the CNN
# @return the network
def build_cnn(input_var=None):

    # TODO: build the CNN with
    #       * input layer
    #       * convolutional layer with 16 filters of size 3x3
    #       * max-pooling layer with pooling-size 2x2
    #       * convolutional layer with 16 filters of size 3x3
    #       * max-pooling layer with pooling-size 2x2
    #       * output layer with 10 units (one for each class)
    #         and softmax nonlinearity

    #input
    network = lasagne.layers.InputLayer(shape=(None, 1, 16, 16),
                                        input_var=input_var)
            
    #2 pais of convolution-pooling                            
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=16, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())        
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))  
                            
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=16, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    #output
    network = lasagne.layers.DenseLayer(
        network,
        num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax)
    

    return network



# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]



def main(num_epochs=20, needsNormalization=True):
    # Load the dataset
    print "Loading data..."
    train_data, train_labels = load_dataset('usps/train.gz')
    test_data, test_labels = load_dataset('usps/test.gz')


    
    print "train_data has dimensions",train_data.shape
    print "train_labels has dimensions", train_labels.shape
    
    print "test_data has dimensions", test_data.shape
    print "test_labels has dimensions", test_labels.shape

    # normalize the data
    if (needsNormalization):
        train_data = normalize(train_data)
        test_data = normalize(test_data)

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create CNN
    print "Building model and compiling functions..."
    network = build_cnn(input_var)

    # TODO: setup training criterion and loss functions
    #       * use categorical crossentropy loss
    #       * use SGD with Nesterov momentum 0.9 and learning rate 0.1 for optimization
    
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.1, momentum=0.9)


    # actual training
    print "Starting training..."
    # Create a loss expression for validation/testing
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)


    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
    
    # We iterate over epochs:
    for epoch in range(num_epochs):
        print "epoch",epoch
        train_err = 0
        train_batches = 0
        # TODO: process an epoch
        #       * use a minibach-size of 128
        #       * keep track of the training loss after each epoch and print it
        
        for batch in iterate_minibatches(train_data, train_labels, 128, shuffle=True):
            inputs, targets = batch
            inputs=inputs.astype(np.float32)
            targets=targets.astype(np.int32)
            
            train_err +=train_fn(inputs, targets)     
            train_batches += 1
        
        
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    # After training, we compute the test error
    # TODO: use the trained network to classify the test data
    #       * print the test loss
    #       * also print the test accuracy
    
    print "STARTING VALIDATION"
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(test_data, test_labels, 128, shuffle=False):
        inputs, targets = batch
        inputs=inputs.astype(np.float32)
        targets=targets.astype(np.int32)
        
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1

    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(
        val_acc / val_batches * 100))



### MAIN PROGRAM ###
print "\nRun network without data normalization...\n"
main(num_epochs=20, needsNormalization=False)

print "\nRun network with data normalization...\n"
main(num_epochs=20, needsNormalization=True)
