# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 14:57:21 2018

@author: Emanuele

Data utilities for TOPIX prediction
"""

import numpy as np
import pandas as pd
import scipy.stats as st
import tensorflow as tf


"""
 Measure randomness of a binary string.
 Takes as input:
     input_:numpy.array is the binary vector;
     significance:float, the significance of the test.
"""
def random_test(input_, significance=5e-2):
    
    input_len = len(input_)
    r = input_len
    
    ones = np.sum(input_)
    zeros = input_len - ones
    
    r_hat = 2*(ones*zeros)/(ones+zeros) + 1 
    s_r = ((2*ones*zeros)*(2*ones*zeros-ones-zeros))/((zeros+ones-1)*(ones+zeros)**2)
    
    z = (r - r_hat)/s_r
        
    # test is not random with this significance
    if np.abs(z) > st.norm.ppf(1-significance/2):
        
        return False  
    
    else:
        
        return True
    

"""
 Measure autocorrelation of a sequence with the lag test from Box and Jenkins, 1976.
 Takes as input:
     input_:numpy.array is the binary vector;
     lag:int, the time-lag used to measure autocorrelation of the input sequence.
     tolerance:float, the tolerance of the test.
"""
def autocorrelation_test(input_, lag, tolerance=1e-2):
    
    input_len = len(input_)
    mean = input_.mean()
    r_k_num = r_k_den = 0.
    
    for i in range(input_len-lag-1):
        
        r_k_num += (input_[i]-mean)*(input_[i+lag]-mean)
        r_k_den += (input_[i]-mean)**2
    
    print(np.abs(r_k_num/(r_k_den+1e-10)), tolerance/2)
    if np.abs(r_k_num/(r_k_den+1e-10)) <= tolerance/2:  # two tail test
        
        return True
    
    else:
        
        return False    
    

"""
 Gaussian pdf estimator.
 Takes as input:
     x:numpy.array, the input vector;
     mean:float, the mean of the gaussian distribution;
     variance:float, the variance of the gaussian distribution.
"""
def gaussian_pdf(x, mean, variance):
    
    denom = (2*np.pi*variance)**.5
    num = np.exp(-(float(x)-float(mean))**2/(2*variance))
    
    return num/denom


"""
 Turn a series into a matrix (i.e. repeated batches).
"""
def series_to_matrix(series, k_shape, striding=1):
    
    res = np.zeros(shape=(int((series.shape[0] - k_shape) / striding) + 1,
                          k_shape)
                   )
    j = 0
    for i in range(0, series.shape[0] - k_shape + 1, striding):
        res[j] = series[i:i + k_shape]
        j += 1

    return res


"""
 Generate batches from a .cvs file.
 Takes as input:
     filename:string, path to the .csv file where data is stored;
     window:int, size of each sample (i.e. a sample at time t is generated as y(t:t+window));
     stride:int, number of timesteps skipped after the generation of each sample;
     mode:string, available split techniques:
        'train': all data is reserved to train;
        'train-test': split between train and test, according to non_train_percentage;
        'validation': data is split among train, test and validation: their percentage is chosen according to the percantge
                      of data that has not been included in train (1-non_train_percentage) and assigned to validation
                      proportionally to val_rel_percentage.;
        'strided-validation': you are likely to use this one for experiments,
                              data is split among train, test and validation: their percentage is chosen according to the percantge
                              of data that has not been included in train (1-non_train_percentage) and assigned to validation
                              proportionally to val_rel_percentage.;
     non_train_percentage:float, given the entire dataset, the percentage of data not used for train;
     val_rel_percentage:float, percentage (relative to non_train_percentage) of dataset used for validation.
         This variable is considered if and only if mode is 'validation';
     normalize:string, available normalization techniques:
         'maxmin01': normalize in the range [0,1];
         'maxmin-11': normalize in the range [-1,1].;
         'gaussian': normalize (as gaussian with mean .0 and standard deviation 1.);
         'haar': Haar-Wavelet normalization;
         'haar01': same as 'haar' but whit previous [0,1] normalization;
         'haar-11': same as 'haar' but whit previous [-1,1] normalization.        
     time-difference:boolean, specify whether time difference techniques are used;
     td_method:function, specify which function is used to perform time-difference on data.
          This variable is considered if and only if time_difference is True;
     subsampling:integer, downsampling frequence: i.e. how many samples are skipped for one that is taken;
     rounding:int, set max number of digits to represent each datum.
 Returns:
     dataset:numpy.array, the data ready to be used.
"""
def generate_batches(filename, 
                     window,
                     stride=1,
                     mode='train-test', 
                     non_train_percentage=.7, 
                     val_rel_percentage=.5,
                     normalize='maxmin01',
                     time_difference=False,
                     td_method=None,
                     subsampling=1,
                     rounding=None):

    data = pd.read_csv(filename, delimiter=',', header=0)
    data = (data.iloc[:, 0]).values
    
    # downsampling
    if int(subsampling) > 1:
        
        data = data[::int(subsampling)]

    # normalize dataset (max-min method)
    if normalize == 'maxmin01':
        
        data = (data-np.min(data))/(np.max(data)-np.min(data))
    
    elif normalize == 'maxmin-11':
    
        avg = (np.mean(data)-np.min(data))/2
        data = (data-avg)/avg
        
    elif normalize == 'gaussian':
        
        data = (data-np.mean(data))/np.std(data)
        
    # with Haar transformation, it is better to use a window of even size
    elif normalize == 'haar':
        
        data = series_to_matrix(data, 2, 1)
        data = np.dot(np.array([[1,1],[1,-1]]), data.T)/2**.5
        data = np.ravel(data) 
        
    elif normalize == 'haar01':

        data = (data-np.min(data))/(np.max(data)-np.min(data))
        data = series_to_matrix(data, 2, 1)
        data = np.dot(np.array([[1,1],[1,-1]]), data.T)/2**.5
        data = np.ravel(data) 
        
    elif normalize == 'haar-11':
        
        avg = (np.mean(data)-np.min(data))/2
        data = (data-avg)/avg
        data = series_to_matrix(data, 2, 1)
        data = np.dot(np.array([[1,1],[1,-1]]), data.T)/2**.5
        data = np.ravel(data) 
    
    else:
        
        print("Dataset is not normalized.")

    if rounding != None:
        
        data = np.round(data, int(rounding))
                
    # if the flag 'time-difference' is enabled, turn the dataset into the variation of each time 
    #  step with the previous value (loose the firt sample)
    if time_difference is True:
        
        if td_method is None:
            
            data = data[1:] - data[:-1]
        
        else:
            
            data = td_method(data+1e-5)
            data = data[1:] - data[:-1]
        
    if mode == 'train':

        y = series_to_matrix(data[window:], 1, stride)
        x = series_to_matrix(data, window, stride)
        
        if stride == 1 or window == 1:
            
            x = x[:-1]

        return x, y

    elif mode == 'train-test':

        train_size = int(np.ceil((1 - non_train_percentage) * len(data)))
        train = data[:train_size]; test = data[train_size:]
        
        y_train = series_to_matrix(train[window:], 1, stride)
        x_train = series_to_matrix(train, window, stride)       
        
        y_test = series_to_matrix(test[window:], 1, striding=1)
        x_test = series_to_matrix(test, window, striding=1)
        
        if stride == 1 or window == 1:
            
            x_train = x_train[:-1]; x_test = x_test[:-1]

        return x_train, y_train, x_test, y_test

    elif mode == 'validation':

        # split between train and validation+test
        train_size = int(np.ceil((1 - non_train_percentage) * len(data)))
        train = data[:train_size]
        
        y_train = series_to_matrix(train[window:], 1, stride)
        x_train = series_to_matrix(train, window, stride)

        # split validation+test into validation and test: no stride is applied
        validation_size = int(val_rel_percentage * np.ceil(len(data) * non_train_percentage))
        val = data[train_size:validation_size+train_size]; test = data[validation_size+train_size:]

        y_val = series_to_matrix(val[window:], 1, striding=1)
        x_val = series_to_matrix(val, window, striding=1)
        
        y_test = series_to_matrix(test[window:], 1, striding=1)
        x_test = series_to_matrix(test, window, striding=1)
                
        if stride == 1 or window == 1:
            
            x_train = x_train[:-1]; x_test = x_test[:-1]; x_val = x_val[:-1]

        return x_train, y_train, x_val, y_val, x_test, y_test

    elif mode == 'strided-validation':

        # split between train and validation+test
        train_size = int(np.ceil((1 - non_train_percentage) * len(data)))
        train = data[:train_size]
        
        y_train = series_to_matrix(train[window:], 1, stride)
        x_train = series_to_matrix(train, window, stride)

        # split validation+test into validation and test: stride *is* applied
        validation_size = int(val_rel_percentage * np.ceil(len(data) * non_train_percentage))
        val = data[train_size:validation_size+train_size]; test = data[validation_size+train_size:]

        y_val = series_to_matrix(val[window:], 1, striding=stride)
        x_val = series_to_matrix(val, window, striding=stride)
        
        y_test = series_to_matrix(test[window:], 1, striding=stride)
        x_test = series_to_matrix(test, window, striding=stride)
                
        if stride == 1 or window == 1:
            
            x_train = x_train[:-1]; x_test = x_test[:-1]; x_val = x_val[:-1]

        return x_train, y_train, x_val, y_val, x_test, y_test
 
"""
 Define lstm for all the /lstm modules
"""
def lstm_exp(filename, 
             window,
             stride,
             batch_size, 
             lstm_params,
             lstm_activation,
             l_rate,
             non_train_percentage, 
             training_epochs, 
             l_rate_test,
             val_rel_percentage,
             normalize, 
             time_difference,
             td_method,
             stop_on_growing_error=False,
             stop_valid_percentage=1.,
             verbose=True):
            
    # clear computational graph
    tf.reset_default_graph()

    # define LSTM features: time steps/hidden layers
    batch_size = batch_size  # length of LSTM networks (n. of LSTM)
    
    # size of each input
    window = window

    # create input,output pairs
    X, Y, X_val, Y_val, X_test, Y_test = generate_batches(filename=filename, 
                                                          window=window, 
                                                          stride=stride,
                                                          mode='strided-validation',
                                                          non_train_percentage=non_train_percentage,
                                                          val_rel_percentage=val_rel_percentage,
                                                          normalize=normalize,
                                                          time_difference=time_difference,
                                                          td_method=td_method)
    

    # suppress second axis on Y values (the algorithms expects shapes like (n,) for the prediction)
    Y = Y[:,0]; Y_val = Y_val[:,0]; Y_test = Y_test[:,0]
    
    # if the dimensions mismatch (somehow, due tu bugs in generate_batches function,
    #  make them match)
    mismatch = False
    
    if len(X) > len(Y):
        
        X = X[:len(Y)]
        mismatch = True
    
    if len(X_val) > len(Y_val):
        
        X_val = X_val[:len(Y_val)]
        mismatch = True
    
    if len(X_test) > len(Y_test):
        
        X_test = X_test[:len(Y_test)]
        mismatch = True
    
    if mismatch is True: 
        
        if verbose: 
        
            print("Mismatched dimensions due to generate batches: this will be corrected automatically.")
    
    if verbose: 
            
        print("Datasets shapes: ", X.shape, Y.shape, X_val.shape, Y_val.shape, X_test.shape, Y_test.shape)

    # final dense layer: declare variable shapes: weights and bias
    weights = tf.get_variable('weights', 
                              shape=[lstm_params[-1], batch_size, batch_size], 
                              initializer=tf.truncated_normal_initializer())
    bias = tf.get_variable('bias', 
                           shape=[1, batch_size], 
                           initializer=tf.truncated_normal_initializer())

    # placeholders (input)
    x = tf.placeholder("float", [None, batch_size, window]) # (batch, time, input)
    y = tf.placeholder("float", [None, batch_size])  # (batch, output)
    
    # define the LSTM cells
    cells = [tf.contrib.rnn.LSTMCell(lstm_params[i],                                   
                                     forget_bias=1.,
                                     state_is_tuple=True,
                                     activation=lstm_activation[i],
                                     initializer=tf.contrib.layers.xavier_initializer()) for i in range(len(lstm_params))]

    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)  
    outputs, _ = tf.nn.dynamic_rnn(multi_rnn_cell, 
                                   x,
                                   dtype="float32")
    
    # dense layer: prediction
    y_hat = tf.tensordot(tf.reshape(outputs, shape=(batch_size, lstm_params[-1])), weights, 2) + bias

    # estimate error as the difference between prediction and target
    error = y - y_hat
    
    # calculate loss
    loss = tf.nn.l2_loss(error)
    
    # optimization
    opt = tf.train.GradientDescentOptimizer(learning_rate=l_rate).minimize(loss)

    with tf.Session() as sess:
        
        init = tf.global_variables_initializer()
        
        sess.run(init)
    
        # train phase
        epochs = training_epochs
        plot_y = list()
        plot_y_hat = list()
        list_validation_error = list()
        list_test_error = list()
        
        # train: stop conditions (may not be evaluated)
        last_error_on_valid = np.inf
        current_error_on_valid = .0
        
        e = 0
        while e < (epochs + 1):
    
            iter_ = 0
            
            if verbose: 
        
                print("\n\nEpoch ", e + 1)
           
            if e < epochs - 2:                
                                           
                while iter_ < int(np.floor(X.shape[0] / batch_size)):
                    
                    batch_x = X[np.newaxis, iter_ * batch_size:batch_size * (iter_ + 1)]
                    batch_y = Y[np.newaxis, iter_ * batch_size:batch_size * (iter_ + 1)]
    
                    sess.run(opt, feed_dict={x: batch_x, y: batch_y})
                                               
                    iter_ = iter_ + 1
                
                if stop_on_growing_error:
    
                    current_error_on_valid = .0
                    
                    # verificate stop condition
                    iter_val_ = 0
                    while iter_val_ < int(stop_valid_percentage * np.floor(X_val.shape[0] / batch_size)):
                        
                        batch_x_val = X_val[np.newaxis, iter_val_ * batch_size:batch_size * (iter_val_ + 1)]
                        batch_y_val = Y_val[np.newaxis, iter_val_ * batch_size:batch_size * (iter_val_ + 1)]
                        
                        # accumulate error
                        current_error_on_valid +=  np.abs(np.sum(sess.run(error, feed_dict={x: batch_x_val, y: batch_y_val})))
    
                        iter_val_ += 1
                     
                    if verbose: 
        
                        print("Past error on valid: ", last_error_on_valid)
                        print("Current total error on valid: ", current_error_on_valid)
                    
                # stop learning if the loss reduction is below 0.5% (current_loss/past_loss)
                if current_error_on_valid > last_error_on_valid or (np.abs(current_error_on_valid/last_error_on_valid) > .995 and e!=0):
            
                    if current_error_on_valid > last_error_on_valid:
                        
                        if verbose: 
                            
                            print("Loss function has increased wrt to past iteration.")
                    
                    else:
                        
                        if verbose: 
                            
                            print("Loss' decrement is below 1% (relative).")
                            
                    if verbose:
                        
                        print("Stop learning at epoch ", e, " out of ", epochs)
                        
                    e = epochs - 3
                        
                last_error_on_valid = current_error_on_valid                
    
            # validation
            elif e == epochs - 2:
    
                if verbose:
                    
                    print(" Validation epoch:")
    
                iter_val_ = 0
                while iter_val_ < int(np.floor(X_val.shape[0] / batch_size)):
    
                    batch_val_x = X_val[np.newaxis, iter_val_ * batch_size:batch_size * (iter_val_ + 1)]
                    batch_val_y = Y_val[np.newaxis, iter_val_ * batch_size:batch_size * (iter_val_ + 1)]
    
                    # estimate validation error and append it to a list
                    list_validation_error.append(sess.run(error, feed_dict={x: batch_val_x, y: batch_val_y}))
    
                    iter_val_ += 1
    
            # test
            elif e == epochs - 1:
                 
                if verbose:
                    
                    print(" Test epoch:")
                
                iter_test_ = 0
                while iter_test_ < int(np.floor(X_test.shape[0] / batch_size)):
    
                    batch_test_x = X_test[np.newaxis, iter_test_ * batch_size:batch_size * (iter_test_ + 1)]
                    batch_test_y = Y_test[np.newaxis, iter_test_ * batch_size:batch_size * (iter_test_ + 1)]
    
                    pred_y = sess.run(y_hat, feed_dict={x: batch_test_x, y: batch_test_y})
                    plot_y_hat.append(pred_y)
                    plot_y.append(batch_test_y)
                    list_test_error.append(sess.run(error, feed_dict={x: batch_test_x, y: batch_test_y}))
    
                    iter_test_ += 1
            
            e += 1

    dict_results = {"Window_size": window,
                    "Batch_size": batch_size, "Learning_rate": l_rate,
                    "Y": plot_y, "Y_HAT": plot_y_hat,
                    "X_train": X, "Y_train": Y, "X_test": X_test, "Y_test": Y_test, "X_valid": X_val, "Y_valid": Y_val, 
                    "Validation_Errors": list_validation_error, "Test_Errors": list_test_error}
    return dict_results
    