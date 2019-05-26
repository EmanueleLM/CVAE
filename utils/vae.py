# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 09:36:16 2019

@author: Emanuele
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scistats
import sys
import tensorflow as tf

sys.path.append('../utils')
import utils_dataset as utils

   
def vae(filename, 
         sequence_len,
         stride,
         batch_size, 
         cnn_sizes,
         cnn_activations,
         cnn_pooling,
         cnn_l_rate,  # used with "auxiliary" loss
         global_features,
         vae_hidden_size,
         tstud_degrees_of_freedom,
         l_rate,
         non_train_percentage, 
         training_epochs, 
         l_rate_test,
         val_rel_percentage,
         normalize, 
         time_difference,
         td_method,
         stop_on_growing_error=False,
         stop_valid_percentage=1.):
    
    # reset computational graph
    tf.reset_default_graph()
        
    # data parameters
    batch_size = 1
    sequence_len = 35
    stride = 10
    
    # training epochs
    epochs = 100
    
    # define VAE parameters
    learning_rate_elbo = 5e-2
    vae_hidden_size = 4
    tstud_degrees_of_freedom = 3.
    sigma_threshold_elbo = 1e-3
       
    # number of sampling per iteration
    samples_per_iter = 1
    
    # early-stopping parameters
    stop_on_growing_error = True  # early-stopping enabler
    stop_valid_percentage = .5  # percentage of validation used for early-stopping    
    min_loss_improvment = .005  # percentage of minimum loss' decrease (.01 is 1%)
    
    # define input/output pairs
    input_ = tf.placeholder(tf.float32, [None, sequence_len, batch_size])  # (batch, input, time)
    target = tf.placeholder(tf.float32, [None, batch_size])  # (batch, output)
    
    # parameters' initialization
    vae_encoder_shape_weights = [batch_size*sequence_len, vae_hidden_size*2]
    vae_decoder_shape_weights = [vae_hidden_size, batch_size*sequence_len]
    
    zip_weights_encoder = zip(vae_encoder_shape_weights[:-1], vae_encoder_shape_weights[1:])
    
    weights_vae_encoder = [tf.Variable(tf.truncated_normal(shape=[shape,
                                                                  next_shape])) for (shape, next_shape) in zip_weights_encoder]
    bias_vae_encoder = [tf.Variable(tf.truncated_normal(shape=[shape])) for shape in vae_encoder_shape_weights[1:]]
    
    zip_weights_decoder = zip(vae_decoder_shape_weights[:-1], vae_decoder_shape_weights[1:])
    weights_vae_decoder = [tf.Variable(tf.truncated_normal(shape=[shape,
                                                                  next_shape])) for (shape, next_shape) in zip_weights_decoder]
    bias_vae_decoder = [tf.Variable(tf.truncated_normal(shape=[shape])) for shape in vae_decoder_shape_weights[1:]]
    
    #
    # VAE graph's definition
    flattened_input = tf.layers.flatten(input_)
    
    vae_encoder = tf.matmul(flattened_input, weights_vae_encoder[0]) + bias_vae_encoder[0]
    
    for (w_vae, b_vae) in zip(weights_vae_encoder[1:], bias_vae_encoder[1:]):
        
        vae_encoder = tf.nn.leaky_relu(vae_encoder)
        vae_encoder = tf.matmul(vae_encoder, w_vae) + b_vae
    
    # means and variances' vectors of the learnt hidden distribution
    #  we assume the hidden gaussian's variances matrix is diagonal
    loc = tf.slice(tf.nn.relu(vae_encoder), [0, 0], [-1, vae_hidden_size])
    loc = tf.squeeze(loc, axis=0)
    scale = tf.slice(tf.nn.softplus(vae_encoder), [0, vae_hidden_size], [-1, vae_hidden_size])
    scale = tf.squeeze(scale, 0)
    
    vae_hidden_distr = tf.contrib.distributions.MultivariateNormalDiag(loc, scale)  
    vae_hidden_state = tf.reduce_mean([vae_hidden_distr.sample() for _ in range(samples_per_iter)], axis=0)
    
    # probability of the *single* sample (no multisampling) --> used in test phase
    vae_hidden_pdf = vae_hidden_distr.prob(vae_hidden_distr.sample())
    
    feed_decoder = tf.reshape(vae_hidden_state, shape=(-1, vae_hidden_size))
    vae_decoder = tf.matmul(feed_decoder, weights_vae_decoder[0]) + bias_vae_decoder[0]
    vae_decoder = tf.nn.leaky_relu(vae_decoder)    
    
    for (w_vae, b_vae) in zip(weights_vae_decoder[1:], bias_vae_decoder[1:]):
        
        vae_decoder = tf.matmul(vae_decoder, w_vae) + b_vae
        vae_decoder = tf.nn.leaky_relu(vae_decoder)
    
    # time-series reconstruction and ELBO loss
    vae_reconstruction = tf.contrib.distributions.StudentT(tstud_degrees_of_freedom,
                                                           tf.constant(np.zeros(batch_size*sequence_len, dtype='float32')),
                                                           tf.constant(np.ones(batch_size*sequence_len, dtype='float32')))
    likelihood = vae_reconstruction.log_prob(flattened_input)
    
    prior = tf.contrib.distributions.MultivariateNormalDiag(tf.constant(np.zeros(vae_hidden_size, dtype='float32')),
                                                            tf.constant(np.ones(vae_hidden_size, dtype='float32')))
    
    divergence = tf.contrib.distributions.kl_divergence(vae_hidden_distr, prior)
    elbo = tf.reduce_mean(likelihood - divergence)
    
    optimizer_elbo = tf.train.AdamOptimizer(learning_rate_elbo).minimize(-elbo)  

    # extract train and test
    x_train, y_train, x_valid, y_valid, x_test, y_test = utils.generate_batches(
                                                             filename='../data/sin.csv', 
                                                             window=sequence_len,
                                                             stride=stride,
                                                             mode='validation', 
                                                             non_train_percentage=.5,
                                                             val_rel_percentage=.5,
                                                             normalize='maxmin-11',
                                                             time_difference=True,
                                                             td_method=None)
       
    # suppress second axis on Y values (the algorithms expects shapes like (n,) for the prediction)
    y_train = y_train[:,0]; y_valid = y_valid[:,0]; y_test = y_test[:,0]
        
    # if the dimensions mismatch (somehow, due tu bugs in generate_batches function,
    #  make them match)
    mismatch = False
    
    if len(x_train) > len(y_train):
        
        x_train = x_train[:len(y_train)]
        mismatch = True
    
    if len(x_valid) > len(y_valid):
        
        x_valid = x_valid[:len(y_valid)]
        mismatch = True
    
    if len(x_test) > len(y_test):
        
        x_test = x_test[:len(y_test)]
        mismatch = True
    
    if mismatch is True: 
        
        print("Mismatched dimensions due to generate batches: this will be corrected automatically.")
        
    print("Datasets shapes: ", x_train.shape, y_train.shape, x_valid.shape, y_valid.shape, x_test.shape, y_test.shape)
    
    # train + early-stopping
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        
        sess.run(init)
        
        # train
        last_error_on_valid = np.inf
        current_error_on_valid = .0
        e = 0
        
        while e < epochs:
            
            print("epoch", e+1)
            
            iter_ = 0
            
            while iter_ < int(np.floor(x_train.shape[0] / batch_size)):
        
                batch_x = x_train[iter_*batch_size: (iter_+1)*batch_size, :].T.reshape(1, sequence_len, batch_size)
        
                # run VAE encoding-decoding
                sess.run(optimizer_elbo, feed_dict={input_: batch_x})
                
                iter_ +=  1

            if stop_on_growing_error:

                current_error_on_valid = .0
                
                # verificate stop condition
                iter_val_ = 0
                while iter_val_ < int(stop_valid_percentage * np.floor(x_valid.shape[0] / batch_size)):
                    
                    batch_x_val = x_valid[iter_val_*batch_size: (iter_val_+1)*batch_size, :].T.reshape(1, sequence_len, batch_size)
                    batch_y_val = y_valid[np.newaxis, iter_val_*batch_size: (iter_val_+1)*batch_size]
                    
                    # accumulate error
                    current_error_on_valid +=  np.abs(np.sum(sess.run(-elbo, feed_dict={input_: batch_x_val, 
                                                                                        target: batch_y_val})))

                    iter_val_ += 1
                 
                # stop learning if the loss reduction is below 1% (current_loss/past_loss)
                if current_error_on_valid > last_error_on_valid or (np.abs(current_error_on_valid/last_error_on_valid) > 1-min_loss_improvment and e!=0):
            
                    if current_error_on_valid > last_error_on_valid:
                        
                        print("Loss function has increased wrt to past iteration.")
                    
                    else:
                        
                        print("Loss' decrement is below 1% (relative).")
                    
                    print("Stop learning at epoch ", e, " out of ", epochs)
                    e = epochs
                        
                last_error_on_valid = current_error_on_valid                

            
            e += 1
            
        # test
        y_test = y_test[:x_test.shape[0]]
        mean_elbo = .0
        std_elbo = 1.
        vae_anomalies = np.zeros(shape=(int(np.floor(x_test.shape[0] / batch_size))))
        
        threshold_elbo = scistats.t.pdf(mean_elbo-sigma_threshold_elbo, 
                                        df=tstud_degrees_of_freedom,
                                        loc=mean_elbo, 
                                        scale=std_elbo)
        
        iter_ = 0
        
        while iter_ < int(np.floor(x_test.shape[0] / batch_size)):
    
            batch_x = x_test[iter_*batch_size: (iter_+1)*batch_size, :].T.reshape(1, sequence_len, batch_size)
                        
            # get probability of the encoding           
            vae_anomalies[iter_] =  sess.run(vae_hidden_pdf, feed_dict={input_: batch_x})
                                       
            iter_ +=  1
            
    # plot vae likelihood values
    fig, ax1 = plt.subplots()
    ax1.plot(vae_anomalies, 'b', label='likelihood')
    ax1.set_ylabel('VAE: Anomalies likelihood')
    plt.legend(loc='best')
    
    # highlights elbo's boundary
    ax1.plot(np.array([threshold_elbo for _ in range(len(vae_anomalies))]), 'r', label='threshold')
    plt.legend(loc='best')
        
    fig.tight_layout()
    plt.show()   
    
    # plot the graph
    fig, ax1 = plt.subplots()
    ax1.plot(y_test, 'b', label='test set')
    ax1.set_ylabel('time')
    plt.legend(loc='best')
    
    fig.tight_layout()
    plt.show() 
    
