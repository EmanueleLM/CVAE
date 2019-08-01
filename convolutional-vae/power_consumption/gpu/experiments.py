# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 09:36:16 2019

@author: Emanuele
"""

import copy as cp
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scistats
import sys
import tensorflow as tf
import tensorflow_probability as tfp

sys.path.append('../../../utils')
import utils_dataset as utils


if __name__ == '__main__':
    
    # parameters of the model
    data_path = '../../../data/power_consumption.csv'
    sequence_len = 150
    batch_size = 1
    stride = 1
    num_conv_channels = 4  # convolutional channels
    
    # convolutional kernels + strides
    vae_encoder_shape_weights = [3, 3, 5]
    vae_decoder_shape_weights = [3, 3, 5]    
    vae_encoder_strides = [2, 2, 3]
    vae_decoder_strides = [2, 2, 3]  
    
    # produce a noised version of training data for each training epoch:
    #  the second parameter is the percentage of noise that is added wrt max-min of the time series'values
    make_some_noise = (False, 5e-2)  
    
    # for each training epoch, use a random value of stride between 1 and stride
    random_stride = False  
    vae_hidden_size = 1
    subsampling = 1
    elbo_importance = (.1, 1.)  # relative importance to reconstruction and divergence
    lambda_reg = (0e-3, 0e-3)  # elastic net 'lambdas', L1-L2
    rounding = None
    
    # maximize precision or F1-score over this vector
    sigma_threshold_elbo = [1e-2] # [i*1e-3 for i in range(1, 100, 10)]
    
    learning_rate_elbo = 1e-4
    vae_activation = tf.nn.relu6
    normalization = 'maxmin-11'
    
    # training epochs
    epochs = 100
       
    # number of sampling per iteration in the VAE hidden layer
    samples_per_iter = 1
    
    # early-stopping parameters
    stop_on_growing_error = True
    stop_valid_percentage = 1.  # percentage of validation used for early-stopping 
    min_loss_improvment = .01  # percentage of minimum loss' decrease (.01 is 1%)
    
    # reset computational graph
    tf.reset_default_graph()
    
    # create the computational graph
    with tf.device('/device:GPU:0'):
        
        # debug variable: encode-decode shapes
        enc_shapes = []
        dec_shapes = []
                
        # define input/output pairs
        input_ = tf.placeholder(tf.float32, [1, sequence_len, batch_size])  # (batch, input, time)        
       
        weights_vae_encoder = [tf.Variable(tf.truncated_normal(shape=[shape,
                                                                      num_conv_channels,
                                                                      num_conv_channels])) for shape in vae_encoder_shape_weights]
            
        weights_vae_decoder = [tf.Variable(tf.truncated_normal(shape=[batch_size,
                                                                      1,
                                                                      shape,
                                                                      num_conv_channels])) for shape in vae_decoder_shape_weights]
        
        # VAE graph's definition 
        
        # use multiple input, one for each input channel
        input_tiled = tf.tile(input_, [1, 1, num_conv_channels])
        
        # debug enc-dec shapes
        enc_shapes.append(input_tiled.get_shape())
        
        vae_encoder = tf.nn.conv1d(input_tiled, 
                                   weights_vae_encoder[0],
                                   vae_encoder_strides[0],
                                   'SAME')
        
        vae_encoder = vae_activation(vae_encoder)
        
        # debug enc-dec shapes
        enc_shapes.append(vae_encoder.get_shape())
        
        for (w_vae, s_vae) in zip(weights_vae_encoder[1:], vae_encoder_strides[1:]):
            
            vae_encoder = tf.nn.conv1d(vae_encoder, 
                                       w_vae,
                                       s_vae,
                                       'SAME')

            vae_encoder = vae_activation(vae_encoder)
            
            # debug enc-dec shapes
            enc_shapes.append(vae_encoder.get_shape())
            
        
        # fully connected hidden layer to shape the nn hidden state
        hidden_enc_weights = tf.Variable(tf.truncated_normal(shape=[vae_encoder.get_shape().as_list()[1],
                                                                    num_conv_channels,
                                                                    2*vae_hidden_size]))
    
        hidden_enc_bias = tf.Variable(tf.truncated_normal(shape=[2*vae_hidden_size,
                                                                 1]))
    
        vae_encoder = tf.tensordot(hidden_enc_weights, tf.transpose(vae_encoder), axes=([1,0],[0,1])) 
        vae_encoder += hidden_enc_bias
        
        # debug enc-dec shapes
        enc_shapes.append(vae_encoder.get_shape())
        
        # means and variances' vectors of the learnt hidden distribution
        #  we assume the hidden gaussian's variances matrix is diagonal
        loc = tf.slice(vae_encoder, [0, 0], [vae_hidden_size, -1])
        scale = tf.slice(tf.nn.softplus(vae_encoder), [vae_hidden_size, 0], [vae_hidden_size, -1])
        
        loc = tf.transpose(loc)
        scale = tf.transpose(scale)
        
        # sample from the hidden ditribution
        vae_hidden_distr = tfp.distributions.MultivariateNormalDiag(loc, scale)
        
        # re-parametrization trick: sample from standard multivariate gaussian,
        #  multiply by std and add mean (from the input sample)
        prior = tfp.distributions.MultivariateNormalDiag(tf.zeros(vae_hidden_size),
                                                         tf.ones(vae_hidden_size))
        
        hidden_sample = prior.sample()*scale + loc
        
        # get probability of the hidden state
        vae_hidden_prob = prior.prob(hidden_sample)
                
        # rebuild the input with 'de-convolution' operations  
        feed_decoder = tf.tile(tf.expand_dims(hidden_sample, -1), [1, 1, num_conv_channels])
        feed_decoder = tf.reshape(feed_decoder, shape=(1,
                                                       1,
                                                       vae_hidden_size, 
                                                       num_conv_channels))
        
        vae_decoder = tf.contrib.layers.conv2d_transpose(feed_decoder,
                                                         num_outputs=num_conv_channels,
                                                         kernel_size=(1, vae_decoder_shape_weights[0]),
                                                         stride=vae_decoder_strides[0],
                                                         padding='SAME')
        
        vae_decoder = vae_activation(vae_decoder)
        
        # debug enc-dec shapes
        dec_shapes.append(vae_decoder.get_shape())
        
        for (w_vae, s_vae) in zip(vae_decoder_shape_weights[1:], vae_decoder_strides[1:]):
            
            vae_decoder = tf.contrib.layers.conv2d_transpose(vae_decoder,
                                                             num_outputs=num_conv_channels,
                                                             kernel_size=(1, w_vae),
                                                             stride=s_vae,
                                                             padding='SAME')
            
            vae_decoder = vae_activation(vae_decoder)
            
            # debug enc-dec shapes
            dec_shapes.append(vae_decoder.get_shape())            
            
        # hidden fully-connected layer
        hidden_dec_weights = tf.Variable(tf.truncated_normal(shape=[vae_decoder.get_shape().as_list()[2],
                                                                    vae_decoder.get_shape().as_list()[1],
                                                                    input_.get_shape().as_list()[1]]))
    
        hidden_dec_bias = tf.Variable(tf.truncated_normal(shape=[1,
                                                                 ]))
           
        vae_decoder = tf.squeeze(vae_decoder, 0)
        vae_decoder = tf.tensordot(hidden_dec_weights, vae_decoder, axes=([1,0],[0,1])) 
        vae_decoder += hidden_dec_bias
        vae_decoder = tf.transpose(tf.expand_dims(vae_decoder, axis=0))
        
        # debug enc-dec shapes
        dec_shapes.append(vae_decoder.get_shape())
        
        # time-series reconstruction and ELBO loss
        vae_reconstruction = tfp.distributions.MultivariateNormalDiag(tf.constant(np.zeros(batch_size*sequence_len, dtype='float32')),
                                                                      tf.constant(np.ones(batch_size*sequence_len, dtype='float32')))
            
        likelihood = elbo_importance[0]*tf.reduce_mean(vae_reconstruction.log_prob(vae_decoder))        
        divergence = elbo_importance[1]*tfp.distributions.kl_divergence(prior, vae_hidden_distr)        
        elbo = tf.reduce_mean(likelihood - divergence)
        
        # apply elastic net regularization (lambda_reg is the couple parameter that controls L1-L2 combination)
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=lambda_reg[0], scope=None)
        nn_params = tf.trainable_variables() # all vars of your graph
        l1_regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, nn_params)
        
        l2_regularizer = tf.contrib.layers.l2_regularizer(scale=lambda_reg[1], scope=None)
        l2_regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, nn_params)
        
        regularized_elbo = -elbo + l1_regularization_penalty + l2_regularization_penalty
            
        optimizer_elbo = tf.train.AdamOptimizer(learning_rate_elbo).minimize(regularized_elbo)
        
    if random_stride == False:
               
        # extract train and test
        x_train, y_train, x_valid, y_valid, x_test, y_test = utils.generate_batches(
                                                                 filename=data_path, 
                                                                 window=sequence_len,
                                                                 stride=stride,
                                                                 mode='strided-validation', 
                                                                 non_train_percentage=.5,
                                                                 val_rel_percentage=.6,
                                                                 normalize=normalization,
                                                                 time_difference=False,
                                                                 td_method=None,
                                                                 subsampling=subsampling,
                                                                 rounding=rounding)
               
        # suppress second axis on Y values (the algorithms expects shapes like (n,) for the prediction)
        y_train = y_train[:,0]; y_valid = y_valid[:,0]; y_test = y_test[:,0]
               
        if len(x_train) > len(y_train):
            
            x_train = x_train[:len(y_train)]
        
        if len(x_valid) > len(y_valid):
            
            x_valid = x_valid[:len(y_valid)]
        
        if len(x_test) > len(y_test):
            
            x_test = x_test[:len(y_test)]
            
        # adjust train test size (for testing purpose)
        y_train = y_train[300:3000]
        x_train = x_train[300:3000,:]
        y_valid = y_valid[300:1600]
        x_valid = x_valid[300:1600,:]
        y_test = y_test[4500:]
        x_test = x_test[4500:,:]
            
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=False,
                                          log_device_placement=False)) as sess:
        
        # print out series of transformations' shapes
        print("\nEncoder shapes: \n", enc_shapes)
        print("\nDecoder shapes:", dec_shapes)
        print()
        
        # create dataset with random stride
        # extract train and test
        if random_stride == True: 
            
            x_train, y_train, x_valid, y_valid, x_test, y_test = utils.generate_batches(
                                                                     filename=data_path, 
                                                                     window=sequence_len,
                                                                     stride=np.random.randint(1, stride),
                                                                     mode='strided-validation', 
                                                                     non_train_percentage=.5,
                                                                     val_rel_percentage=.5,
                                                                     normalize=normalization,
                                                                     time_difference=False,
                                                                     td_method=None,
                                                                     subsampling=subsampling,
                                                                     rounding=rounding)
           
            # suppress second axis on Y values (the algorithms expects shapes like (n,) for the prediction)
            y_train = y_train[:,0]; y_valid = y_valid[:,0]; y_test = y_test[:,0]
            
            if len(x_train) > len(y_train):
                
                x_train = x_train[:len(y_train)]
            
            if len(x_valid) > len(y_valid):
                
                x_valid = x_valid[:len(y_valid)]
            
            if len(x_test) > len(y_test):
                
                x_test = x_test[:len(y_test)]
        
        # train + early-stopping
        init = tf.global_variables_initializer()
        
        sess.run(init)
        tf.random.set_random_seed(42)
        
        # train
        last_error_on_valid = np.inf
        current_error_on_valid = .0
        e = 0
        
        while e < epochs:
            
            # inject some random noise in the training data
            if make_some_noise[0] == True:
                
                t_min, t_max = (np.min(y_train), np.max(y_train))
                
                random_noise = make_some_noise[1]*(t_max-t_min)
                random_noise = np.random.rand(*x_train.shape)*random_noise
                x_train += random_noise
                
            print("Train epoch ", e)
            iter_ = 0
            
            train_num_samples = int(np.floor(x_train.shape[0] / batch_size))            
            while iter_ < train_num_samples:
                   
                #print("Sample ", iter_,"out of ", train_num_samples)
                
                batch_x = x_train[iter_*batch_size: (iter_+1)*batch_size, :].T.reshape(1, sequence_len, batch_size)
                
                # run VAE encoding-decoding
                sess.run(optimizer_elbo, feed_dict={input_: batch_x})
                
                iter_ +=  1

            if stop_on_growing_error:
                
                print("Validation")
                                               
                current_error_on_valid = .0
                                
                # verificate stop condition
                iter_val_ = 0
                validation_num_samples = int(stop_valid_percentage * np.floor(x_valid.shape[0] / batch_size))
                while iter_val_ < validation_num_samples:
                    
                    #print("Sample ", iter_,"out of ", validation_num_samples)

                    batch_x_val = x_valid[iter_val_*batch_size: (iter_val_+1)*batch_size, :].T.reshape(1, sequence_len, batch_size)
                    
                    # accumulate error
                    current_error_on_valid +=  np.abs(np.sum(sess.run(regularized_elbo, feed_dict={input_: batch_x_val})))

                    iter_val_ += 1
                    
                print("Previous error on valid ", last_error_on_valid)
                print("Current error on valid ", current_error_on_valid)
                                 
                # stop learning if the loss reduction is below the threshold (current_loss/past_loss)
                if current_error_on_valid > last_error_on_valid or (np.abs(current_error_on_valid/last_error_on_valid) > 1-min_loss_improvment and e!=0):
                    
                    print("Early stopping: validation error has increased since last epoch.")
                    e = epochs
                        
                last_error_on_valid = current_error_on_valid
               
            # eliminate the noise from training (if injected)
            if make_some_noise[0] == True:
                
                x_train -= random_noise
                
            e += 1
            
        # anomaly detection on test set
        y_test = y_test[:x_test.shape[0]]
        
        # find the thershold that maximizes the F1-score
        best_precision = best_recall = best_threshold = .0
        best_predicted_positive = np.array([])
        condition_positive = np.array([])
        
        for t in sigma_threshold_elbo:
            
            print("Optimizing with threshold's value: ", t)
            
            vae_anomalies = []
            p_anom = np.zeros(shape=(int(np.floor(x_test.shape[0] / batch_size)),))
            threshold_elbo = (t, 1.-t)            
            iter_ = 0
            
            while iter_ < int(np.floor(x_test.shape[0] / batch_size)):
                        
                batch_x = x_test[iter_*batch_size: (iter_+1)*batch_size, :].T.reshape(1, sequence_len, batch_size)
                            
                # get probability of the encoding and a boolean (anomaly or not)        
                p_anom[iter_] = sess.run(vae_hidden_prob, feed_dict={input_: batch_x})  
                
                """
                # highlight anomalies   (the whole window is considered)                 
                if (p_anom[iter_] <= threshold_elbo[0] and iter_<int(np.floor(x_test.shape[0] / batch_size))-sequence_len):
                    
                    for i in range(iter_, iter_+sequence_len):
                        
                        vae_anomalies.append(i)
                """
                                           
                iter_ +=  1
                
            # predictions
            predicted_positive = np.array([vae_anomalies]).T
            
            if len(vae_anomalies) == 0:
                
                continue
                
            # caveat: define the anomalies based on absolute position in test set (i.e. size matters!)
            # train 50%, validation_relative 50%
            # performances
            target_anomalies = np.zeros(shape=int(np.floor(y_test.shape[0] / batch_size))*batch_size)
            target_anomalies[160:200] = 1
        
            # real values
            condition_positive = np.argwhere(target_anomalies == 1)
        
            # precision and recall
            try:
                
                precision = len(np.intersect1d(condition_positive, predicted_positive))/len(predicted_positive)
                recall = len(np.intersect1d(condition_positive, predicted_positive))/len(condition_positive)
            
            except ZeroDivisionError:
                
                precision = recall = .0
                
            print("Precision and recall for threshold: ", t, " is ", (precision, recall))
            
            if precision >= best_precision:
                
                best_threshold = t
                best_precision = precision
                best_recall = recall
                best_predicted_positive = cp.copy(predicted_positive)
                
        # plot data series    
        fig, ax1 = plt.subplots()
        
        print("\nTime series:")
        ax1.plot(y_test, 'b', label='index')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Space Shuttle')
        
        # plot predictions
        for i in vae_anomalies:
    
            plt.axvspan(i, i+1, color='yellow', alpha=0.5, lw=0)
    
        fig.tight_layout()
        plt.show()
        
        fig, ax1 = plt.subplots()

        ax1.scatter([i for i in range(len(p_anom))],p_anom)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Likelihood')
                
        fig.tight_layout()
        plt.show()
                
        print("Anomalies in the series:", condition_positive.T)
        print("Anomalies detected with threshold: ", best_threshold)
        print(best_predicted_positive.T)
        print("Precision and recall ", best_precision, best_recall)