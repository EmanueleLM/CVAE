# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 15:18:22 2019

@author: Emanuele
"""

import numpy as np
import matplotlib.pyplot as plt


"""
 Calculate the autocovariance function between a signal y(t) and istelf shifted
  towards the future, i.e. y(t+\delta), where delta is a variable you can set.
  autocorr(y(t), y(t-k)) = sum{i \in [0, n-k]}((y(t)-mu)*(y(t-i)))/(n-1)
  Takes as input:
      data:numpy.array, array of size (n,);
      max_lag:integer, the max timestep you want to calculate covariance between data and its shifted version
       e.g. max_lag=0 means correlating the input with itself, with no shift, max_lag=1 means correlating
        the input with itself shifted in the future of 1 timestep, i.e. max_lag=k means correlating
        y(t) with y(t+k);
      plot:boolean, True if you want to plot the autocovariance function. 
  Returns:
      res:numpy.array, the autocovariance vector whose size is (max_lag,)
"""
def autocovariance_function(data, max_lag, plot=False):
    
    ts_length = len(data)    
    autocovariance = np.zeros(shape=(max_lag,))
    mean_data = data.mean()
    
    for i in range(max_lag):
        
        autocovariance[i] = np.sum((data[:ts_length-i]-mean_data)*(data[i:]-mean_data))
        autocovariance[i] /= (ts_length-1)
        
    autocovariance = autocovariance.flatten()
        
    if plot is True:
        
        plt.plot(autocovariance)
        
    return autocovariance
