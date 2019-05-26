# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 13:05:10 2018

@author: Emanuele

Test if a dataset follows a gaussian distribution.
"""

import matplotlib.pyplot as plt
import scipy.stats as stats


"""
 Test 4 different gaussian test and report the results.
 The test are in order: the Shapiro-Wilk's test, a normal test included in scipy
  package bassed on D’Agostino and Pearson’s test, the Kolmogorov-Smirnov's test
  and the Andreson's test.  
 For the first 3 tests, a pvalue that is too low (let's say lower than 1e-2) 
  indicates that we cannot reject the null hypotesis, i.e. the evidence is 
  against the assumption data comes from a gaussian distribution.
 For the last test (Andreson), we have to compare the 'critical_values' against 
  a table of critical values that depends also on number of samples.
 The function takes as input:
     data:np.array, a 1d dataset whose shape is (n,), where n is the number of samples
     in the dataset.
""" 
def gaussian_test(data):
    
    # plot histograms
    plt.hist(data, bins=30)
    
    # gaussian's test
    print("\n\nShapiro:", stats.shapiro(data), 
          "\n\nNormal test:", stats.normaltest(data), 
          "\n\nKolmogorov-Smirnov:", stats.kstest(data, 'norm'), 
          "\n\nAnderson test:", stats.anderson(data, 'norm'))
