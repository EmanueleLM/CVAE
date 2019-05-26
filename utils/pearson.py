# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 13:05:10 2018

@author: Emanuele

Test perason coefficient between two datasets.
"""

from scipy.stats import pearsonr


"""
 Test perason coefficient between two datasets.
""" 
def gaussian_test(data1, data2):
    
    rho, p = pearsonr(data1, data2)
    
    print("Rho coefficient: ", rho)
    print("P-value (low is how confident we are on rho): ", p)
