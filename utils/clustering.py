# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 13:15:11 2019

@author: Emanuele
"""

from sklearn.cluster import KMeans


"""
 Simple wrapper for k-means: each sample is represented as a single number (its cluster's number).
 Takes as input:
     data:numpy.array, dataset that comes in the shape (n, m), where n is the 
      number of datapoints and m the dimensions of each sample;
     n_clusters:int, number of clusters ('k' of k-means).
  Returns:
      the object that represent the clusters and other info:
          e.g. obj.cluster_centers_.mean(axis=1), obj.labels_ return respectively,
          the cluster's means and a vector where for each point, the belonging cluster
          is indicated.
"""
def k_means(data, n_clusters):

    kmeans  = KMeans(n_clusters=n_clusters,
                     init='k-means++',
                     n_init=10,
                     max_iter=100).fit(data)
    
    return kmeans
    