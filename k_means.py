#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    @author: Cheng Wang
    @contact: dr.rer.nat.chengwang@gmail.com
    @12.04.2019
'''

import matplotlib.pyplot as plt
import numpy as np
import random
import pdb

'''
K means
'''
class K_means():

    def __init__(self, K=3, num_epoch=5):
        self.K=K
        self.num_epoch=num_epoch
    # compute L2 distance
    def distance(self, x):        
        d = [np.linalg.norm(x-c) for c in self.centroids]
        return d
        
    def fit(self, X):
        X=np.asarray(X)
        N = len(X) # number of samples
        random_idx = random.sample(range(N), self.K)
        
        self.centroids={}
        self.clusters={}
        
        #random select K sample as init centroids
        for k, i in zip(range(self.K), random_idx):
            self.centroids[k] = X[i]
            
        epoch=0
        optimized=False
        while(epoch<self.num_epoch or optimized==false):
         
            for x in X:
                distances = self.distance(x) # distance between sample X[i] to all centroids
                cluster_idx = np.argmin(distances)
                if self.clusters.get(cluster_idx)==None:
                    self.clusters[cluster_idx]=[]
                self.clusters[cluster_idx].append(x)
               
            pre_centroids=dict(self.centroids)
            
            #compute new centroids
            for idx in self.clusters:
                self.centroids[idx] = np.mean(self.clusters[idx], axis=0) 
            

            #update centroids
            for c in self.centroids:
                d = abs(np.sum(pre_centroids[c]-self.centroids[c]))
                print("centroid distance: "+str(d))
                if d<0.001: # stop criterion, when the changes of centroids is small enough
                    optimized =True 
            if(optimized):
                break
        return self.centroids, self.clusters  
        
    def predict(self, x):
        distance = self.distance(x)
        cluster = np.argmin(distance)
        return cluster        
            

if __name__ == '__main__':

    train_X= np.random.randint(50, size=(100, 2))
    test_X= np.random.randint(5, size=(20, 2))
    plt.scatter(train_X[:,0], train_X[:,1], s=10)
    #plt.show()
    K_clsuter = K_means(K=5, num_epoch=5)
    
    centroids, clusters = K_clsuter.fit(train_X)
    
    for k in centroids:
        plt.scatter(centroids[k][0], centroids[k][1], color='r', linewidths=5)
  
    for k in clusters:
        for x in clusters[k]:
            plt.scatter(x[0],x[1], color='blue', marker="x", s=5)
    plt.show()
 
    cnt=0
    for x in test_X:
        pred = K_clsuter.predict(x)
    print("clsuter: %d"%(pred))
            
        
        
        
        
        
        
        
        
        
        
        
        
        
