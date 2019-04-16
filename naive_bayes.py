#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    @author: Cheng Wang
    @contact: dr.rer.nat.chengwang@gmail.com
    @16.04.2019
'''

import matplotlib.pyplot as plt
import numpy as np
import random
import operator
import pdb

'''
Naive_Bayes
'''
class Naive_Bayes():

    def __init__(self):
        self.prior = {}
        self.likelihood={}

    # compute prior probabilities
    def compute_prior(self, Y):        
        for y in Y:
            key = 'Y='+str(y)
            if key not in self.prior:
                self.prior[key]=1
            else: 
                self.prior[key]+=1
        for k in self.prior.keys():
            self.prior[k] = float(self.prior[k])/len(Y)
        print("prior: ", self.prior)
       
    def compute_likelihood(self, X, Y):
        self.classes = np.unique(Y)
        self.features = np.unique(X)
        
        for c in self.classes:
            # the sample with a given label c
            samples_with_label = np.where(Y == c)[0] 
            X_sub = X[samples_with_label,:]
            for feat_idx in range(len(self.features)):
                # the samples with label c, and a given feature
                samples_with_label_and_feature = np.where(X_sub[:,feat_idx] == self.features[feat_idx])[0]
                
                key='X='+str(feat_idx)+'|'+'Y='+str(c)
                self.likelihood[key]=float(len(samples_with_label_and_feature))/len(samples_with_label)
        print("likelihood: ", self.likelihood)       
           
    def fit(self, X, Y):
        self.compute_prior(Y)
        self.compute_likelihood(X, Y)

    def predict(self, x):
        predictions={}
        for c in self.classes:
            p_key='Y='+str(c)
            p = 1
            for feature in x:
                l_key='X='+str(feature)+'|'+'Y='+str(c)
                p *= self.likelihood[l_key]*self.prior[p_key]
            predictions[c]=p
        # get the key of maximum value    
        pred = max(predictions.iteritems(), key=operator.itemgetter(1))[0]
        return pred

if __name__ == '__main__':

    train_X= np.random.randint(4, size=(500, 4))
    train_Y = np.random.binomial(1, 0.5, 100)
    
    test_X= np.random.randint(4, size=(20, 4))
    test_Y = np.random.binomial(1, 0.5, 20)
    
    #pdb.set_trace()
    NB = Naive_Bayes()
    
    NB.fit(train_X, train_Y)
    
   
    cnt=0
    cnt=0
    for X, y in zip(test_X, test_Y):
        pred = NB.predict(X)
        #print(pred)
        #pdb.set_trace()
        print(pred,y)
        if(pred==y):
            cnt +=1
    print("accuracy: %f"%(float(cnt)/len(test_Y)))
    
            
        
        
        
        
        
        
        
        
        
        
        
        
        
