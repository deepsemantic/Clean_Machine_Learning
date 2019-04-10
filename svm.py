#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    @author: Cheng Wang
    @contact: dr.rer.nat.chengwang@gmail.com
    @22.03.2019
'''

import numpy as np
import random
import pdb

'''
Support Vector Machine
'''
class SVM():

    def __init__(self, lr=0.01, num_epoch=100, C=1, eps=0.001):
        self.lr = lr 
        self.num_epoch=num_epoch
        self.C=C
        self.eps=eps
    
    def init_params(self, W_shape, b_shape):
        self.W = np.ones(W_shape)
        self.b = np.ones(b_shape)
    
    def loss(self, h, y):
        pass
        
    def get_index(self, a,b,z):
        i = z
        cnt=0
        while i == z and cnt<1000:
            i = random.randint(a,b)
            cnt=cnt+1
        return i
        
    def linear(self, x, W, b):
        #pdb.set_trace()
        p = np.sign(np.dot(x, W)+b).astype(int)
        return p
    
    def fit(self, X, Y):
        X=np.asarray(X)
        Y=np.asarray(Y)
        N = len(X) # number of samples
        #pdb.set_trace()
        D = X[0].shape[0] # dim
        self.init_params((D, 1),(1,)) 
        alpha = np.zeros((N)) # the alpha
        epoch=0
        while(epoch<self.num_epoch):
            for j in range(0, N):
                i = self.get_index(0, N-1, j) # Get random int i~=j
               
                xi, xj, yi, yj = X[i], X[j], Y[i], Y[j]
                k_ij = self.kernel(xi,xj)
                
                if k_ij==0:
                    continue
                ai, aj = alpha[i], alpha[j]
                ''' SMO  '''
                # compute the lower and upper bounds for alpha
                L, H  = self.bound(self.C, ai,aj,yi,yj)
                
                #pdb.set_trace()
                # compute weights
                self.W  = np.dot(X.T, alpha * Y) #p106 (2)
                self.b  =sum(Y - np.dot(X, self.W))/len(Y) #p106
                
                #pdb.set_trace()
                Ei = self.linear(xi, self.W, self.b)-yi # p127, 7.105
                Ej = self.linear(xj, self.W, self.b)-yj
                
                alpha[j]=aj + float(yj*(Ei-Ej))/k_ij  #p127 7.108
                alpha[j]=max(alpha[j], L)
                alpha[j]=min(alpha[j], H)
                
                alpha[i]=ai + yi*yj*(aj-alpha[j])
            print("epoch: %d "%(epoch))
            epoch +=1
            
        ''' compute final model parameters'''
        self.W  = np.dot(X.T, alpha * Y)
        self.b  = sum(Y - np.dot(X, self.W))/len(Y)
        
        ''' get support vectors'''
        alpha_idx = np.where(alpha > 0)[0]
        support_vectors = X[alpha_idx, :]
        
        
        return support_vectors
        
    def kernel(self, xi, xj):
        return np.dot(xi.T, xj)
    
    ''' the boundary of L<= a <=H '''    
    def bound(self, C, ai, aj, yi, yj):
        L = max(0, aj-ai) if yi!=yj else max(0, aj+ai-C)
        H = min(C, C-aj+ai) if yi!=yj else min(C, ai+aj)
        return L, H
           
    def predict(self, x):
        #pdb.set_trace()
        return np.sign(np.dot(x, self.W) + self.b).astype(int)         
            
        
def load_mnist():
    # Load the dataset
    train = np.load("data/minst2_train.npy")
    test = np.load("data/minst2_test.npy")
    return train, test   


if __name__ == '__main__':
    trainSet, testSet = load_mnist()
    train_X= [X[0] for X in trainSet]
    train_Y= [Y[1] for Y in trainSet]
    
    test_X= [X[0] for X in testSet]
    test_Y= [Y[1] for Y in testSet]
    SVM_classifier = SVM(lr=0.01, num_epoch=10)
    
    SVM_classifier.fit(train_X, train_Y)
    cnt=0
    for X, y in zip(test_X, test_Y):
        pred = SVM_classifier.predict(X)
        #pdb.set_trace()
        if pred==-1:
            pred=0
        print(pred,y)
        if(pred==y):
            cnt +=1
    print("accuracy: %f"%(float(cnt)/len(test_Y)))
            
        
        
        
        
        
        
        
        
        
        
        
        
        
