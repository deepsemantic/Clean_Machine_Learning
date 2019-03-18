#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    @author: Cheng Wang
    @contact: dr.rer.nat.chengwang@gmail.com
'''

import numpy as np
import pdb

'''
Logistic Regression
'''
class Logistic_Regression():

    def __init__(self, lr=0.01, num_epoch=10):
        self.lr = lr 
        self.num_epoch=num_epoch
    
    def init_params(self, shape):
        self.W = np.zeros(shape)
    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x));
    
    def loss(self, h, y):
        L= -y * np.log(h)-(1-y)*np.log(1-h)
        return L.mean()
        
    def update_params(self, grad):
        self.W -=self.lr * grad 
       
    def fit(self, train_X, train_Y):
        self.init_params(train_X[0].shape)
        for epoch in range(self.num_epoch):
            for X, y in zip(train_X, train_Y):
                h = self.sigmoid(np.dot(X, self.W))
                gradient = np.dot(X.T, (h-y))
                self.update_params(gradient)
                loss = self.loss(h, y)
            print("epoch: %d   loss:%f"%(epoch, loss))
    
    def predict(self, X):
        return 1 if self.sigmoid(np.dot(X, self.W))>=0.5 else 0          
            
        
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
    LR_classifier = Logistic_Regression(lr=0.01, num_epoch=10)
    
    LR_classifier.fit(train_X, train_Y)
    cnt=0
    for X, y in zip(test_X, test_Y):
        pred = LR_classifier.predict(X)
        #pdb.set_trace()
        print(pred,y)
        if(pred==y):
            cnt +=1
    print("accuracy: %f"%(float(cnt)/len(test_Y)))
            
        
        
        
        
        
        
        
        
        
        
        
        
        
