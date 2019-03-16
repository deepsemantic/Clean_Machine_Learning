#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    @author: Cheng Wang
    @contact: dr.rer.nat.chengwang@gmail.com
'''

import numpy as np
import pdb

'''
KNN
'''
class KNN():

    def __init__(self):
        pass
    #euclinean distance
    def compute_distance(self, X1, X2):
        dist = 0
        for (x1, x2) in zip(X1, X2):
            dist += (x1 - x2)**2
        return dist
    
    def kNeighbors(self, train_X, train_Y, test_sample, k):
        distances = [] # distance of testsample to all trainsamples
        for train_sample, train_label in zip(train_X, train_Y):
            d = self.compute_distance(train_sample, test_sample)
            distances.append((train_sample, train_label, d))
        distances.sort(key=lambda x: x[-1]) ## sort by distance
        k_neighbors=[sample[0:-1] for sample in distances[0:k]]
        return k_neighbors

    def compute_votes(self, neighbors):
        label_votes={}
        for neighbor in neighbors:
            label = neighbor[-1]
            if label in label_votes.keys():
                label_votes[label] +=1
            else:
                label_votes[label]  =1
        sorted_votes=sorted(label_votes.items(), key=lambda kv: kv[1]) ## sorted by vote numbers
        return sorted_votes[0][0]

        
def load_mnist():
    # Load the dataset
    train = np.load("data/minst5_train.npy")
    test = np.load("data/minst5_test.npy")
    return train, test   


if __name__ == '__main__':
    trainSet, testSet = load_mnist()
    train_X= [X[0] for X in trainSet]
    train_Y= [Y[1] for Y in trainSet]
    
    test_X= [X[0] for X in testSet]
    test_Y= [Y[1] for Y in testSet]
    KNN_classifier = KNN()
    k=1
    cnt=0
    for test_x, test_y in zip(test_X, test_Y):
        neighbors = KNN_classifier.kNeighbors(train_X, train_Y, test_x, k)
        pred = KNN_classifier.compute_votes(neighbors)
        if(test_y==pred): 
            cnt +=1
    print("accuracy: %d%",(float(cnt)/len(test_Y)))
        
        
        
        
        
        
        
        
        
        
        
        
        
