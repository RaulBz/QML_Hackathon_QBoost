import os

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split


X_tr = np.load('X_train.npy')
X_te = np.load('X_test.npy')
y1_tr = np.load('y1_train.npy')
y2_tr = np.load('y2_train.npy')
#%%

mean = np.mean(X_tr, axis = 0)
std = np.std(X_tr, axis = 0)

sample_vector = np.random.normal(loc = mean, scale = std)

#%%
import csv
def vec_selec(X, i):
    
    mean = np.mean(X, axis = 0)
    std = np.std(X, axis = 0)
    
    V = X[i]
    
    return V, mean, std

def add_noise(X, N_entries):
    X_new = []
    for i in range(len(X)):
        V, mean, std = vec_selec(X, i)
        
        sample_vector = np.random.normal(loc = mean, scale = std)
        
        indexes = np.arange(V.shape[0])
        np.random.shuffle(indexes)
        
        R = indexes[:N_entries]
        
        
        for i in R:
            
            V[i] = sample_vector[i]
        X_new.append(V)
    X_new = np.array(X_new)
    
    return X_new

def grid_search(X):
    
    X_train_new = np.array
    for N_entries in range(0, 21):
    
        X_tarin_new = add_noise(X, N_entries)
        name = 'X_train_drop_'+ str(N_entries)
        P = np.save(X_train_new, (name, N_entries))
    return P
        
        