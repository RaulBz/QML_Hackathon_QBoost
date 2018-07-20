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

def ve_selec(X):
    
    mean = np.mean(X, axis = 0)
    std = np.std(X, axis = 0)
    
    V = random.choice(X)
    
    return V, mean, std

def add_noise(X, N_entries):
    
    V, mean, std = vec_selec(X)
    
    sample_vector = np.random.normal(loc = mean, scale = std)
    
    indexes = np.array(len(V))
    
    R_indexes = np.shuffle(indexes)
    
    R = R_indexes[:N_entries]
    
    
    for i in R:
        
        V[i] = sample_vector[i]
    
    return V

    