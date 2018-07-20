import os

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


X_tr = np.load('X_train.npy')
X_te = np.load('X_test.npy')
y1_tr = np.load('y1_train.npy')
y2_tr = np.load('y2_train.npy')

# performance metric
def rmsle(h, y): 
    """
    Compute the Root Mean Squared Log Error for hypthesis h and targets y

    Args:
        h - numpy array containing predictions with shape (n_samples, n_targets)
        y - numpy array containing targets with shape (n_samples, n_targets)
    """
    
#     h, y = np.expm1(h), np.expm1(y)
    
    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())

from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor

# run different model for different Target Variables

grad_1 = GradientBoostingRegressor(
                loss='ls',
                learning_rate = 0.0035,
                max_depth=7,
                n_estimators=1120,
                max_features=7,
                min_samples_leaf=43,
                min_samples_split=14,
                min_weight_fraction_leaf=0.01556)

grad_2 = GradientBoostingRegressor(
                loss='ls',
                learning_rate = 0.0035,
                max_depth=6,
                n_estimators=3275,
                max_features=2,
                min_samples_leaf=2,
                min_samples_split=2,
                min_weight_fraction_leaf=0.08012)

def assess_grad(X, y_list, model_list):
    """ Used to access model performance. Returns the mean rmsle score of cross validated data
    """
    final = []
    best_iter = [[], []]
    for idx, y in enumerate(y_list):
        kfold = KFold(n_splits=10, shuffle=True)
        out = []
        for train_index, test_index in kfold.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = model_list[idx]
            model.fit(X_train, y_train)
            h =  model.predict(X_test)
            e = rmsle(np.expm1(h), np.expm1(y_test))
            print(e)
            out.append(e)
        final.append(np.array(out).mean())
                      
    return(np.array(final).mean(), np.array(final).std())

#model = assess_grad(X_tr, [y1_tr, y2_tr], [grad_1, grad_2])
#print("Model RMSLE: {}, std: {}".format(model[0], model[1]))


# Fill in your API token:


sapi_token = 'CDL8-df1de1d5d76560ee73a82ffca3833a1a444536d3'
url = 'https://cloud.dwavesys.com/sapi'
token = sapi_token
solver_name = 'c4-sw_sample'

# import necessary packages
from sklearn import preprocessing, metrics
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, AdaBoostRegressor, RandomForestRegressor
from sklearn.datasets.mldata import fetch_mldata
from sklearn.datasets import load_breast_cancer
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from qboost import WeakRegressor, QBoostRegressor, QboostPlus, QboostPlusRegression


# Define the functions required in this example

#def metric(y, y_pred):
#    """
#    :param y: true label
#    :param y_pred: predicted label
#    :return: metric score
#    """
#
#    return metrics.accuracy_score(y, y_pred)

# performance metric
#def metric(h, y): 
#    """
#    Compute the Root Mean Squared Log Error for hypthesis h and targets y

#    Args:
#        h - numpy array containing predictions with shape (n_samples, n_targets)
#        y - numpy array containing targets with shape (n_samples, n_targets)
#    """
#    
##     h, y = np.expm1(h), np.expm1(y)
#    
#    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())


def metric(y, y_pred):
    """
    :param y: true label
    :param y_pred: predicted label
    :return: metric score
    """

    return metrics.accuracy_score(y, y_pred)

def train_model(X_train, y_train, X_test, y_test, lmd):
    """
    :param X_train: training data
    :param y_train: training label
    :param X_test: testing data
    :param y_test: testing label
    :param lmd: lambda used in regularization
    :return:
    """

    # define parameters used in this function
    NUM_READS = 1000
    NUM_WEAK_CLASSIFIERS = 30
    TREE_DEPTH = 4
    DW_PARAMS = {'num_reads': NUM_READS,
                 'auto_scale': True,
                 'num_spin_reversal_transforms': 10,
                 'postprocess': 'optimization',
                 }

    # define sampler
    dwave_sampler = DWaveSampler(token=sapi_token, endpoint = url)
    emb_sampler = EmbeddingComposite(dwave_sampler)

    N_train = len(X_train)
    N_test = len(X_test)
    print("\n======================================")
    print("Train size: %d, Test size: %d" %(N_train, N_test))
    print('Num weak classifiers:', NUM_WEAK_CLASSIFIERS)

    # Preprocessing data
    imputer = preprocessing.Imputer()
    scaler = preprocessing.StandardScaler()
    normalizer = preprocessing.Normalizer()

    X_train = scaler.fit_transform(X_train)
    X_train = normalizer.fit_transform(X_train)

    X_test = scaler.fit_transform(X_test)
    X_test = normalizer.fit_transform(X_test)

#    GradientBoost
    clf0 = GradientBoostingRegressor(n_estimators=NUM_WEAK_CLASSIFIERS)
    clf0.fit(X_train, y_train)
    y_train1 = clf0.predict(X_train)
    y_test1 = clf0.predict(X_test)
#     print(clf1.estimator_weights_)
    print('accu (train): %5.2f'%(rmsle(y_train, y_train1)))
    print('accu (test): %5.2f'%(rmsle(y_test, y_test1)))


    ## Adaboost
    print('\nAdaboost')
    clf1 = AdaBoostRegressor(n_estimators=NUM_WEAK_CLASSIFIERS)
    clf1.fit(X_train, y_train)
    y_train1 = clf1.predict(X_train)
    y_test1 = clf1.predict(X_test)
#     print(clf1.estimator_weights_)
    print('accu (train): %5.2f'%(rmsle(y_train, y_train1)))
    print('accu (test): %5.2f'%(rmsle(y_test, y_test1)))

    # Ensembles of Decision Tree
    print('\nDecision tree')
    clf2 = WeakRegressor(n_estimators=NUM_WEAK_CLASSIFIERS, max_depth=TREE_DEPTH)
    clf2.fit(X_train, y_train)
    y_train2 = clf2.predict(X_train)
    y_test2 = clf2.predict(X_test)
#     print(clf2.estimator_weights)
    print('accu (train): %5.2f' % (rmsle(y_train, y_train2)))
    print('accu (test): %5.2f' % (rmsle(y_test, y_test2)))
    
    # Random forest
    print('\nRandom Forest')
    clf3 = RandomForestRegressor(max_depth=TREE_DEPTH, n_estimators=NUM_WEAK_CLASSIFIERS)
    clf3.fit(X_train, y_train)
    y_train3 = clf3.predict(X_train)
    y_test3 = clf3.predict(X_test)
    print('accu (train): %5.2f' % (rmsle(y_train, y_train3)))
    print('accu (test): %5.2f' % (rmsle(y_test, y_test3)))

    # Qboost
    print('\nQBoost')
    clf4 = QBoostRegressor(n_estimators=NUM_WEAK_CLASSIFIERS, max_depth=TREE_DEPTH)
    clf4.fit(X_train, y_train, emb_sampler, lmd=lmd, **DW_PARAMS)
    y_train4 = clf4.predict(X_train)
    y_test4 = clf4.predict(X_test)
    print(clf4.estimator_weights)
    print('accu (train): %5.2f' % (rmsle(y_train, y_train4)))
    print('accu (test): %5.2f' % (rmsle(y_test, y_test4)))

    # QboostPlus
    print('\nQBoostPlus')
    clf5 = QboostPlusRegression([clf0, clf1, clf2, clf3, clf4])
    clf5.fit(X_train, y_train, emb_sampler, lmd=lmd, **DW_PARAMS)
    y_train5 = clf5.predict(X_train)
    y_test5 = clf5.predict(X_test)
    print(clf5.estimator_weights)
    print('accu (train): %5.2f' % (rmsle(y_train, y_train5)))
    print('accu (test): %5.2f' % (rmsle(y_test, y_test5)))

    
    return [clf4, y_train4, y_train5]

# start training the model
#X_train = X_tr
#y_train = y1_tr
#y_train = 2*(y_train >0.25) - 1
#X_test = X_train
#y_test = y_train

random_state = np.random.RandomState(0)
x = X_tr
y = y1_tr
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=0)

lmd = 0.2
#clfs = train_model(X_train, y_train, X_test, y_test, 1.0)

# define parameters used in this function
NUM_READS = 1000
NUM_WEAK_CLASSIFIERS = 30
TREE_DEPTH = 4
DW_PARAMS = {'num_reads': NUM_READS,
             'auto_scale': True,
             'num_spin_reversal_transforms': 10,
             'postprocess': 'optimization',
             }

# define sampler
dwave_sampler = DWaveSampler(token=sapi_token, endpoint = url)
emb_sampler = EmbeddingComposite(dwave_sampler)

N_train = len(X_train)
N_test = len(X_test)
print("\n======================================")
print("Train size: %d, Test size: %d" %(N_train, N_test))
print('Num weak classifiers:', NUM_WEAK_CLASSIFIERS)

# Preprocessing data
imputer = preprocessing.Imputer()
scaler = preprocessing.StandardScaler()
normalizer = preprocessing.Normalizer()

X_train = scaler.fit_transform(X_train)
X_train = normalizer.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)
X_test = normalizer.fit_transform(X_test)

#    GradientBoost
clf0 = GradientBoostingRegressor(n_estimators=NUM_WEAK_CLASSIFIERS)
clf0.fit(X_train, y_train)
y_train1 = clf0.predict(X_train)
y_test1 = clf0.predict(X_test)
#     print(clf1.estimator_weights_)
print('accu (train): %5.2f'%(rmsle(y_train, y_train1)))
print('accu (test): %5.2f'%(rmsle(y_test, y_test1)))


## Adaboost
print('\nAdaboost')
clf1 = AdaBoostRegressor(n_estimators=NUM_WEAK_CLASSIFIERS)
clf1.fit(X_train, y_train)
y_train1 = clf1.predict(X_train)
y_test1 = clf1.predict(X_test)
#     print(clf1.estimator_weights_)
print('accu (train): %5.2f'%(rmsle(y_train, y_train1)))
print('accu (test): %5.2f'%(rmsle(y_test, y_test1)))

## Ensembles of Decision Tree
#print('\nDecision tree')
#clf2 = WeakRegressor(n_estimators=NUM_WEAK_CLASSIFIERS, max_depth=TREE_DEPTH)
#clf2.fit(X_train, y_train)
#y_train2 = clf2.predict(X_train)
#y_test2 = clf2.predict(X_test)
#     print(clf2.estimator_weights)
#print('accu (train): %5.2f' % (rmsle(y_train, y_train2)))
#print('accu (test): %5.2f' % (rmsle(y_test, y_test2)))


# Random forest
print('\nRandom Forest')
clf3 = RandomForestRegressor(max_depth=TREE_DEPTH, n_estimators=NUM_WEAK_CLASSIFIERS)
clf3.fit(X_train, y_train)
y_train3 = clf3.predict(X_train)
y_test3 = clf3.predict(X_test)
print('accu (train): %5.2f' % (rmsle(y_train, y_train3)))
print('accu (test): %5.2f' % (rmsle(y_test, y_test3)))

# Qboost
print('\nQBoost')
clf4 = QBoostRegressor(n_estimators=NUM_WEAK_CLASSIFIERS, max_depth=TREE_DEPTH)
clf4.fit(X_train, y_train, emb_sampler, lmd=lmd, **DW_PARAMS)
y_train4 = clf4.predict(X_train)
y_test4 = clf4.predict(X_test)
print(clf4.estimator_weights)
print('accu (train): %5.2f' % (rmsle(y_train, y_train4)))
print('accu (test): %5.2f' % (rmsle(y_test, y_test4)))

regressor_list = [clf0, clf1, clf3, clf4]

for i in range(3):
    regressor_list += regressor_list
# QboostPlus
print('\nQBoostPlus')
clf5 = QboostPlusRegression(regressor_list)
clf5.fit(X_train, y_train, emb_sampler, lmd=lmd, **DW_PARAMS)
y_train5 = clf5.predict(X_train)
y_test5 = clf5.predict(X_test)
print(clf5.estimator_weights)
print('accu (train): %5.2f' % (rmsle(y_train, y_train5)))
print('accu (test): %5.2f' % (rmsle(y_test, y_test5)))

