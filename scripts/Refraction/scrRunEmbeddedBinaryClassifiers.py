# Run Random Forest classifier to predict subjective refraction from objective measurements
import sys
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
# from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.feature_selection import mutual_info_classif
import scipy.stats
# from sklearn.ensemble import GradientBoostingClassifier

# Remove previous instances of the autorefeyelib classes
for kIdx in list(sys.modules.keys()):
    if kIdx.find('autorefeyelib')!=-1:
        print(f'delteting {kIdx}')
        del sys.modules[kIdx]
from autorefeyelib.Refraction import Input
from autorefeyelib.Refraction import refraction_params as params

# TODO: add localOutlierFactor calculation on the whole dataset after applying inclusion criteria
# TODO: compute the expected number of observations in each class after the training data is removed from the set
# That is, the possible classes are derived from the remaining fraction for training and according to the distribution
# of classes in the population
def ConfMatrix(labels_gt, labels_predicted,numClasses):
    """
     construct a confusion matrix based on integer class assignment
    """
    confMat   = np.zeros((numClasses,numClasses),dtype=int)
    labels_gt = list(labels_gt)
    labels_predicted = list(labels_predicted)
    if len(labels_gt)!=len(labels_predicted):
        raise ValueError('The length of the ground-truth labels vector and predicted labels must be similar')
    for lIdx in range(len(labels_gt)):
        confMat[int(labels_gt[lIdx]),int(labels_predicted[lIdx])]+=1
    return confMat

def RefMetric1(label,predicted_label):
    # score acording to exact match and those in a range of 0.25 diopter (i.e. neighboring labels)
    diff = np.abs(label-predicted_label) # assuming successive labels indicate diopter difference of 0.25D
    return np.sum(diff<=1)/np.sum(diff>1)

def RefMetric2(label,predicted_label):
    diff = np.abs(label-predicted_label) # assuming successive labels indicate diopter difference of 0.25D
    return  np.sum(diff==0)+(np.sum(diff==1))*np.exp(-np.sum(diff>1))

def RefMetric3(label,predicted_label):
    """
     Assign score (to be maximized) to  a cross-validation batch.
     The score is maximal when all labels predicted equal gt labels

     Parameters:
     -----------
     label, array int
       array of gt labels
     predicted_label, array int
       array of predicted labels

     Returns:
     -----------
      score, float
    """
    diff = np.abs(label-predicted_label) # assuming successive labels indicate diopter difference of 0.25D
    return np.sum(np.exp(-diff))

def ClassToDelta(classNum,deltas):
    """
     Assuming class numbers are intgers
    """
    d = np.zeros(len(classNum))
    for cIdx in range(len(classNum)):
        try:
            d[cIdx] = deltas[int(classNum[cIdx])]
        except:
            d[cIdx] = None
    return d

sphDelta = list(np.arange(params.inclusion_criteria['SphereDelta'][0],params.inclusion_criteria['SphereDelta'][1]+0.25,0.25))
cylDelta = list(np.arange(params.inclusion_criteria['CylinderDelta'][0],params.inclusion_criteria['CylinderDelta'][1]+0.25,0.25))

# Training db
# params['inclusion_criteria'] = {}
Inp        = Input.Loader()
dbFilePath = os.path.join(os.path.dirname(__file__),'..','..','autorefeyelib','Refraction','data','Prevot_EMR_VX_jointDB_2021_05_18.csv')
Inp.Load(fileName = dbFilePath)
Inp.Parse(inclusion_criteria=params.inclusion_criteria)
data       = Inp.Both.copy()

# create a sequence of embedded binary classifiers
fMatCyl    = data.loc[:,params.featuresCyl]
fMatSph    = data.loc[:,params.featuresSph]

# Set indices for training and validation
allInds   = np.random.permutation(len(fMatCyl))
testInds  = allInds[:1000]
trainInds = allInds[1001:]

ref_score = make_scorer(RefMetric3)


thresh          = [0]
num_levels      = len(thresh)
sph_class_stack = np.ndarray(num_levels,dtype=RandomForestClassifier)
cyl_class_stack = np.ndarray(num_levels,dtype=RandomForestClassifier)
pred_sph        = np.ndarray(shape = (len(testInds),num_levels))
pred_cyl        = np.ndarray(shape = (len(testInds),num_levels))
accuracy_sph    = np.zeros(num_levels)
accuracy_cyl   = np.zeros(num_levels)
for sIdx in range(num_levels):
    labels_sph = data.loc[trainInds,'SphereDelta'].abs()<=thresh[sIdx]
    sample_weights_sph              = np.ones(len(labels_sph))
    sample_weights_sph[labels_sph]  = len(labels_sph)/labels_sph.sum()
    sample_weights_sph[~labels_sph] = len(labels_sph)/(~labels_sph).sum()

    labels_cyl = data.loc[trainInds,'CylinderDelta'].abs()<=thresh[sIdx]
    sample_weights_cyl              = np.ones(len(labels_cyl))
    sample_weights_cyl[labels_cyl]  = len(labels_cyl)/labels_cyl.sum() # inverse of their frequency
    sample_weights_cyl[~labels_cyl] = len(labels_cyl)/(~labels_cyl).sum()
    print(f'\n__Training sphere binary classifier for threshold = {thresh[sIdx]}')
    sph_class_stack[sIdx] = RandomForestClassifier(n_estimators      = 10000,#params.dev['sph_model_params']['n_estimators'],
                                                   max_depth         = 2,#params.dev['sph_model_params']['max_depth'],
                                                   verbose           = params.dev['sph_model_params']['verbose'],
                                                   max_leaf_nodes    = params.dev['sph_model_params']['max_leaf_nodes'],
                                                   min_samples_split = params.dev['sph_model_params']['min_samples_split'])
    # sph_class_stack[sIdx] = DecisionTreeClassifier()
    sph_class_stack[sIdx].fit(fMatSph.loc[trainInds,:], labels_sph,sample_weight=sample_weights_sph)
    pred_sph[:,sIdx]   = sph_class_stack[sIdx].predict(fMatSph.loc[testInds,:])
    accuracy_sph[sIdx] = (pred_sph[:,sIdx]==(data.loc[testInds,'SphereDelta'].abs()<=thresh[sIdx])).sum()/len(trainInds)
    print(f'Accuracy={accuracy_sph[sIdx]:.2f}')
    print(f'\n__Training cylinder binary classifier for threshold = {thresh[sIdx]}')
    # cyl_class_stack[sIdx] = RandomForestClassifier(n_estimators     = 1000,#params.dev['cyl_model_params']['n_estimators'],
    #                                               max_depth         = 2,#params.dev['cyl_model_params']['max_depth'],
    #                                               verbose           = params.dev['cyl_model_params']['verbose'],
    #                                               max_leaf_nodes    = params.dev['cyl_model_params']['max_leaf_nodes'],
    #                                               min_samples_split = params.dev['cyl_model_params']['min_samples_split'])
    cyl_class_stack[sIdx] = DecisionTreeClassifier()
    cyl_class_stack[sIdx].fit(fMatCyl.loc[trainInds,:], labels_cyl,sample_weight=sample_weights_cyl)
    pred_cyl[:,sIdx]   = cyl_class_stack[sIdx].predict(fMatCyl.loc[testInds,:])
    accuracy_cyl[sIdx] = (pred_cyl[:,sIdx]==(data.loc[testInds,'CylinderDelta'].abs()<=thresh[sIdx])).sum()/len(trainInds)
    print(f'Accuracy={accuracy_cyl[sIdx]:.2f}')


# Compute the mean absolute error of each trained model
# mae_predicted_sph   = (rf_lab_sph-labelsSph.iloc[testInds]).abs().mean()*0.25
# mae_predicted_cyl   = (rf_lab_cyl-labelsCyl.iloc[testInds]).abs().mean()*0.25

