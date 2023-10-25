# Run Random Forest classifier to predict subjective refraction from objective measurements
import sys
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
# from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import json
from sklearn.feature_selection import mutual_info_classif
import scipy.stats
# from sklearn.ensemble import GradientBoostingClassifier

# Remove previous instances of the autorefeyelib classes
for kIdx in list(sys.modules.keys()):
    if kIdx.find('autorefeyelib')!=-1:
        print(f'deleting {kIdx}')
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

# Load parameters
# with open(os.path.join(os.path.dirname(__file__),'..','..','autorefeyelib','Refraction',"refraction_params.json")) as paramsFile:
#     params = json.load(paramsFile)

sphDelta = list(np.arange(params.inclusion_criteria['SphereDelta'][0],params.inclusion_criteria['SphereDelta'][1]+0.25,0.25))
cylDelta = list(np.arange(params.inclusion_criteria['CylinderDelta'][0],params.inclusion_criteria['CylinderDelta'][1]+0.25,0.25))

# Training db
# params['inclusion_criteria'] = {}
Inp        = Input.Loader()
dbFilePath = os.path.join(os.path.dirname(__file__),'..','..','autorefeyelib','Refraction','data','Prevot_EMR_VX_jointDB_2022_02_03.csv')
Inp.Load(fileName = dbFilePath)
Inp.Parse(inclusion_criteria=params.inclusion_criteria)
data       = Inp.Both.copy()
labelsSph  = data['SphereDelta'].apply(Inp._AssignLabel,classes=sphDelta,groupEndBins=False)
labelsCyl  = data['CylinderDelta'].apply(Inp._AssignLabel,classes=cylDelta,groupEndBins=False)
validInds  = labelsSph.notna()&labelsCyl.notna()
labelsSph  = labelsSph.loc[validInds]
labelsCyl  = labelsCyl.loc[validInds]
fMatCyl    = data.loc[validInds,params.featuresCyl]
fMatSph    = data.loc[validInds,params.featuresSph]

# Create a histogram of the sphere and cylinder classes
# use the histogram as class weights for classification

sph_hist = np.zeros(len(sphDelta))
cyl_hist = np.zeros(len(cylDelta))
for sIdx in labelsSph:
    sph_hist[int(sIdx)]+=1
sph_hist/=len(labelsSph)
sph_weights = np.zeros(len(labelsSph))
for sIdx in range(len(labelsSph)):
    sph_weights[sIdx] = sph_hist[int(labelsSph.iloc[sIdx])]

for cIdx in labelsCyl:
    cyl_hist[int(cIdx)]+=1
cyl_hist/=len(labelsCyl)
cyl_weights = np.zeros(len(labelsCyl))
for cIdx in range(len(labelsCyl)):
    cyl_weights[cIdx] = cyl_hist[int(labelsCyl.iloc[cIdx])]

# Set indices for training and validation
allInds   = np.random.permutation(len(fMatCyl))
testInds  = allInds[:1000]
trainInds = allInds[1001:]

# Class statistics
numCyl = np.zeros(len(cylDelta))
numSph = np.zeros(len(sphDelta))
for cIdx in range(len(cylDelta)):
    numCyl[cIdx] = (data['CylinderDelta']==cylDelta[cIdx]).sum()
for sIdx in range(len(sphDelta)):
    numSph[sIdx] = (data['SphereDelta']==sphDelta[sIdx]).sum()

# testing db
# InpTesting = Input.Loader()
# testingDB  = os.path.join(os.path.dirname(__file__),'..','..','autorefeyelib','Refraction','data','EMR_VX_Joint_DB_CIV_2020_07_27.csv')
# InpTesting.Load(fileName=testingDB)
# InpTesting.Parse(inclusion_criteria=inclusion_criteria)

# ref_score1 = make_scorer(RefMetric1)
# ref_score2 = make_scorer(RefMetric2)
ref_score = make_scorer(RefMetric3)

# Analysis of prognostic factors (feature selection), min max mean, median, std
# - -----------------------------
means_sph = pd.DataFrame(index=sphDelta,columns=fMatSph.keys())
meds_sph  = pd.DataFrame(index=sphDelta,columns=fMatSph.keys())
stds_sph  = pd.DataFrame(index=sphDelta,columns=fMatSph.keys())
min_sph   = pd.DataFrame(index=sphDelta,columns=fMatSph.keys())
max_sph   = pd.DataFrame(index=sphDelta,columns=fMatSph.keys())

means_cyl = pd.DataFrame(index=cylDelta,columns=fMatCyl.keys())
meds_cyl  = pd.DataFrame(index=cylDelta,columns=fMatCyl.keys())
stds_cyl  = pd.DataFrame(index=cylDelta,columns=fMatCyl.keys())
min_cyl   = pd.DataFrame(index=cylDelta,columns=fMatCyl.keys())
max_cyl   = pd.DataFrame(index=cylDelta,columns=fMatCyl.keys())

sph_keys = fMatSph.keys()
for sIdx in range(len(sphDelta)):
    sph_inds = labelsSph==sIdx
    means_sph.loc[sphDelta[sIdx],sph_keys] = fMatSph.loc[sph_inds].mean()
    meds_sph.loc[sphDelta[sIdx],sph_keys]  = fMatSph.loc[sph_inds].median()
    stds_sph.loc[sphDelta[sIdx],sph_keys]  = fMatSph.loc[sph_inds].std()
    min_sph.loc[sphDelta[sIdx],sph_keys]   = fMatSph.loc[sph_inds].min()
    max_sph.loc[sphDelta[sIdx],sph_keys]   = fMatSph.loc[sph_inds].max()

cyl_keys = fMatCyl.keys()
for cIdx in range(len(cylDelta)):
    cyl_inds = labelsCyl==cIdx
    means_cyl.loc[cylDelta[cIdx],cyl_keys] = fMatCyl.loc[cyl_inds].mean()
    meds_cyl.loc[cylDelta[cIdx] ,cyl_keys] = fMatCyl.loc[cyl_inds].median()
    stds_cyl.loc[cylDelta[cIdx] ,cyl_keys] = fMatCyl.loc[cyl_inds].std ()
    min_cyl.loc[cylDelta[cIdx]  ,cyl_keys] = fMatCyl.loc[cyl_inds].min()
    max_cyl.loc[cylDelta[cIdx]  ,cyl_keys] = fMatCyl.loc[cyl_inds].max()

feat_score_sph = means_sph.std()/(means_sph.max()-means_sph.min())
feat_score_cyl = means_cyl.std()/(means_cyl.max()-means_cyl.min())


# _____ Fearture selection ______
# build a mask to select classes within the error range and outside it
z_sph        = np.where(np.asanyarray(sphDelta)==0)[0]
z_cyl        = np.where(np.asanyarray(cylDelta)==0)[0]
in_inds_sph  = [z_sph-1,z_sph,z_sph+1] # labels for which Delta_S is 0±0.25D
out_inds_sph = np.setdiff1d(range(len(sphDelta)),in_inds_sph)
in_inds_cyl  = [z_cyl-1,z_cyl,z_cyl+1] # labels for which Delta_C is 0±0.25D
out_inds_cyl = np.setdiff1d(range(len(cylDelta)),in_inds_cyl)
ks_score     = pd.Series(dtype=object)
ks_score_sph = pd.Series(dtype=object)
ks_score_cyl = pd.Series(dtype=object)
mask_sph     = labelsSph.isin(in_inds_sph)
mask_cyl     = labelsCyl.isin(in_inds_cyl)
mask         = mask_sph&mask_cyl

if params.dev['feature_selection']['method'].lower()=='ks-test':
    # Perform a kolmogorov-smirnoff test for distribution similarity, where The null hpothesis is that
    # the two distributions are similar
    print(f'[scrRunRfClassifier] Computing the K-S statistics sphere')

    for pIdx in params.featuresSph:
        ks_score_sph[pIdx] = scipy.stats.ks_2samp(data.loc[mask_sph,pIdx],data.loc[~mask_sph,pIdx],alternative='two-sided').statistic

    print(f'[scrRunRfClassifier] Computing the K-S statistics cylinder')
    for pIdx in params.featuresCyl:
        ks_score_cyl[pIdx] = scipy.stats.ks_2samp(data.loc[mask_cyl,pIdx],data.loc[~mask_cyl,pIdx],alternative='two-sided').statistic

    print(f'[scrRunRfClassifier] Computing the K-S statistics sphere+cylinder')
    for pIdx in np.unique(params.featuresSph+params.featuresCyl):
        ks_score[pIdx] = scipy.stats.ks_2samp(data.loc[mask,pIdx],data.loc[~mask,pIdx],alternative='two-sided').statistic

    k_sph        = ks_score_sph[params.featuresSph]
    n_feat_sph   = params.dev['feature_selection']['n_feat_sph']
    features_sph = k_sph[np.argsort(k_sph)[-n_feat_sph:]].index
    k_cyl        = ks_score_cyl[params.featuresCyl]
    n_feat_cyl   = params.dev['feature_selection']['n_feat_cyl']
    features_cyl = k_cyl[np.argsort(k_cyl)[-n_feat_cyl:]].index
elif params.dev['feature_selection']['method'].lower()=='mutual-information':
    # compute mutual information on the test data
    mi_score_sph = mutual_info_classif(fMatSph.iloc[testInds],labelsSph.iloc[testInds],
                                discrete_features='auto', n_neighbors=15,
                                copy=True, random_state=None)
    n_feat_sph   = params.dev['feature_selection']['n_feat_sph']
    # features_sph = fMatSph.keys()[np.argsort(mi_score_sph)[-n_feat_sph:]]
    features_sph = fMatSph.keys()[mi_score_sph>np.mean(mi_score_sph)]

    mi_score_cyl = mutual_info_classif(fMatCyl.iloc[testInds],labelsCyl.iloc[testInds],
                                discrete_features='auto', n_neighbors=10,
                                copy=True, random_state=None)
    n_feat_cyl   = params.dev['feature_selection']['n_feat_cyl']
    # features_cyl = fMatCyl.keys()[np.argsort(mi_score_cyl)[-n_feat_cyl:]]
    features_cyl = fMatCyl.keys()[mi_score_cyl>np.mean(mi_score_cyl)]
elif params.dev['feature_selection']['method'].lower()=='input':
    # use the input features
    features_sph = params.featuresSph
    features_cyl = params.featuresCyl

#___Hyperparameter tuning____
# Start with a grid-search with cross-validation to obtain an initial estimate for model hyperparameters
if params.dev['tune_hyperparameters']:
    clf_sph_gs = GridSearchCV(RandomForestClassifier(),
                            param_grid=params.dev["param_grid"],
                            scoring=ref_score,
                            verbose=2,
                            n_jobs=10)
    clf_sph_gs.fit(fMatSph.iloc[trainInds],labelsSph.iloc[trainInds])
    clf_cyl_gs = GridSearchCV(RandomForestClassifier(),
                            param_grid=params.dev["param_grid"],
                            scoring=ref_score,
                            verbose=2,
                            n_jobs=10)
    clf_cyl_gs.fit(fMatCyl.iloc[trainInds],labelsCyl.iloc[trainInds])

    # Create a refined Bayesian grid based on the grid-search cv for hyperparameter tuning
    param_grid_bayesian_sph = {
                            'max_depth':Integer(clf_sph_gs.best_params_['max_depth']-10,clf_sph_gs.best_params_['max_depth']+10),
                            'n_estimators':Integer(clf_sph_gs.best_params_['n_estimators']-50,clf_sph_gs.best_params_['n_estimators']+50),
                            'min_samples_split':Integer(2,10),
                            'max_leaf_nodes':Integer(2,10),
                            'criterion':Categorical(['gini','entropy'])
                            #   'ccp_alpha':Real(0,1e-5)
                            }
    param_grid_bayesian_cyl = {
                            'max_depth':Integer(clf_cyl_gs.best_params_['max_depth']-10,clf_cyl_gs.best_params_['max_depth']+10),
                            'n_estimators':Integer(clf_cyl_gs.best_params_['n_estimators']-50,clf_cyl_gs.best_params_['n_estimators']+50),
                            'min_samples_split':Integer(2,10),
                            'max_leaf_nodes':Integer(2,10),
                            'criterion':Categorical(['gini','entropy'])
                            #   'ccp_alpha':Real(0,1e-5)
                            }

    cv = StratifiedKFold(n_splits=5,shuffle=False, random_state=None)
    scv_sph = BayesSearchCV(RandomForestClassifier(),
                    param_grid_bayesian_sph,
                    scoring=ref_score,
                    refit=True, # fit best estimator with entire dataset
                    iid=False,
                    cv = cv, # cross validation generator
                    verbose=2,
                    n_jobs=10,
                    return_train_score=True,
                    n_iter=5)# number of parameter setting to try
    scv_cyl = BayesSearchCV(RandomForestClassifier(),
                    param_grid_bayesian_cyl,
                    scoring=ref_score,
                    iid=False,
                    verbose=2,
                    n_jobs=10,
                    return_train_score=True,
                    n_iter=5)

    clf_sph = Pipeline([
        ('feature_selection',SelectFromModel(clf_sph_gs.best_estimator_,prefit=False,threshold='mean')),
        ('classification', scv_sph)])

    clf_cyl = Pipeline([
        ('feature_selection', SelectFromModel(clf_cyl_gs.best_estimator_,prefit=False,threshold='mean')),
        ('classification', scv_cyl)])

    print('Fitting Random forest classifier: Sphere')
    clf_sph.fit(fMatSph.iloc[trainInds],labelsSph.iloc[trainInds])
    rf_model_sph              = clf_sph['classification'].best_estimator_
    rf_model_sph.featuresUsed = list(fMatSph.keys()[clf_sph['feature_selection'].get_support()])
    rf_model_sph.sphereDelta  = sphDelta
    rf_lab_sph                = rf_model_sph.predict(fMatSph.iloc[testInds][rf_model_sph.featuresUsed])

    print('Fitting Random forest classifier: Cylinder')
    clf_cyl.fit(fMatCyl.iloc[trainInds],labelsCyl.iloc[trainInds])
    rf_model_cyl               = clf_cyl['classification'].best_estimator_
    rf_model_cyl.featuresUsed  = list(fMatCyl.keys()[clf_cyl['feature_selection'].get_support()])
    rf_model_cyl.cylinderDelta = cylDelta
    rf_lab_cyl                 = rf_model_cyl.predict(fMatCyl.iloc[testInds][rf_model_cyl.featuresUsed])
else:
    # Train RF classifiers with default parameters using selected features
    rf_model_sph = RandomForestClassifier(n_estimators     = params.dev['sph_model_params']['n_estimators'],
                                         max_depth         = params.dev['sph_model_params']['max_depth'],
                                         verbose           = params.dev['sph_model_params']['verbose'],
                                         max_leaf_nodes    = params.dev['sph_model_params']['max_leaf_nodes'],
                                         min_samples_split = params.dev['sph_model_params']['min_samples_split'])
                                        #  n_jobs            = params['dev']['sph_model_params']['n_jobs'])
    rf_model_sph.fit(fMatSph.iloc[trainInds][features_sph],labelsSph.iloc[trainInds],sample_weight=sph_weights[trainInds])
    rf_model_sph.featuresUsed = list(features_sph)
    rf_model_sph.sphereDelta  = sphDelta
    rf_lab_sph   = rf_model_sph.predict(fMatSph.iloc[testInds][features_sph])

    rf_model_cyl = RandomForestClassifier(n_estimators     = params.dev['cyl_model_params']['n_estimators'],
                                         max_depth         = params.dev['cyl_model_params']['max_depth'],
                                         verbose           = params.dev['cyl_model_params']['verbose'],
                                         max_leaf_nodes    = params.dev['cyl_model_params']['max_leaf_nodes'],
                                         min_samples_split = params.dev['cyl_model_params']['min_samples_split'])
                                        #  n_jobs            = params['dev']['cyl_model_params']['n_jobs'])

    rf_model_cyl.fit(fMatCyl.iloc[trainInds][features_cyl],labelsCyl.iloc[trainInds],sample_weight=cyl_weights[trainInds])
    rf_model_cyl.featuresUsed  = list(features_cyl)
    rf_model_cyl.cylinderDelta = cylDelta
    rf_lab_cyl = rf_model_cyl.predict(fMatCyl.iloc[testInds][features_cyl])

    # train binary classifiers to predict if a measurement is within or outside the aeptable error range
    print('\n ___Training binary classifier sphere')
    rf_model_bin_sph = RandomForestClassifier(n_estimators = params.dev['sph_model_params']['n_estimators'],
                                         max_depth         = params.dev['sph_model_params']['max_depth'],
                                         verbose           = params.dev['sph_model_params']['verbose'],
                                         max_leaf_nodes    = params.dev['sph_model_params']['max_leaf_nodes'],
                                         min_samples_split = params.dev['sph_model_params']['min_samples_split'])
    rf_model_bin_sph.fit(fMatSph.iloc[trainInds],mask_sph.iloc[trainInds])
    rf_label_bin_sph    = rf_model_bin_sph.predict(fMatSph.iloc[testInds])
    score_model_bin_sph = (rf_label_bin_sph==mask_sph.iloc[testInds]).sum()/len(testInds)
    print(f'Binary classifier sphere accuracy: {score_model_bin_sph}')

    print('\n___Training binary classifier cyinder')
    rf_model_bin_cyl = RandomForestClassifier(n_estimators = params.dev['cyl_model_params']['n_estimators'],
                                         max_depth         = params.dev['cyl_model_params']['max_depth'],
                                         verbose           = params.dev['cyl_model_params']['verbose'],
                                         max_leaf_nodes    = params.dev['cyl_model_params']['max_leaf_nodes'],
                                         min_samples_split = params.dev['cyl_model_params']['min_samples_split'])
    rf_model_bin_cyl.fit(fMatCyl.iloc[trainInds],mask_cyl.iloc[trainInds])
    rf_label_bin_cyl    = rf_model_bin_cyl.predict(fMatCyl.iloc[testInds])
    score_model_bin_cyl = (rf_label_bin_cyl==mask_cyl.iloc[testInds]).sum()/len(testInds)
    print(f'Binary classifier cylinder accuracy: {score_model_bin_cyl}')

# Compute the mean absolute error of each trained model
mae_predicted_sph   = (rf_lab_sph-labelsSph.iloc[testInds]).abs().mean()*0.25
mae_predicted_cyl   = (rf_lab_cyl-labelsCyl.iloc[testInds]).abs().mean()*0.25

## Display results and statistics
print('____Results Random Forest___')
lims =[0.0,0.25,0.5]
for dIdx in range(len(lims)):
    print(f"|S|<={lims[dIdx]}: {((rf_lab_sph-labelsSph.iloc[testInds]).abs()<=dIdx).sum()/len(testInds):.2f}")
for dIdx in range(len(lims)):
    print(f"|C|<={lims[dIdx]}: { ((rf_lab_cyl-labelsCyl.iloc[testInds]).abs()<=dIdx).sum()/len(testInds):.2f}")
for dIdx in range(len(lims)):
    print(f"|S|&|C|<={lims[dIdx]}: { (((rf_lab_sph-labelsSph.iloc[testInds]).abs()<=dIdx)&((rf_lab_cyl-labelsCyl.iloc[testInds]).abs()<=dIdx)).sum()/len(testInds):.2f}")
print(f"MAE sphere: {mae_predicted_sph:.2f}")
print(f"MAE cylinder: {mae_predicted_cyl:.2f}")

# Present refractometer results for comparison
for dIdx in range(len(lims)):
    print(f"AutoRef |S|<={lims[dIdx]}: {(data['SphereDelta'].abs()<=lims[dIdx]).sum()/len(data):.2f}")
for dIdx in range(len(lims)):
    print(f"AutoRef |C|<={lims[dIdx]}: {(data['CylinderDelta'].abs()<=lims[dIdx]).sum()/len(data):.2f}")
for dIdx in range(len(lims)):
    print(f"AutoRef |S|&|C|<={lims[dIdx]}: {((data['CylinderDelta'].abs()<=lims[dIdx])&(data['SphereDelta'].abs()<=lims[dIdx])).sum()/len(data):.2f}")
mae_autoref_sph = (data.iloc[testInds]['EMR:VisualAcuitySphere']  - data.iloc[testInds]['WF_SPHERE_R_3']).abs().mean()
mae_autoref_cyl = (data.iloc[testInds]['EMR:VisualAcuityCylinder']- data.iloc[testInds]['WF_CYLINDER_R_3']).abs().mean()
print(f'AutoRef MAE sphere: {mae_autoref_sph:.2f}')
print(f'AutoRef MAE cylinder: {mae_autoref_cyl:.2f}')


print('\nClass statistics')
print('________________')
print(f'Using Sphere delta of                     {sphDelta}')
print(f'Number of observations in Sphere class:   {numSph}')
print(f'Using Cylinder delta                      {cylDelta}')
print(f'Number of observations in Cylinder class: {numCyl}')

# Statistics on the objective data
left_err_sph  = []
right_err_sph = []
left_err_cyl  = []
right_err_cyl = []
both_err_sph  = []
both_err_cyl  = []
for lIdx in lims:
    left_err_sph.append((Inp.Left['SphereDelta'].abs()<=lIdx).sum()/len(Inp.Left))
    right_err_sph.append((Inp.Right['SphereDelta'].abs()<=lIdx).sum()/len(Inp.Right))
    both_err_sph.append((Inp.Both['SphereDelta'].abs()<=lIdx).sum()/len(Inp.Both))
    left_err_cyl.append((Inp.Left['CylinderDelta'].abs()<=lIdx).sum()/len(Inp.Left))
    right_err_cyl.append((Inp.Right['CylinderDelta'].abs()<=lIdx).sum()/len(Inp.Right))
    both_err_cyl.append((Inp.Both['CylinderDelta'].abs()<=lIdx).sum()/len(Inp.Both))

if params.dev['export_classifiers']:
    print("Exporting models")
    joblib.dump(rf_model_sph,'SphModel.sav',compress=9)
    joblib.dump(rf_model_cyl,'CylModel.sav',compress=9)
    # export parameters associated with the models
    # with open("refraction_params.json","w") as param_file:
    #     param_file.writelines(json.dumps(params.__dict__))

