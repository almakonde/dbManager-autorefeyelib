"""
    scrupt to predict the fincal refraction from objective measurements powst catarct surgery
"""
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dateutil import parser as dateParser
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import pickle
for kIdx in list(sys.modules.keys()):
    if kIdx.find('autorefeyelib')!=-1:
        print(f'delteting {kIdx}')
        del sys.modules[kIdx]
from autorefeyelib.IOLpower import Predictor as iolPredictor
from autorefeyelib.IOLpower import Classifier as iolClassifier

# Parameters
#------------
percTraining      = 0.9   # fraction of the db used for training
numExp            = 3     # number of experiments
alphaDelta        = 0.05
alpha             = np.arange(0,1+alphaDelta,alphaDelta)  # weight assigned to refraction error out of the goal function
Aconst            = 118.9 # Manufacturar IOL A constant
n_a               = 1.3375 # refractive index
exportModel       = False
features          = ['age','meanK','ACD','WTW','axialLength','targetRefraction']
validRanges       = {"age":[40,100],
                     "meanK":[35,47],
                     "axialLength":[20,40],
                     "ACD":[2,5],
                     "WTW":[10,13],
                     "targetRefraction":[-10,10]}
# jointDBpath  = os.path.join(os.getcwd(),'..','Data','jointIOLEMR.csv')

# define parameter grid for hyperparameter tuning
paramGrid = {'n_estimators':[50,60,70,80,90,100],
             'criterion':['entropy','gini'],
             'max_depth':[50,100,150,200,250,300],
             'ccp_alpha':[1e-3, 1e-4, 1e-5, 1e-6]}

# initialization
e = iolPredictor.Predictor()
c = iolClassifier.Classifier()
# e.LoadData(jointDBpath)#  preOpFile, opDayFile, postOpFile,iolData,iolConstants)

### prepare a combined right/left eye database parameters
data                  = e.data.copy()
#  compute time difference between measurement and operation
for dIdx in range(len(data)):

    iolDate  = dateParser.parse(data.loc[data.index[dIdx],'measurement_date'])
    emrDate  = dateParser.parse(data.loc[data.index[dIdx],'ExamDate'])
    data.loc[data.index[dIdx],'followup_days'] = (emrDate-iolDate).days

data = data.loc[data['followup_days']>=0]
C                     = pd.DataFrame()
r1                    = data['l_radius_r1'].append(data['r_radius_r1']) # cornea radius
r2                    = data['l_radius_r2'].append(data['r_radius_r2']) # cornea radius
Rc                    = 0.5*(r1+r2)
k1                    = 1000*(n_a-1)/r1
k2                    = 1000*(n_a-1)/r2
C['meanK']            = 0.5*(k1+k2)# mean keratometry
C['axialLength']      = data['l_axial_length_mean'].append(data['r_axial_length_mean'])
C['ACD']              = data['l_acd_mean'].append(data['r_acd_mean'])
C['targetRefraction'] = data['IolDesiredRefractionPostOp_Left'].append(data["IolDesiredRefractionPostOp_Right"])
C.loc[C['targetRefraction'].isna(),'targetRefraction'] = 0.0 # assume missing value represent emmetropia
C['finalRefraction']  = data['IolFinalRefraction_Left'].append(data['IolFinalRefraction_Right'])
C['age']              = data['Age'].append(data['Age'])
C['WTW']              = data['l_wtw_mean'].append(data['r_wtw_mean'])
C['Pt']               = data['IolPower_Left'].append(data['IolPower_Right']) # implanted IOL power
C['ExamDate']         = data['ExamDate'].append(data['ExamDate'])
C['Gender']           = data['Gender'].append(data['Gender'])

# Filter DB by valid ranges
for kIdx in validRanges.keys():
    # find values out of range
    inds =(C[kIdx]<validRanges[kIdx][0]) | (C[kIdx]>validRanges[kIdx][1])
    C.loc[inds,kIdx]= None
C.dropna(inplace=True)
C.index = range(len(C))
# define feature matrix
featureMat = C[features]