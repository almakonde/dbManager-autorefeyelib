# scrRunRFDirectionalClassifier
#  Predict the direction of needed correction from objective to subjective
import sys
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
# remove instances of the autorefeyelib classes
for kIdx in list(sys.modules.keys()):
    if kIdx.find('autorefeyelib')!=-1:
        print(f'delteting {kIdx}')
        del sys.modules[kIdx]
from autorefeyelib.Refraction import Input
inclusion_criteria = {
                     'Age':[15,100],
                     'WF_SPHERE_R_3':[-5,6],                       # sphere
                     'WF_CYLINDER_R_3':[-6,0],                     # cylinder
                     'WF_RT_Fv_Zernike_Photo_di_Z_2_-2':[-1,1],    # astigmatism
                     'WF_RT_Fv_Zernike_Photo_di_Z_4_0':[-0.2,0.2], # spherical aberration
                     'WF_RT_Fv_Zernike_Photo_di_Z_3_1':[-0.3,0.3], # primary coma photo
                     'WF_RT_Fv_Zernike_Meso_di_Z_2_-2':[-2,2],     # astigmatism
                     'WF_RT_Fv_Zernike_Meso_di_Z_4_0':[-0.3, 0.3], # spherical abberation diopter
                     'WF_RT_Fv_Zernike_Meso_di_Z_3_1':[-0.4,0.4],  # primary coma meso
                     'Pachy_MEASURE_Acd':[2,5],                    # anterior chamber depth
                     'Pachy_MEASURE_WhiteToWhite':[8,16],          # white to white
                     'Pachy_MEASURE_KappaAngle':[0,25],            # kappa angle
                     'Pachy_MEASURE_Thickness':[400,600],          # pachimetry
                     'SphereDelta':[-1.0,1.00],
                     'CylinderDelta':[-1.0,0.5]
                     }
dbFilePath = os.path.join(os.path.dirname(__file__),'..','..','autorefeyelib','Refraction','data','Prevot_EMR_VX_jointDB_2021_05_18.csv')
exportClassifieres = False

featuresCyl  = [
                'Age',
                # 'Gender',
                # 'Topo_Sim_K_K1',
                # 'Topo_Sim_K_K2',
                # 'WF_SPHERE_R_3',
                'WF_CYLINDER_R_3',
                # 'WF_AXIS_R_3',
                # 'WF_RT_Fv_Zernike_Photo_di_Z_2_-2',
                # 'WF_RT_Fv_Zernike_Photo_di_Z_4_0',
                # 'WF_RT_Fv_Zernike_Photo_di_Z_3_1',
                # 'WF_RT_Fv_Meso_PupilRadius',
                # 'WF_RT_Fv_Zernike_Meso_di_Z_2_-2',
                # 'WF_RT_Fv_Zernike_Meso_di_Z_4_0',
                # 'WF_RT_Fv_Zernike_Meso_di_Z_3_1',
                # 'Pachy_MEASURE_Acd',
                # 'ObjectiveSphericalEquivalent',
                # 'WF_RT_Fv_Zernike_Meso_Coma',
                # 'WF_RT_Fv_Zernike_Meso_Trefoil',
                # 'Topo_GENERAL_Geo_Q',
                # 'Topo_GENERAL_Geo_P',
                # 'Topo_GENERAL_Geo_e',
                # 'Pachy_MEASURE_WhiteToWhite',
                # 'Pachy_MEASURE_KappaAngle',
                # 'Pachy_MEASURE_Thickness',
                # 'Tono_MEASURE_Average',
                # 'Topo_KERATOCONUS_Kpi',
                # 'AnteriorChamberVol',
                # 'AcdToPupilRadius_Ratio',
                # 'PupilRadiusToW2W_Ratio',
                # 'kRatio',
                # 'J0_3',
                'J45_3',
                # 'BlurStrength_3'
                ]

featuresSph = [
                'Age',
                # 'Gender',
                'Topo_Sim_K_K1',
                'Topo_Sim_K_K2',
                'WF_SPHERE_R_3',
                'WF_CYLINDER_R_3',
                # 'WF_AXIS_R_3',
                # 'WF_RT_Fv_Zernike_Photo_di_Z_2_-2',
                # 'WF_RT_Fv_Zernike_Photo_di_Z_4_0',
                # 'WF_RT_Fv_Zernike_Photo_di_Z_3_1',
                # 'WF_RT_Fv_Meso_PupilRadius',
                # 'WF_RT_Fv_Zernike_Meso_di_Z_2_-2',
                # 'WF_RT_Fv_Zernike_Meso_di_Z_4_0',
                # 'WF_RT_Fv_Zernike_Meso_di_Z_3_1',
                'Pachy_MEASURE_Acd',
                'Pachy_MEASURE_WhiteToWhite',
                'Pachy_MEASURE_KappaAngle',
                'Pachy_MEASURE_Thickness',
                'Tono_MEASURE_Average',
                'Topo_KERATOCONUS_Kpi',
                'AnteriorChamberVol',
                'AcdToPupilRadius_Ratio',
                'PupilRadiusToW2W_Ratio',
                'ObjectiveSphericalEquivalent',
                'WF_RT_Fv_Zernike_Meso_Coma',
                'WF_RT_Fv_Zernike_Meso_Trefoil',
                # 'kRatio',
                'J0_3',
                'J45_3',
                'BlurStrength_3'
                ]

Inp        = Input.Loader()
Inp.Load(fileName = dbFilePath)
Inp.Parse(inclusion_criteria=inclusion_criteria)
sphDelta     = list(np.arange(inclusion_criteria['SphereDelta'][0],inclusion_criteria['SphereDelta'][1]+0.25,0.25))
cylDelta     = list(np.arange(inclusion_criteria['CylinderDelta'][0],inclusion_criteria['CylinderDelta'][1]+0.25,0.25))
data         = Inp.Both

# validIndsSph = (data['SphereDelta']<=max(sphDelta))&(data['SphereDelta']>=min(sphDelta))
# validIndsCyl = (data['CylinderDelta']<=max(cylDelta))&(data['CylinderDelta']>=min(cylDelta))
# validInds    = validIndsSph&validIndsCyl
# data         = data.loc[validInds] # truncate by valid
fMatDirCyl   = data[featuresCyl]
fMatDirSph   = data[featuresSph]

# assign 0 when objective and subjective are equal
# assign 1 when objective is larger than subjective
# assign 2 when objective is smaller than subjective
zeroSph      = data['SphereDelta']==0
reduceSph    = data['SphereDelta']>0
addSph       = data['SphereDelta']<0
labelsDirSph = np.zeros(len(zeroSph),dtype=int)
labelsDirSph[zeroSph.values]   = 0
labelsDirSph[reduceSph.values] = 1
labelsDirSph[addSph.values]    = 2
zeroCyl      = data['CylinderDelta']==0
reduceCyl    = data['CylinderDelta']>0
addCyl       = data['CylinderDelta']<0
labelsDirCyl = np.zeros(len(zeroCyl),dtype=int)
labelsDirCyl[zeroCyl.values]   = 0
labelsDirCyl[reduceCyl.values] = 1
labelsDirCyl[addCyl.values]    = 2

# Train a classifier for the direction of correction

rf_model_dir_sph=RandomForestClassifier(max_depth=None,
                                        n_estimators=500,
                                        # min_samples_split=50,
                                        min_impurity_split = None,
                                        # criterion = 'gini',
                                        warm_start=False, # n_estimators increases with each trial to fit new trees
                                        oob_score=True)
                                        # ccp_alpha=0.01)
# rf_model_dir_sph = LogisticRegression()
# rf_model_dir_sph = AdaBoostClassifier(n_estimators=500)
rf_model_dir_cyl=RandomForestClassifier(max_depth=None,
                                        n_estimators=500,
                                        # min_samples_split=5,
                                        # min_impurity_split = None,
                                        # criterion = 'gini',
                                        warm_start=False, # n_estimators increases with each trial to fit new trees
                                        oob_score=True
                                        )
                                    # ccp_alpha=0.05)
# rf_model_dir_cyl = AdaBoostClassifier(n_estimators=500,learning_rate=0.5,base_estimator=LogisticRegression())

allInds   = np.random.permutation(len(data))
testInds  = allInds[:1000] # test on 1000 randomly selected samples
trainInds = allInds[1000:]

# # Compute the medians of features for each class
# cylMedians = pd.DataFrame(index=[0,1,2])
# fCyl       = fMatDirCyl.iloc[trainInds]
# for cIdx in featuresCyl:
#     zMed    = zeroCyl.iloc[trainInds]
#     cylMedians.loc[0,cIdx] = fCyl.loc[zMed,cIdx].median()
#     sMed    = reduceCyl.iloc[trainInds]
#     cylMedians.loc[1,cIdx] = fCyl.loc[sMed,cIdx].median()
#     aMed    = addCyl.iloc[trainInds]
#     cylMedians.loc[2,cIdx] = fCyl.loc[aMed,cIdx].median()

# sphMedians = pd.DataFrame(index=[0,1,2])
# fSph       = fMatDirSph.iloc[trainInds]
# for cIdx in featuresSph:
#     zMed    = zeroSph.iloc[trainInds]
#     sphMedians.loc[0,cIdx] = fSph.loc[zMed,cIdx].median()
#     sMed    = reduceSph.iloc[trainInds]
#     sphMedians.loc[1,cIdx] = fSph.loc[sMed,cIdx].median()
#     aMed    = addSph.iloc[trainInds]
#     sphMedians.loc[2,cIdx] = fSph.loc[aMed,cIdx].median()

# labelsCylMedian = pd.DataFrame()
# labelsSphMedian = pd.DataFrame()
# for tIdx in range(len(testInds)):
#     indCyl = fMatDirCyl.index[testInds[tIdx]]
#     indSph = fMatDirSph.index[testInds[tIdx]]
#     labelsCylMedian.loc[indCyl,'Label'] = (((cylMedians-fMatDirCyl.loc[indCyl,:])**2).sum(axis=1)**0.5).argmin()
#     labelsSphMedian.loc[indSph,'Label'] = (((sphMedians-fMatDirSph.loc[indSph,:])**2).sum(axis=1)**0.5).argmin()
# print(f"k-median classifier sph. Accuracy: {(labelsSphMedian['Label'].values==labelsDirSph[testInds]).sum()/len(testInds)}")
# print(f"k-median classifier cyl. Accuracy:{(labelsCylMedian['Label'].values==labelsDirCyl[testInds]).sum()/len(testInds)}")

print('Fitting Random forest directional classifier Sphere')
rf_model_dir_sph.fit(fMatDirSph.iloc[trainInds],labelsDirSph[trainInds])
rf_model_dir_sph.featuresUsed = featuresSph
rf_lab_sph                    = rf_model_dir_sph.predict(fMatDirSph.iloc[testInds])

print('Fitting Random forest directional classifier Cylinder')
rf_model_dir_cyl.fit(fMatDirCyl.iloc[trainInds],labelsDirCyl[trainInds])
rf_model_dir_cyl.featuresUsed  = featuresCyl
rf_lab_cyl                     = rf_model_dir_cyl.predict(fMatDirCyl.iloc[testInds])

# Summary of results
print(f'Accuracy Sph: {np.sum(rf_lab_sph.round()==labelsDirSph[testInds])/len(testInds)}')
print(f'Accuracy Cyl: {np.sum(rf_lab_cyl.round()==labelsDirCyl[testInds])/len(testInds)}')
print(f'Accuracy total {np.sum((rf_lab_sph.round()==labelsDirSph[testInds])&(rf_lab_cyl==labelsDirCyl[testInds]))/len(testInds)}')

# Create confusion matrix sph
confMatDirSph = np.zeros((3,3),dtype=int)
for sIdx in range(len(rf_lab_sph)):
    confMatDirSph[rf_lab_sph[sIdx].round(),labelsDirSph[testInds[sIdx]]]+=1
# Create confusion matrix cylinder
confMatDirCyl = np.zeros((3,3),dtype=int)
for cIdx in range(len(rf_lab_cyl)):
    confMatDirCyl[rf_lab_cyl[cIdx].round(),labelsDirCyl[testInds[cIdx]]]+=1