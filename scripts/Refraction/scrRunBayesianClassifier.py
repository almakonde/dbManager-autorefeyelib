import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

# remove instances of the autorefeyelib classes
for kIdx in list(sys.modules.keys()):
    if kIdx.find('autorefeyelib')!=-1:
        print(f'delteting {kIdx}')
        del sys.modules[kIdx]

from autorefeyelib.Refraction import BayesienRefraction
from autorefeyelib.Refraction import Classifier as VXClassifier
from autorefeyelib.Refraction import Input

#%%
# p = VXClassifier.Classifier()

# p.Load(fileName = os.path.join(os.path.dirname(__file__),'..','Refraction','data','Prevot_EMR_VX_jointDB_2021_03_09.csv'))
# p.Parse(handleMissing='impute',imputeMethod='univariate')

l  = Input.Loader()
l.Load(fileName = os.path.join(os.path.dirname(__file__),'..','..','autorefeyelib','Refraction','data','Prevot_EMR_VX_jointDB_2021_03_09.csv'))
l.Parse(handleMissing='impute',imputeMethod='univariate')



bCyl = BayesienRefraction.Classifier()
bSph = BayesienRefraction.Classifier()

sphDelta     = list(np.arange(-1,1.5,0.25))
cylDelta     = list(np.arange(-1.25,0.75,0.25))
data         = l.Both.dropna()
validIndsSph = (data['SphereDelta']<=max(sphDelta))&(data['SphereDelta']>=min(sphDelta))
validIndsCyl = (data['CylinderDelta']<=max(cylDelta))&(data['CylinderDelta']>=min(cylDelta))
validInds    = validIndsSph&validIndsCyl
labelsSph    = data.loc[validInds,'SphereDelta'].apply(bSph._AssignLabel,args=[sphDelta])
labelsCyl    = data.loc[validInds,'CylinderDelta'].apply(bCyl._AssignLabel,args=[cylDelta])


# define features for classification
featuresCyl  = [
                # 'Age',
                # 'Gender',
                'Topo_Sim_K_K1',
                'Topo_Sim_K_K2',
                'WF_SPHERE_R_3',
                'WF_CYLINDER_R_3',
                # 'WF_AXIS_R_3',
                # 'WF_RT_Fv_Zernike_Photo_di_Z_2_-2',
                'WF_RT_Fv_Zernike_Photo_di_Z_4_0',
                # 'WF_RT_Fv_Zernike_Photo_di_Z_3_1',
                # 'WF_RT_Fv_Meso_PupilRadius',
                # 'WF_RT_Fv_Zernike_Meso_di_Z_2_-2',
                'WF_RT_Fv_Zernike_Meso_di_Z_4_0',
                # 'WF_RT_Fv_Zernike_Meso_di_Z_3_1',
                'Pachy_MEASURE_Acd',
                # 'Pachy_MEASURE_WhiteToWhite',
                # 'Pachy_MEASURE_KappaAngle',
                # 'Pachy_MEASURE_Thickness',
                # 'Tono_MEASURE_Average',
                # 'Topo_KERATOCONUS_Kpi',
                # 'AnteriorChamberVol',
                # 'AcdToPupilRadius_Ratio',
                # 'PupilRadiusToW2W_Ratio',
                'kRatio',
                'J0_3',
                # 'J45_3',
                'BlurStrength_3']

featuresSph = [
                # 'Age',
                # 'Gender',
                # 'Topo_Sim_K_K1',
                # 'Topo_Sim_K_K2',
                'WF_SPHERE_R_3',
                # 'WF_CYLINDER_R_3',
                # 'WF_AXIS_R_3',
                # 'WF_RT_Fv_Zernike_Photo_di_Z_2_-2',
                # 'WF_RT_Fv_Zernike_Photo_di_Z_4_0',
                # 'WF_RT_Fv_Zernike_Photo_di_Z_3_1',
                # 'WF_RT_Fv_Meso_PupilRadius',
                # 'WF_RT_Fv_Zernike_Meso_di_Z_2_-2',
                # 'WF_RT_Fv_Zernike_Meso_di_Z_4_0',
                # 'WF_RT_Fv_Zernike_Meso_di_Z_3_1',
                'Pachy_MEASURE_Acd',
                # 'Pachy_MEASURE_WhiteToWhite',
                # 'Pachy_MEASURE_KappaAngle',
                # 'Pachy_MEASURE_Thickness',
                # 'Tono_MEASURE_Average',
                # 'Topo_KERATOCONUS_Kpi',
                # 'AnteriorChamberVol',
                # 'AcdToPupilRadius_Ratio',
                # 'PupilRadiusToW2W_Ratio',
                # 'kRatio',
                'J0_3',
                'J45_3',
                'BlurStrength_3']


fMatCyl  = data.loc[validInds,featuresCyl]
fMatSph  = data.loc[validInds,featuresSph]

deltaS = data.loc[data.index[validInds],'SphereDelta']
deltaC = data.loc[data.index[validInds],'CylinderDelta']

# Preallocations
deltaCyl  = np.zeros(len(fMatCyl))
deltaCylC = np.zeros(len(fMatCyl))
deltaCylE = np.zeros(len(fMatCyl))
deltaSph  = np.zeros(len(fMatSph))
deltaSphC = np.zeros(len(fMatSph))
deltaSphE = np.zeros(len(fMatSph))

# for cylinder classifier
bCyl.Fit(fMatCyl.copy(),deltaC,cylDelta,smooth_epdf=True,smoothing_sigma=5,med_filt_size=5)
probDeltaCyl = []
cChoice      = []
ecChoice     = []
for dIdx in range(len(fMatCyl)):
    pCyl, cylDeltaChoice,expChoice = bCyl.Predict(fMatCyl.loc[fMatCyl.index[dIdx]])
    probDeltaCyl.append(pCyl)
    deltaCyl[dIdx] = cylDelta[pCyl.argmax()]

    cChoice.append(cylDeltaChoice)
    aMax = np.bincount(cylDeltaChoice).argmax()
    deltaCylC[dIdx] = cylDelta[int(aMax)]

    ecChoice.append(expChoice)
    eMax = np.bincount(expChoice).argmax()
    deltaCylE[dIdx] = cylDelta[int(eMax)]


bSph.Fit(fMatSph.copy(),deltaS,sphDelta,smooth_epdf=True,smoothing_sigma=5,med_filt_size=5)
sChoice  = []
esChoice = []
for dIdx in range(len(fMatSph)):
    probDeltaSph, sphDeltaChoice,expChoice = bSph.Predict(fMatSph.loc[fMatSph.index[dIdx]])
    deltaSph[dIdx] = sphDelta[probDeltaSph.argmax()]
    # majority vote by distribution peak
    sChoice.append(sphDeltaChoice)
    aMax = np.bincount(sphDeltaChoice).argmax() # majority vote
    deltaSphC[dIdx] = sphDelta[int(aMax)]

    # majority vote by feature expectation value
    esChoice.append(expChoice)
    eMax = np.bincount(expChoice).argmax()
    deltaSphE[dIdx] = sphDelta[int(eMax)]

# Report
print('____Majority vote___')
for dIdx in [0.00,0.25,0.50]:
    print(f"|S|<={dIdx}: {((deltaS-deltaSphC).abs()<=dIdx).sum()/len(validInds):.2f}")
for dIdx in [0.00,0.25,0.50]:
    print(f"|C|<={dIdx}: { ((deltaC-deltaCylC).abs()<=dIdx).sum()/len(validInds):.2f}")

print('____Expectation___')
for dIdx in [0.00,0.25,0.50]:
    print(f"|S|<={dIdx}: {((deltaS-deltaSphE).abs()<=dIdx).sum()/len(validInds):.2f}")
for dIdx in [0.00,0.25,0.50]:
    print(f"|C|<={dIdx}: { ((deltaC-deltaCylE).abs()<=dIdx).sum()/len(validInds):.2f}")

print('____Probability____')
for dIdx in [0.00,0.25,0.50]:
    print(f"|S|<={dIdx}: {((deltaS-deltaSph).abs()<=dIdx).sum()/len(validInds):.2f}")
for dIdx in [0.00,0.25,0.50]:
    print(f"|C|<={dIdx}: { ((deltaC-deltaCyl).abs()<=dIdx).sum()/len(validInds):.2f}")

print('____Refractometer___')
for dIdx in [0.00,0.25,0.50]:
    print(f"|S|<={dIdx}: {((deltaS).abs()<=dIdx).sum()/len(validInds):.2f}")
for dIdx in [0.00,0.25,0.50]:
    print(f"|C|<={dIdx}: { ((deltaC).abs()<=dIdx).sum()/len(validInds):.2f}")
for dIdx in [0.00,0.25,0.50]:
    print(f"|S|&|C|<={dIdx}: { (((deltaS).abs()<=dIdx)&((deltaC).abs()<=dIdx)).sum()/len(validInds):.2f}")
