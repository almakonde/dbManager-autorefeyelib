"""
 a script to run classification of IOL
"""

import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RANSACRegressor

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC

plt.rcParams.update({'font.size': 22})
from sklearn.metrics import make_scorer
import pickle
for kIdx in list(sys.modules.keys()):
    if kIdx.find('autorefeyelib')!=-1:
        print(f'delteting {kIdx}')
        del sys.modules[kIdx]
from autorefeyelib.IOLpower import Predictor as iolPredictor
from autorefeyelib.IOLpower import Classifier as iolClassifier
from autorefeyelib.IOLpower import iol_params as params

def deltaP(n_c,elp,meanK,Rt,dR):
    """
     Compute the change in power based on the change of refraction
     Parameters:
     ----------
     n_c, float,
      refraction index of cornea
     elp, float
      effective lens position (mm)
     meanK, float
      average keratometry (D)
     Rt, float
      target refraction (D)
     dR, float
      refraction delta

     Returns
     --------
     dP, float
      power delta (D)
    """
    alpha = n_c-(elp/1000)*(meanK+Rt)
    return -(n_c**2)*dR/(alpha*(alpha-(elp/1000)*dR))

def ELP(n_c,k,Rt,dP,dR):
    A = dP*(k+Rt)*(k+Rt+dR)
    B = -dP*n_c*(2*(k+Rt)+dR)
    C = n_c**2 *(dR+dP)
    elp = (-B +(B**2 -4*A*C)**0.5)/(2*A)
    return elp

def KNNDistFunc(dists,dr=None):
    # for each one of the K neightbors, for each point, assign a weight
    # give weights according to distances from 0.25
    n_neighbors = dists.shape[1]
    w      = [0, 0.25, 0.5, 0.75, 1]
    ws      = [5/5, 4/5, 3/5, 2/5, 1/5]
    weights = (4*dists).round()/4
    for wIdx in range(len(w)):
        weights[weights==w[wIdx]] = ws[wIdx]
    weights[weights>1] = 0
    return weights

def _RegressionLoss(gt_vals,pred_vals,rDelta=0.25, residual_threshold=1.5):
    """
     Loss function for RANSACregression methodologies (to minimize)

     Parameters:
     -----------
     gt_vals, array float,
      1d array of ground truth values
     pred_vals, array float
      1d array same length as gt_vals, with predicted values
     rDelta, float, default=0.25
      dioptric interval for refraction
     residual_threshold, float, default = 1.5
      threshold above which the loss will be considered an outlier in the RANSAC fitting procedure

     Returns:
     ---------
      loss, array float
       loss for each observation based on error intervals of 0, 0.25, 0.5, 0.75,...
    """
    # compute loss
    loss     = np.ones(len(gt_vals))*residual_threshold
    abs_diff = abs(gt_vals-(pred_vals/rDelta).round()*rDelta)

    # loss[abs_diff==0]    = 0
    loss[abs_diff<=0.25] = 0
    loss[abs_diff==0.5]  = 0.5*residual_threshold
    loss[abs_diff>0.5]   = 2*residual_threshold
    # print(loss)
    return loss

def _Round(vals,delta):
    return np.round(vals/delta)*delta

# Initialization
e = iolPredictor.Predictor()
c = iolClassifier.Classifier()

# Prepare a combined right/left eye database parameters
data         = e.data.copy()

data['IolDesiredRefractionPostOp_Left']  = _Round(data['IolDesiredRefractionPostOp_Left'], params.rDelta)
data['IolDesiredRefractionPostOp_Right'] = _Round(data['IolDesiredRefractionPostOp_Right'],params.rDelta)
C                     = pd.DataFrame()
r1                    = data['l_radius_r1'].append(data['r_radius_r1']) # cornea radius
r2                    = data['l_radius_r2'].append(data['r_radius_r2']) # cornea radius
Rc                    = 0.5*(r1+r2)
k1                    = 1000*(1.336-1)/r1 #TODO: pass the r and the specific n_c, n_v to the formulas to compute their own mean K
k2                    = 1000*(1.336-1)/r2
C['meanK']            = 0.5*(k1+k2)# mean keratometry
C['meanR']            = Rc
C['axialLength']      = data['l_axial_length_mean'].append(data['r_axial_length_mean'])
C['ACD']              = data['l_acd_mean'].append(data['r_acd_mean'])
C['targetRefraction'] = data['IolDesiredRefractionPostOp_Left'].append(data["IolDesiredRefractionPostOp_Right"])
C.loc[C['targetRefraction'].isna(),'targetRefraction'] = 0.0 # assume missing value represent emmetropia
C['finalRefraction']  = data['VisualAcuitySphere_Left'].append(data['VisualAcuitySphere_Right'])
C['age']              = data['Age'].append(data['Age'])
C['WTW']              = data['l_wtw_mean'].append(data['r_wtw_mean'])
C['Pi']               = data['IolPowerImplanted_Left'].append(data['IolPowerImplanted_Right']) # implanted IOL power
C['ExamDate']         = data['ExamDate'].append(data['ExamDate'])
C['deltaR']           = C['finalRefraction']-C['targetRefraction']
C['Followup_Days']    = data['Followup_Days']

# Filter DB by valid ranges
inds = np.ones(len(C),dtype=bool)
for kIdx in params.valid_ranges.keys():
    inds = inds & (C[kIdx]<=max(params.valid_ranges[kIdx])).values & (C[kIdx]>=min(params.valid_ranges[kIdx])).values
    # C.loc[inds,kIdx]= None
C = C.iloc[inds]
C.dropna(inplace=True)
data    = data.loc[C.index]
C.index = range(len(C))
# define feature matrix
featureMat = C[params.features]
# copute extea features
featureMat.iloc[:]['acd_al_ratio'] = featureMat['ACD']/featureMat['axialLength']
# TODO: get specific A constant for each IOL before training

# Classification, preallocation
score        = np.zeros((len(params.alpha,)))
Tind         = round(params.dev.frac_training*len(featureMat))
nValidate    = len(featureMat)-Tind

shuffledInds = np.random.permutation(len(featureMat))
trainInds    = shuffledInds[:Tind]
validateInds = shuffledInds[Tind:]
# Train the model
trainingFeatMat = featureMat.iloc[trainInds]
#
# Get Classes, power predicted, refraction predicted, ELP, and AL for each formula
labels, P, R, elps, al = c.GetClasses(params.formulas,
                                    params.Aconst,
                                    C['meanK'],
                                    C['ACD'],
                                    C['WTW'],
                                    C['axialLength'],
                                    C['targetRefraction'],
                                    C['Pi'],
                                    C['finalRefraction'],
                                    e._averages['CornealHeight'],
                                    e._averages['ACD'],
                                    e._constants['HolladaySurgeonFactor'],
                                    e._constants['HofferQPersonalizedACD'],
                                    pDelta=None,
                                    rDelta=None)

# Get training labels
trainingLabels  = labels[trainInds]

refErr   = (C['finalRefraction']-C['targetRefraction']) # ground truth delta R
fMat_reg = {} # feature matrices
rr       = {} # regressors
dR       = pd.DataFrame(columns = params.formulas)
dP       = pd.DataFrame(columns = params.formulas)
reg_err  = pd.DataFrame(columns = params.formulas)

for fIdx in params.formulas:
    featureMat.iloc[:]['P']      = P[fIdx].values # add the power predicted
    fTemp                        = featureMat[params.features_reg].copy()
    fTemp.iloc[:]['ACD']         = elps[fIdx].to_numpy()
    fTemp.iloc[:]['axialLength'] = al[fIdx].to_numpy()
    fMat_reg[fIdx]               = fTemp
    # construct regressors for the current IOL formula
    rr[fIdx]   = RANSACRegressor(base_estimator = LinearRegression(fit_intercept=True),
                                                                    min_samples=0.33,
                                                                    max_trials=300,
                                                                    residual_threshold=0.75,
                                                                    loss=e.RegressionLoss)
    rr[fIdx].fit(fMat_reg[fIdx].iloc[trainInds],refErr.iloc[trainInds])
    # predict the refraction delta for the whole data set (trainnig +validation)
    dR[fIdx]      = rr[fIdx].predict(fMat_reg[fIdx]) # predict deviation from target refraction, i.e. final refraction
    dP[fIdx]      = deltaP(params.n_c[fIdx],elps[fIdx],featureMat['meanK'],featureMat['targetRefraction'],dR[fIdx])
    reg_err[fIdx] = (dR[fIdx]-refErr).abs() # (NOTE:Do not round, otherwise class assignment won't work properly due to argmin)

def CVScoring(X,Y,reg_err_cv=reg_err):
    """
        Score function for cross-validation for iol formula classifier
        Parameters:
        -----------
        reg_err, the rgression errors for each class
    """
    # score the model by the regression errors at 0 and 0.25 diopters
    score = 0
    for xIdx in range(len(X)):
        # print(X)
        if (np.abs(reg_err_cv.loc[X.index[xIdx],reg_err_cv.keys()[X.iloc[xIdx]]])<=0.25):
            score+=1
        elif (np.abs(reg_err_cv.loc[X.index[xIdx],reg_err_cv.keys()[X.iloc[xIdx]]])<=0.5):
            score+=0.5
        elif (np.abs(reg_err_cv.loc[X.index[xIdx],reg_err_cv.keys()[X.iloc[xIdx]]])<=0.75):
            score-=0.5
        elif (np.abs(reg_err_cv.loc[X.index[xIdx],reg_err_cv.keys()[X.iloc[xIdx]]])>0.75):
            score-=1
    return score

# For each observation in the training set, assign a label according to the formula number producing the minimal prediction error
# Create a label array, in which multiple labels might be possible
# When chacking the accuracy of the model, check if the predicted label is a member of the list of possible classe fro each observation
listMinClass = []
for lIdx in range(len(reg_err)):
    reTemp = _Round(reg_err.iloc[lIdx],params.rDelta)

    listMinClass.append(np.where(reTemp==min(reTemp))[0])
# labels from regression (the regressor's number)
reg_labels = reg_err.apply(np.argmin,axis=1) # ground truth

# Train a classifier that predicts the IOL formula according which gives the smallest refraction error
if params.dev.tune_hyperparameters:
    # tune hyper-parameters
    reg_chooser = GridSearchCV(estimator=KNeighborsClassifier(),
                                param_grid={'n_neighbors':[5,10,20,50],
                                            'weights':['distance','uniform'],
                                            'p':[1,2],
                                            'algorithm':['kd_tree','ball_tree'],
                                            'leaf_size':[5, 10,20,30,40]},
                                verbose=2,
                                refit=True,
                                scoring=make_scorer(CVScoring,greater_is_better=True))
else:
    # Run with default params
    # reg_chooser     = KNeighborsClassifier(algorithm=params.dev.reg_chooser_params['algorithm'],
    #                                        n_neighbors=params.dev.reg_chooser_params['n_neighbors'],
    #                                        p=params.dev.reg_chooser_params['p'],
    #                                        leaf_size=params.dev.reg_chooser_params['leaf_size'])
    # reg_chooser  = LogisticRegressionCV(max_iter=1000,
    #                 scoring=make_scorer(CVScoring,greater_is_better=True),
    #                 verbose=2)
    reg_chooser = SVC(break_ties=True,)
# Train and predict which regressor to use

reg_chooser.fit(featureMat[params.features_class].iloc[trainInds],reg_labels.iloc[trainInds])
pred_reg_labels = reg_chooser.predict(featureMat[params.features_class].iloc[validateInds]).astype(int)

# For logistic regression
# reg_chooser_l.fit(featureMat[params.features_class].iloc[trainInds],reg_labels.iloc[trainInds])
# pred_reg_labels = reg_chooser_l.predict(featureMat[params.features_class].iloc[validateInds]).astype(int)

#__ Summarize findings___
pred_formula    = np.asanyarray(params.formulas)[pred_reg_labels]
reg_class_score = 0
nInds           = len(validateInds)
pred_dr         = np.zeros(nInds)
pred_r          = np.zeros(nInds)
pred_p          = np.zeros(nInds)
pred_dp         = np.zeros(nInds)
pred_err        = np.zeros(nInds)
r_from_p        = np.zeros(nInds)
for lIdx in range(nInds):
    reg_class_score+=(pred_reg_labels[lIdx] in listMinClass[validateInds[lIdx]])
    pred_r[lIdx]   = R.iloc[validateInds[lIdx],pred_reg_labels[lIdx]]   # The refraction based on the formula chosen
    pred_dr[lIdx]  = dR.iloc[validateInds[lIdx],pred_reg_labels[lIdx]]  # Predicted difference in refraction
    pred_p[lIdx]   = P.iloc[validateInds[lIdx],pred_reg_labels[lIdx]]   # IOL power based on formula chosen
    pred_dp[lIdx]  = dP.iloc[validateInds[lIdx],pred_reg_labels[lIdx]]  # differenc in IOL power to arraive at R+dR
    pred_err[lIdx] = _Round(np.abs(pred_r[lIdx]+pred_dr[lIdx] - C['finalRefraction'].iloc[validateInds[lIdx]]),params.rDelta)
    r_from_p[lIdx] = e.formulas.Refraction(pred_formula[lIdx],pred_p[lIdx]+pred_dp[lIdx],C['meanK'].iloc[validateInds[lIdx]],elps[pred_formula[lIdx]].iloc[validateInds[lIdx]],al[pred_formula[lIdx]].iloc[validateInds[lIdx]],n_c=params.n_c[pred_formula[lIdx]],n_v=params.n_v[pred_formula[lIdx]])
reg_class_score /=nInds


deltas        = [0,0.25,0.5,0.75,1,1.25,1.5,1.75]
reg_cum_err   = pd.DataFrame(columns=params.formulas)
train_score   = pd.DataFrame(columns = params.formulas)
final_cum_err = pd.DataFrame(index=deltas,columns=['Absolute Err'])
per_formula_cum_err_before = pd.DataFrame(index = deltas, columns = params.formulas) # before correction by regression
per_formula_cum_err_after  = pd.DataFrame(index = deltas, columns = params.formulas) # after correction by regression
for dIdx in deltas:
    final_cum_err.loc[dIdx] = (pred_err<=dIdx).sum()/len(pred_err)
    for fIdx in params.formulas:
        train_score.loc[dIdx,fIdx] = (_Round(reg_err[fIdx].iloc[trainInds],params.rDelta) <=dIdx).mean()
        reg_cum_err.loc[dIdx,fIdx] = (_Round(reg_err[fIdx].iloc[validateInds],params.rDelta) <=dIdx).mean()
        per_formula_cum_err_before.loc[dIdx,fIdx] = ((R[fIdx].iloc[validateInds]-C['finalRefraction'].iloc[validateInds]).abs()<=dIdx).mean()
        per_formula_cum_err_after.loc[dIdx,fIdx]  = (((R[fIdx]+_Round(dR[fIdx],params.rDelta)).iloc[validateInds]-C['finalRefraction'].iloc[validateInds]).abs()<=dIdx).mean()

# Refraction error conditional on the powert implanted
formulaRefMAE = pd.DataFrame(columns=params.formulas)
for fIdx in params.formulas:
    formulaRefMAE.loc[:,fIdx] = (e.formulas.Refraction(fIdx,C['Pi'],C['meanK'],elps[fIdx],al[fIdx],n_c=params.n_c[fIdx],n_v=params.n_v[fIdx])-C['finalRefraction']).abs()
formulaCumError = pd.DataFrame(index=deltas,columns=params.formulas)
for dIdx in deltas:
    formulaCumError.loc[dIdx,:] = ((_Round(formulaRefMAE,params.rDelta))<=dIdx).sum()/len(formulaRefMAE)

# # Compute errors between predicted and observed
# fRefErr = pd.DataFrame(columns = R.columns)
# pRefErr = pd.DataFrame(columns = P.columns)
# for pIdx in P.columns:
#     pRefErr[pIdx] = P[pIdx]-C['Pi']
#     fRefErr[pIdx] = _Round(R[pIdx],params['rDelta'])-C['finalRefraction'] # dR

# Print results
print(f"Datbase feature statistics: \n {featureMat.describe().drop(['count','25%','50%','75%'])}")
print(f"Mean Power implanted - power predicted <P_i-P_srkt>: {(C['Pi']-P['SRKT']).mean():.2f}, Mean absolute difference: {(C['Pi']-P['SRKT']).abs().mean():.2f}")
print(f"<|TargetRefraction-FinalRefraction|>: {(C['finalRefraction']-C['targetRefraction']).abs().mean():.2f}")
print(f'Cumulative conditional MAE formulas given P_i:\n {formulaCumError}')
print(f'Cumulative absolute error training regressors:\n {train_score}')
print(f'Cumulative absolute error validation regressor:\n {reg_cum_err}')
print(f'Final cumulative absolute error classifier:\n{final_cum_err}')
print(f"Regression classifier accuracy: {reg_class_score:.2f}")
print(f'Predicted MAE after regression and classification (validation set): {pred_err.mean():.2f}')
print(f'Theoretical miminal MAE after regression (validation): {reg_err.iloc[validateInds].apply(min,axis=1).mean():.2f}')
for fIdx in P.keys():
    print(f'Mean delta P {fIdx}:{(C.iloc[validateInds]["Pi"]-P[fIdx]).mean():.2f}')
print(f'Mean delta P (predicted) {pred_dp.mean():.2f}')
print(f'Training size {len(trainInds)}, validation size {len(validateInds)}. total {len(C)}')

    #--------------------

    # clf[aIdx] = c.Fit(params["Aconst"],
    #                 trainingFeatMat,
    #                 trainingLabels,
    #                 numExp=params['dev']['num_exp'],
    #                 alpha=params['alpha'][aIdx])

    # # Predict labels
    # validateFeatMat = featureMat.iloc[validateInds]
    # pLabels         = clf[aIdx].predict(validateFeatMat)
    # score[aIdx]     = clf[aIdx].score(validateFeatMat,labels[validateInds])

    # if np.isnan(score[aIdx]):
    #     # record
    #     badInds = validateInds
    # if score[aIdx]>bestScore:
    #     bestInds       = validateInds
    #     bestScore      = score[aIdx]
    #     bestClassifier = aIdx
    #     bestPlabels    = pLabels

    # # fImportance[aIdx]  = clf[aIdx].feature_importances_
    # for fIdx in range(len(pLabels)):
    #     # compute final refraction error
    #     err[fIdx,aIdx] = fRefErr.iloc[validateInds[fIdx],pLabels[fIdx]]
    #     # compute power difference
    #     pErr[fIdx,aIdx]  = pRefErr.iloc[validateInds[fIdx],pLabels[fIdx]]
    #     Ralgo[fIdx,aIdx] = R.iloc[validateInds[fIdx],pLabels[fIdx]]
    #     Palgo[fIdx,aIdx] = P.iloc[validateInds[fIdx],pLabels[fIdx]]
    # # compare mean predicted refErr with the mean refraction error of each method
    # dMean += fRefErr.iloc[validateInds].abs().mean(axis=0).values

    # # Average final refraction error for the data set for each formula
    # # dMean/=numExp
    # CC        = C.iloc[validateInds]
    # elp       = elps.iloc[validateInds]
    # als       = al.iloc[validateInds]
    # CC.index  = range(len(validateInds))
    # als.index = range(len(validateInds))
    # elp.index = range(len(validateInds))
    # ref_error_model = np.zeros(len(validateInds))
    # for vIdx in range(len(validateInds)):
    #     ref_error_model[vIdx] = abs((e.formulas.PredictedRefraction(Palgo[vIdx], # power to arrive at target refraction
    #                          validateFeatMat.iloc[vIdx]['meanK'],als.iloc[vIdx,pLabels[vIdx]],
    #                          elp.iloc[vIdx,pLabels[vIdx]],
    #                          n_c=params["n_c"][params["formulas"][pLabels[vIdx]]],
    #                          n_v=params["n_v"][params["formulas"][pLabels[vIdx]]])/params['rDelta']).round()*params['rDelta'] -CC.iloc[vIdx]["finalRefraction"])

    # # Compute the error for the used predicted IOL formula
    # pMean  = pd.DataFrame(abs(err)).mean(axis=1).dropna().mean()

    # # compute the cumulative error
    # limF = np.arange(0,1.5,0.25)
    # limP = np.arange(0,8,0.125)
    # pCum = pd.DataFrame(index=limP)
    # fCum = pd.DataFrame(index=limF)
    # for lIdx in limF:
    #     # Refraction cumulative error
    #     fCum.loc[lIdx,fRefErr.columns] = (fRefErr.iloc[bestInds].abs()<=lIdx).sum()/len(bestInds)
    #     # fCum.loc[lIdx,'Predicted']     = (abs(err[:,bestClassifier])<=limF[lIdx]).sum()/len(err[:,bestClassifier])
    #     fCum.loc[lIdx,'Predicted']     = (abs(ref_error_model)<=lIdx).sum()/len(validateInds)
    #     # cumulative power error
    # for lIdx in range(len(limP)):
    #     pCum.loc[lIdx,pRefErr.columns] = (pRefErr.iloc[bestInds].abs()<=limP[lIdx]).sum()/len(bestInds)
    #     pCum.loc[lIdx,'Predicted']     = (abs(pErr[:,bestClassifier])<=limP[lIdx]).sum()/len(pErr[:,bestClassifier])

    # ratio               = pd.DataFrame()
    # ratio['Power']      = pRefErr.abs().mean(axis=0)/abs(pErr[:,bestClassifier]).mean()
    # ratio['Refraction'] = fRefErr.abs().mean(axis=0)/abs(err[:,bestClassifier]).mean()

    # print(f'\nMAE-final refraction  Algo. {pMean:.2f}\n  Mean score:{score[aIdx].mean():.2f}\n')
    # print(f'MAE-final refraction\n {fRefErr.abs().mean()}\n')
    # print(f'Accuracy ratio (err. formula)/(err. prediction) for best score: \n {ratio}')
    # print(f'Prediction score, best {score[aIdx]:.2f}, average:{score[aIdx].mean():.2f}, min:{score[aIdx].min():.2f}')

# ---- Export the best model ---
if params.dev.export_model:
    # export all regressors
    with open('regressors.sav', 'wb') as fileName:
        pickle.dump(rr,fileName)
    # export regressor classifier
    with open('reg_chooser.sav','wb') as fileName:
        pickle.dump(reg_chooser,fileName)



#%% Plots
if params.dev.showResultFigures:
    # parameters
    plt.rcParams.update({'font.size': 22})
    markers = ['+','2','x','v','s','*','^','d','o','-']

    # Plot the cumulative error of formulas
    fig0 = plt.figure()
    ax0  = fig0.add_subplot(111)
    ax0.plot(per_formula_cum_err_before,'s-',linewidth=2)
    ax0.plot(per_formula_cum_err_after.mean(axis=1),'o--',color='k',linewidth=3)
    leg0 = list(per_formula_cum_err_before.keys())
    leg0.append('Predited')
    ax0.legend(leg0)
    ax0.set_xlabel('$\\delta$')
    ax0.set_ylabel('Fraction$<=\\delta$')
    ax0.set_xticks(deltas)
    fig0.show()


    fig1 = plt.figure('Percent improvement cum err')
    ax1  = fig1.add_subplot(111)
    ax1.plot((per_formula_cum_err_after- per_formula_cum_err_before)*100,'s-',linewidth=2)
    ax1.legend(list(per_formula_cum_err_before.keys()))
    ax1.set_xlabel('$\\delta$')
    ax1.set_ylabel('% improvement (Predicted-formulas)')
    ax1.set_xticks(deltas)
    fig1.show()

    # power implanted vs power predicted
    fig2 = plt.figure('Conditional MAE')
    ax2  = fig2.add_subplot(111)
    ax2.plot(formulaCumError.index,formulaCumError,'-o')
    ax2.legend(list(formulaCumError.keys()))
    ax2.set_xlabel('$\\delta')
    ax2.set_ylabel('Fraction$<=\\delta$')
    fig2.show()

    fig3 = plt.figure('Histogram of selected formulas')
    ax3  = fig3.add_subplot(111)
    ax3.hist(pred_formula)
    ax3.set_ylabel('Number of cases')
    fig3.show()

    # Plot absolute refraction error
    # fig0 = plt.figure()
    # ax0  = fig0.add_subplot(111)
    # for fIdx in range(len(fRefErr.keys())):
    #     ax0.plot(fRefErr.iloc[bestInds,fIdx].abs().values,marker=markers[fIdx],markeredgecolor='k')
    # ax0.plot(abs(err[:,score.argmax()]),'-o',color='k',linewidth=3,markeredgecolor='k',markersize=8)
    # ax0.set_xlabel('Observation #')
    # ax0.set_ylabel('Absolute error final refraction (D)')
    # ax0.legend(np.append(fRefErr.columns,'Predicted'))
    # fig0.show()

    # fig1 = plt.figure()
    # ax1  = fig1.add_subplot(111)
    # for pIdx in range(len(P.columns)):
    #     ax1.plot(C['Pi'][bestInds].values,P.iloc[bestInds,pIdx].values,markers[pIdx])
    # ax1.plot(C['Pi'][bestInds].values,Palgo[:,bestClassifier],'o',color='k')
    # ax1.plot([C['Pi'][bestInds].min(),C['Pi'][bestInds].max()],[C['Pi'][bestInds].min(), C['Pi'][bestInds].max()],color='k')
    # ax1.legend(np.append(P.columns,'Predicted'))
    # ax1.set_xlabel('IOL power implanted [D]')
    # ax1.set_ylabel('IOL power predicted [D]')
    # fig1.show()

    # plot predicted refraction vs target refraction
    # fig2 = plt.figure()
    # ax2  = fig2.add_subplot(111)
    # for pIdx in range(len(R.columns)):
    #     ax2.plot(C['targetRefraction'][bestInds],R.iloc[bestInds,pIdx],markers[pIdx],
    #             markeredgecolor='k',
    #             markerfacecolor="none",
    #             markersize=8)
    # ax2.plot(C['targetRefraction'][bestInds],Ralgo[:,bestClassifier],'o',
    #             markeredgecolor='k',markerfacecolor='r',alpha=0.8,markersize=8)
    # ax2.plot([C['targetRefraction'][bestInds].min(),C['targetRefraction'][bestInds].max()],\
    #         [C['targetRefraction'][bestInds].min(),C['targetRefraction'][bestInds].max()],color='k')
    # ax2.set_xlabel('Target refraction [D]')
    # ax2.set_ylabel('Predicted refraction [D]')
    # ax2.legend(np.append(R.columns,'Predicted'))
    # fig2.show()

    # plot final refraction error vs axial length
    # fig3 = plt.figure()
    # ax3  = fig3.add_subplot(111)
    # for pIdx in range(len(fRefErr.keys())):
    #     # ax3.violinplot(fRefErr.iloc[bestInds,pIdx])
    #     ax3.plot(C.loc[C.index[bestInds],'axialLength'],fRefErr.iloc[bestInds,pIdx],markers[pIdx],
    #             markeredgecolor='k',
    #             markerfacecolor="none",
    #             markersize=8)
    # ax3.plot(C.loc[C.index[bestInds],'axialLength'],(err[:,bestClassifier]),'o',
    #         color='r',markersize=8,markeredgecolor='k')
    # ax3.plot([min(C.loc[C.index[bestInds],'axialLength']), max(C.loc[C.index[bestInds],'axialLength'])] ,[0,0],
    #         color='k',linewidth=3)
    # ax3.set_xlabel('Axial length (mm)')
    # ax3.set_ylabel('Refractive error [D]')
    # ax3.legend(np.append(fRefErr.columns,'Predicted'))
    # fig3.show()

    # plot cumulative refraction and power error
    # fig4 = plt.figure()
    # ax4  = fig4.add_subplot(121)
    # ax4.plot(limF,fCum.iloc[:,:-1],'o-',linewidth=3)
    # ax4.plot(limF,fCum.iloc[:,-1],'-s',color='k',linewidth=4,markeredgecolor='r',markersize=7)
    # ax4.set_xlabel('|Refraction error| (D)')
    # ax4.set_ylabel('Cumulative fraction')
    # ax4.legend(fCum.columns)
    # ax5  = fig4.add_subplot(122)
    # ax5.plot(limP,pCum.iloc[:,:-1],'-o',linewidth=3)
    # ax5.plot(limP,pCum.iloc[:,-1],'-s',linewidth=4,markeredgecolor='r',color='k',markersize=7)
    # ax5.set_xlabel('|IOL power error| (D)')
    # fig4.show()

    # # plot histogram of classes for the specific choice of alpha
    # fig5 = plt.figure('histogram of Predicted classes')
    # ax5 = fig5.add_subplot(111)
    # # construct histogram of labels
    # h = np.zeros(len(fRefErr.columns))
    # for i in range(len(fRefErr.columns)):
    #     h[i] = np.sum(labels==i)
    # ax5.bar(np.arange(0,len(fRefErr.columns),1),h)
    # ax5.set_xticks(np.arange(0,len(fRefErr.columns),1))
    # ax5.set_xticklabels(labels=np.asanyarray(fRefErr.columns))
    # ax5.set_ylabel('Class histogram')
    # ax5.set_xlabel('Formula')
    # fig5.show()

    # # Violin plot for the refraction error post op
    # fig6, ax6 = plt.subplots(nrows=1, ncols=len(fRefErr.columns)+1)
    # for aIdx in range(len(fRefErr.keys())):
    #     ax6[aIdx].violinplot(fRefErr.iloc[bestInds,aIdx],showextrema=False,points=len(bestInds),showmedians=True,widths=0.5)
    #     ax6[aIdx].set_title(fRefErr.keys()[aIdx])
    #     ax6[aIdx].set_ylim([-3,3])
    #     ax6[aIdx].set_xlim([-0.5,2])
    # ax6[aIdx+1].violinplot(err[:,bestClassifier],showextrema=False,points=len(bestInds),showmedians=True,widths=0.5)
    # ax6[aIdx+1].set_title('Predicted')
    # ax6[aIdx+1].set_ylim([-0.3,0.4])
    # ax6[aIdx+1].set_xlim([-0.5,2])
    # fig6.show()


    # # fig7, ax7 = plt.subplots(nrows=1, ncols=len(fRefErr.columns)+1)
    # fig7 = plt.figure()
    # ax7  = fig7.add_subplot(111)
    # for aIdx in range(len(fRefErr.keys())):
    #     ax7.hist(fRefErr.iloc[bestInds,aIdx],density=True,bins=40,color=None,histtype='step')
    #     ax7.set_title(fRefErr.keys()[aIdx])
    #     # ax7[aIdx].set_ylim([-0.3,0.4])
    #     # ax7[aIdx].set_xlim([-0.5,2])
    # ax7.hist(err[:,bestClassifier],density=True,bins=40,color=None,histtype='bar')
    # # ax7.set_title('Predicted')
    # # ax7[aIdx+1].set_ylim([-0.3,0.4])
    # # ax7[aIdx+1].set_xlim([-0.5,2])
    # fig7.show()

    # Statistics of the formula used
    formulas_civ = (data['IolFormula_Right'].append(data['IolFormula_Left'])).dropna()
    fNames       = formulas_civ.unique()
    for fIdx in fNames:
        N = (formulas_civ==fIdx).sum()
        print(f'{fIdx}: {N} ({100*N/len(formulas_civ):.2f}%)')