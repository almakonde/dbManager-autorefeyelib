#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Create a DB containing pre and post refractive surgery data from  CIV EMR

"""
import sys
import os
import numpy as np
import pandas as pd
import dateutil
import datetime
# sys.path.append(os.path.join(os.getcwd(),'..','..','Parsers','Code'))  # add parser folder to path

# import OplusEMRParser
# o = OplusEMRParser.Parser()
# o.Load(emrFileName='../../../Data/EMR/CIV/OplusCIV_parsed_2020_08_25.csv')
# o.RenameColumns()
print("Loading data")
data = pd.read_csv('../../../Data/EMR/CIV/OplusCIV_parsed_2020_08_25.csv',low_memory=False)
#%%
# for each index in left, search for pre and post operative operations
# remove all lines where the Exam date is not recorded
validEdate   = data['ExamDate'].isna()==False
validBdate   = data['BirthDate'].isna()==False
validName    = data['FirstName'].isna()==False
validSurname = data['Surname'].isna()==False
# data         = data.loc[data.index[validEdate&validBdate]]
# data         = data.loc[data.index[validBdate]]
data         = data.loc[data.index[validName&validSurname&validEdate&validBdate]]

# find all lines with either left or right eye IOL implant procedure
indsLeft  = np.where(data['IolPower_Left'].isna()==False)[0]  # inidicator of surgery
indsRight = np.where(data['IolPower_Right'].isna()==False)[0]

# parse examdate
examDate  = np.ndarray(shape = len(data), dtype=datetime.datetime)
patID     = np.ndarray(shape= len(data),dtype = list)
for eIdx in range(len(data)):
    # record the examination dat
    examDate[eIdx] = dateutil.parser.parse(data.loc[data.index[eIdx],'ExamDate'])
    patID[eIdx]    = str(data.loc[data.index[eIdx],'FirstName'].replace(' ','').lower()+data.loc[data.index[eIdx],'Surname'].replace(' ','').lower()+data.loc[data.index[eIdx],'BirthDate'])

#%% find patients
preop  = pd.DataFrame() # pre operation
postop = pd.DataFrame() # post operation
opday  = pd.DataFrame() # day of operation
temp   = pd.DataFrame(data= np.nan*np.ones(shape=(1,len(data.keys()))),columns = data.keys())
# we treat the preop as all examination up to and including the day of operation
for il in indsLeft:
    # find all pre op examination for this particular patient
    opday         = opday.append(data.loc[data.index[il]])
    patInds       = np.where(patID == patID[il])[0]
    print("Patient {}: found {} matches".format(patID[il], len(patInds)))
    exams         = examDate[patInds] # all the examination of the current patient
    preExamsInds  = patInds[exams<examDate[il]]
    postExamsInds = patInds[exams>examDate[il]]
    # find the last exam before current and treat it as pre op
    if len(preExamsInds)>0:
        preopInd = preExamsInds[np.argsort(exams[exams<examDate[il]])[-1]]
        preop    = preop.append(data.loc[data.index[preopInd]])
    else:
        preop     = preop.append(temp)
    if len(postExamsInds)>0:
        postopInd = postExamsInds[np.argsort(exams[exams>examDate[il]])[0]]
        postop    = postop.append(data.loc[data.index[postopInd]])
    else:
        postop  = postop.append(temp)

for ir in indsRight:
# find all pre op examination for this particular patient
    opday         = opday.append(data.loc[data.index[ir]])
    patInds       = np.where(patID == patID[ir])[0]
    print("Patient {}: found {} matches".format(patID[ir], len(patInds)))
    exams         = examDate[patInds] # all the examination of the current patient
    preExamsInds  = patInds[exams<examDate[ir]]
    postExamsInds = patInds[exams>examDate[ir]]
    # find the last exam before current and treat it as pre op
    if len(preExamsInds)>0:
        preopInd = preExamsInds[np.argsort(exams[exams<examDate[ir]])[-1]]
        preop    = preop.append(data.loc[data.index[preopInd]])
    else:
        preop   = preop.append(temp)
    if len(postExamsInds)>0:
        postopInd = postExamsInds[np.argsort(exams[exams>examDate[ir]])[0]]
        postop    = postop.append(data.loc[data.index[postopInd]])
    else:
        postop  = postop.append(temp)

preop.to_csv("PreOp.csv",index=False)
postop.to_csv("PostOp.csv",index=False)
opday.to_csv("OpDay.csv",index=False)
