#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import re
import pandas as pd
import numpy as np
import dateutil
import datetime
#import ParserUtils
for kIdx in list(sys.modules.keys()):
    if kIdx.find('autorefeyelib')!=-1:
        print(f'Delteting {kIdx}')
        del sys.modules[kIdx]

def _DictReplacePD(st:str,d:dict)->str:
    """ Replace a string st with any of the values in dictionary d """
    if isinstance(st,str):
        for k in d.keys():
            st = st.replace(k,d[k])
        return st
    else:
        return st

def _ParseDate(st,dayfirst=True):
    """ Service function to parse date into uniform format """
    try:
        st_p     = dateutil.parser.parse(st,dayfirst=dayfirst)
        cur_year = datetime.datetime.now().year
        if st_p.year>cur_year:
            st_p = st_p.replace(year = st_p.year-100)
        d = f'{st_p.day:02d}/{st_p.month:02d}/{st_p.year}'
    except:
        d = None
    return d

# to translate date strings to the format needed to create ID
dateDict = {'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06','Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12',' ': '/'}
# dict to translate special characters to english characters
langDict = {'è':'e','é':'e','â':'a','ê':'e','î':'i','ô':'o','û':'u','ç':'c','à':'a','ì':'i','ò':'o','ù':'u','ë':'e','ï':'i','ü':'u'}
idDict   = {' ':'','-':'',"'":'','_':'','>':'','<':'','(':'',')':'','#':''} # replace characters related to ID creation
#___Parameters____
oplusFile  = os.path.join(os.path.dirname(__file__),'..','..','..','..','Data','EMR','CIV','Parsed','OplusCIV_parsed_2022_03_22.csv')
vxFile     = os.path.join(os.path.dirname(__file__),'..','..','..','..','Data','VisionixVX130CIV','Parsed','vxData_CIV_Parsed_2022_03_18.csv')
max_followup_days = 0 # maximal days between vx measuremtns and subjective refraction (days). set 0 to get the same day

# Load data
print('[Info[[Merger] Loading VX data')
vx         = pd.read_csv(vxFile,low_memory=False)
print('[Info][Merger] Loading EMR data')
# Load parsed EMR data
emr  = pd.read_csv(oplusFile,encoding="utf8",low_memory=False,skip_blank_lines=True,delimiter=",")
keys = ['FirstName','Surname','ExamDate','BirthDate'] # keys which make up the ID

# Prepare ID
eInds    = np.where(np.sum(pd.isnull(emr.loc[:,keys]),axis=1)==0)[0]
eIndices = emr.index[eInds]
# truncate
emrData  = emr.loc[eIndices]
# make ID
emrID  = emrData[['FirstName','Surname','BirthDate']].sum(axis=1)
emrID  = emrID.apply(_DictReplacePD,args=[langDict|idDict]).to_numpy()
vInds  = np.where(vx.loc[:,['Firstname','Surname','BirthDate']].isna().sum(axis=1)==0)[0]
vxData = vx.loc[vx.index[vInds]]
# Translate the vx Current data to strings
vxData['CurrentDate'] = vxData['CurrentDate'].apply(_DictReplacePD,args=[dateDict]).apply(_ParseDate)
vxData['BirthDate']   = vxData['BirthDate'].apply(_ParseDate)

# make ID1
vxID1 = vxData[['Firstname','Surname','BirthDate']].sum(axis=1).str.lower().apply(_DictReplacePD,args=[langDict|idDict]).to_numpy()
# make ID 2 by reversing day and month in exam date
eDate = vxData['BirthDate'].values
for vIdx in range(len(eDate)):
    m = re.findall('\d+',eDate[vIdx])
    if len(m)==3:
        # switch day and month
        eDate[vIdx] = m[1]+'/'+m[0]+'/'+m[2]
vxID2 = (vxData['Firstname']+vxData['Surname']+eDate).str.lower().apply(_DictReplacePD,args=[langDict|idDict]).to_numpy()

# Match
match    = []
no_match = []
no_visual_acuity = []
jointDB  = pd.DataFrame()
jIdx     = 0
for vIdx in range(len(vxData)):
    # print progress bar
    perc = '#'*round((vIdx/len(vxData))*100/5)*5+'->'
    print(f'\r[Info][Merger] {perc} ({100*vIdx/len(vxData):.2f}\%) found {len(jointDB)} matches', end='\r')

    eInds1 = np.where(emrID==vxID1[vIdx])[0]
    eInds2 = np.where(emrID==vxID2[vIdx])[0]

    if len(eInds1)>0:
        eInds = eInds1
        match.append(vxID1[vIdx])
    elif len(eInds2)>0:
        eInds = eInds2
        no_match.append(vxID2[vIdx])
    else:
        no_match.append(vxID1[vIdx])
        eInds = []

    # take the examination closest to the date of vx120
    sub_emr = emrData.iloc[eInds,:]
    if sub_emr.shape[0]>0:
        # Rank entries according to the availability of subjective refraction closest to the vx120 examination date
        date_diff = np.ndarray(len(sub_emr))
        vx_date   = dateutil.parser.parse(vxData.loc[vxData.index[vIdx]]['CurrentDate'])
        for edIdx in range(len(sub_emr)):
            emr_date         = dateutil.parser.parse(sub_emr.iloc[edIdx]['ExamDate'])
            date_diff[edIdx] = (emr_date-vx_date).days
        # Find those measurements for which the diffrence between the record in the EMR
        # and the examination of the vx is no greater than a threshold
        inds = np.where((date_diff>=0)&(date_diff<=max_followup_days))[0]
        vals = sub_emr.iloc[inds][['VisualAcuitySphere_Left','VisualAcuitySphere_Right']].isna().sum(axis=1)
        if len(vals)>0:
            inds = np.argmin(vals)
            jointDB.loc[jIdx,vxData.keys()] = vxData.iloc[vIdx]
            for kIdx in emrData.keys():
                    jointDB.loc[jIdx,'EMR:'+kIdx] = sub_emr.iloc[inds][kIdx]
            jIdx+=1
        else:
            no_visual_acuity.append(emrID[eInds[0]])



