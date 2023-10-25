#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:07:08 2020

@author: ofir
 Match between the EMR data and the vx120 data from the Prevot clinic (Province)
"""

import os
import re
import pandas as pd
import numpy as np
import dateutil
import datetime

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

# Load data
dataFolder = os.path.join(os.path.dirname(__file__),'..','..','..','..','Data')
print('[Info][Merger] Loading VX 130 data')
vx  = pd.read_csv(os.path.join(dataFolder,'vx120','Prevot','Provins','Parsed','vx130_Prevot_2022_02_03_Parsed.csv'),low_memory=False)
print('[Info][Merger] Loading EMR data')
emr = pd.read_csv(os.path.join(dataFolder,'EMR','Prevot','Provins','Parsed','Provins_Parsed_2022_02_04.csv'),low_memory=False)

# Parse data
# keys    = ['FirstName','Surname','ExamDate','BirthDate'] # keys which make up the ID

dateDict = {'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06','Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12',' ': '/'}
# dict to translate special characters to english characters
langDict = {'è':'e','é':'e','â':'a','ê':'e','î':'i','ô':'o','û':'u','ç':'c','à':'a','ì':'i','ò':'o','ù':'u','ë':'e','ï':'i','ü':'u'}
idDict   = {' ':'','-':'',"'":'','_':'','>':'','<':'','(':'',')':'','#':''} # replace characters related to ID creation

# Remove empty rows
emrData  = emr.loc[emr[['FirstName','Surname','ExamDate','BirthDate']].notna().all(axis=1),:]

# make emr ID
emrID  = emrData[['FirstName','Surname','ExamDate']].sum(axis=1).str.lower().apply(_DictReplacePD,args=[langDict|idDict])

vxData = vx.loc[vx[['Firstname','Surname','CurrentDate','BirthDate']].notna().all(axis=1),:]
vxData['CurrentDate'] = vxData['CurrentDate'].apply(_DictReplacePD,args=[dateDict]).apply(_ParseDate,dayfirst=True)
# make vx ID 1
vxID1 = vxData[['Firstname','Surname','CurrentDate']].sum(axis=1).str.lower().apply(_DictReplacePD,args=[langDict|idDict])

# make vx ID 2
eDate = vxData['CurrentDate'].values
bDate = vxData['BirthDate'].values
for vIdx in range(len(eDate)):
    m = re.findall(r'\d+',eDate[vIdx])
    if len(m)==3:
        # switch day and month
        eDate[vIdx] = m[1]+'/'+m[0]+'/'+m[2]

vxID2 = (vxData[['Firstname','Surname']].sum(axis=1)+eDate).str.lower().apply(_DictReplacePD,args=[langDict|idDict])

# for vIdx in vxID1.index:
#     vxID1[vIdx] = re.sub('[^a-zA-Z0-9]+','',vxID1[vIdx].lower())
# for vIdx in vxID2.index:
#     vxID2[vIdx] = re.sub('[^a-zA-Z0-9]+','',vxID2[vIdx].lower())
# for eIdx in emrID.index:
#     emrID[eIdx] = re.sub('[^a-zA-Z0-9]+','',emrID[eIdx].lower())

# Match
indexVX  = []
indexEMR = []

for vIdx in range(len(vxData)):
    perc = '#'*round((vIdx/len(vxData))*100/5)*5+'->'
    print(f'\r[Info][Merger] {perc} ({100*vIdx/len(vxData):.2f}\%) found {len(indexVX)} matches', end='\r')

    eInds1 = emrData.index[emrID==vxID1.iloc[vIdx]]
    eInds2 = emrData.index[emrID==vxID2.iloc[vIdx]]

    eInds  = np.union1d(eInds1,eInds2)
    # print(f'[Match] {vIdx+1}/{len(vxData)} found {len(eInds)} matches')
    for eIdx in eInds:
        indexEMR.append(eIdx)
        indexVX.append(vxData.index[vIdx])

vxDataNew        = vxData.loc[indexVX,:]
emrDataNew       = emrData.loc[indexEMR,:]
emrDataNew.index = np.arange(0,len(emrDataNew),1)
vxDataNew.index  = np.arange(0,len(vxDataNew),1)
jointDB          = vxDataNew.copy()
# indicate columns names coming from the EMR
for cIdx in emrDataNew.columns:
    jointDB.loc[:,f'EMR:{cIdx}'] = emrDataNew[cIdx]

