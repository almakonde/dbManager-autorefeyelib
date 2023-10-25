# scr generate iolEMRjointDB
import sys
import os
import pandas as pd
for kIdx in list(sys.modules.keys()):
    if kIdx.find('autorefeyelib')!=-1:
        print(f'deleting {kIdx}')
        del sys.modules[kIdx]

from autorefeyelib.IOLpower import iolDB

iolPath   = os.path.join(os.path.dirname(__file__),'..','..','..','..','Data','IOLmaster','IOLMaster_2021_07_06.csv')
oplusPath = os.path.join(os.path.dirname(__file__),'..','..','..','..','Data','EMR','CIV','Parsed','EMR_CIV_Parsed_2021_07_07.csv')
oplusData = pd.read_csv(oplusPath,low_memory=False)
iolData   = pd.read_csv(iolPath,low_memory=False,delimiter = ';')
idb       = iolDB.DataBase()
jdb       = idb.Merge(iolData,oplusData,
                      max_measurement_to_op_days=120,
                      min_followup_days=10,
                      max_followup_days=90)
