import sys
for kIdx in list(sys.modules.keys()):
    if kIdx.find('autorefeyelib')!=-1:
        print(f'delteting {kIdx}')
        del sys.modules[kIdx]
from autorefeyelib.Parsers import Oplus
emrDataFile = '/Users/ofirshukron/Work/Eyelib/Data/EMR/CIV/Raw/stat_2021_07_06.csv'
o = Oplus.Parser()
o.Load(emrDataFile)
pdb = o.Parse()