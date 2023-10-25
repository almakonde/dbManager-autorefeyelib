import sys
import os

# remove previous modules
for kIdx in list(sys.modules.keys()):
    if kIdx.find('autorefeyelib')!=-1:
        print(f'Deleting {kIdx}')
        del sys.modules[kIdx]

from autorefeyelib.Retroillumination import Retro

data_folder = os.path.join(os.path.dirname(__file__),'..','..','..','..','Data','VisionixVX130CIV','Parsed','Retro','samples')
file_list   = os.listdir(data_folder)
r           = Retro.ImageAnalyzer()
for fIdx in file_list:
    if fIdx.endswith('png'):
        r.Start(os.path.join(data_folder,fIdx))