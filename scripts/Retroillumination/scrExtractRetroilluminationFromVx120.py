"""
    Extract all retroillumination images from a vx120/130 results folder
    Result folder should include the zip files containing clientdb folder from the vx120

"""

import os
import shutil
import numpy as np
import pandas as pd
import cv2
import zipfile
import re
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

dataFolder        = os.path.join(os.path.dirname(__file__),'..','..','..','..','Data','VisionixVX130CIV','zipFiles')
resultFolder      = os.path.join(dataFolder,'..','Parsed')
tempExtractFolder = os.path.join(resultFolder,'temp')
retro_folder      = os.path.join(resultFolder,'Retro')
wf_folder         = os.path.join(resultFolder,'WF')
pachy_folder      = os.path.join(resultFolder,'Pachy')
topo_folder       = os.path.join(resultFolder,'Topo')

if not os.path.exists(dataFolder):
    raise IsADirectoryError(f'the path {dataFolder} could not be found')

if not os.path.isdir(resultFolder):
    os.mkdir(resultFolder)

if not os.path.exists(tempExtractFolder):
    # create it
    os.mkdir(tempExtractFolder)
else:
    # make sure we get fresh new directory
    shutil.rmtree(tempExtractFolder)
    os.mkdir(tempExtractFolder)

# Make sure the Result directory includes the needed subdirectories
if not os.path.isdir(retro_folder):
    os.mkdir(retro_folder)
if not os.path.isdir(wf_folder):
    os.mkdir(wf_folder)
if not os.path.isdir(topo_folder):
    os.mkdir(topo_folder)
if not os.path.isdir(pachy_folder):
    os.mkdir(pachy_folder)

#____ Parameters _____
retro_image_shape = (512,540)
wf_image_shape    = (512,540)
topo_image_shape  = (512,540)
pachy_image_shape = (512,540)
image_shape       = {'Retro':(512,540),'WF':(512,540),'Pachy':(512,540),'Topo':(512,540)}

pat_list      = os.listdir(dataFolder)
im_format_out = '.png'  # format of images to export
invalid_set   = [-1000,np.nan, None] # representing invalid values
num_rings     = 24  # number of rings in the topo maps
num_phy       = 256 # number of angles
x             = np.zeros((num_phy,num_rings))
y             = np.zeros((num_phy,num_rings))
p_num         = 0   # running counter
# angles        = np.linspace(0,2*np.pi,num_phy) # angles for the topo maps
export_images = False
for pIdx in pat_list[:10]:
    # check if the file is a zip file
    if zipfile.is_zipfile(os.path.join(dataFolder,pIdx)):
        # extract the content into a temporary folder
        print(f'\n({p_num}/{len(pat_list)}) [Info] opening directory {os.path.join(dataFolder,pIdx)}')
        try:
            zipfile.ZipFile(os.path.join(dataFolder,pIdx)).extractall(tempExtractFolder)

            # Fetch the different examination
            f = os.listdir(os.path.join(tempExtractFolder,'clientdb'))
            for fIdx in f:
                k = os.listdir(os.path.join(tempExtractFolder,'clientdb',fIdx))
                # for each examination of the patient
                for kIdx in k:
                    # Extract Retroillumination images

                    for mIdx in ['Retro','Topo','WF','Pachy']:
                        if mIdx=='Retro':
                            eyes = ['Left','right']
                        else:
                            eyes = ['Left','Right']

                        for eIdx in eyes:
                            raw_file_names  = {'Retro':f'Retro_{eIdx}_1.raw',
                                                'WF':f'{eIdx}_Mesopic_1.raw',
                                                'Topo': f'{eIdx}EyeImage_1.raw',
                                                'Pachy':f'Visu_{eIdx}_1.raw'}
                            raw_file_path = os.path.join(tempExtractFolder,'clientdb',fIdx,kIdx,mIdx,raw_file_names[mIdx])
                            if os.path.exists(raw_file_path):
                                print(f'[Info] {mIdx} image found eye {eIdx}')
                                im  = np.fromfile(raw_file_path,dtype=np.uint8)
                                imr = im.reshape(image_shape[mIdx])
                                cv2.imwrite(os.path.join(resultFolder,mIdx,pIdx+f'_{eIdx}').replace('.zip','')+im_format_out,imr)
                            else:
                                print(f'[Warn][RetroImageParser] file {raw_file_path} does not exist')

        except:
            print(f'[Error] Cannot process file: {os.path.join(dataFolder,pIdx)}')
        # empty the temp folder
        shutil.rmtree(os.path.join(tempExtractFolder,'clientdb'))
        p_num+=1
