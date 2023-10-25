""" Extract all retroillumination images from a vx120/130 results folder
    Parse all topo maps: Tangential, Axial, Height, and the Radii and export to .csv
"""

import os
import shutil
import numpy as np
import pandas as pd
# import cv2
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
topo_maps         = os.path.join(topo_folder,'Maps')
axial_folder      = os.path.join(topo_maps,'Axial')
tangential_folder = os.path.join(topo_maps,'Tangential')
height_folder     = os.path.join(topo_maps,'Height')
radii_folder      = os.path.join(topo_maps,'Radii')

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
if not os.path.isdir(topo_maps):
    os.mkdir(topo_maps)
    os.mkdir(axial_folder)
    os.mkdir(tangential_folder)
    os.mkdir(height_folder)
    os.mkdir(radii_folder)

#____ Parameters _____
# retro_image_shape = (512,540)
# wf_image_shape    = (512,540)
# topo_image_shape  = (512,540)
# pachy_image_shape = (512,540)
# image_shape       = {'Retro':(512,540),'WF':(512,540),'Pachy':(512,540),'Topo':(512,540)}

pat_list      = os.listdir(dataFolder)
# im_format_out = '.png'  # format of images to export
invalid_set   = [-1000,np.nan, None] # representing invalid values
num_rings     = 24  # number of rings in the topo maps
num_phy       = 256 # number of angles
x             = np.zeros((num_phy,num_rings))
y             = np.zeros((num_phy,num_rings))
# Preallocation for topo maps
maps_radial = {'Left':{'Radii':np.zeros((num_phy,num_rings)),
                        'Axial':np.zeros((num_phy,num_rings)),
                        'Tangential':np.zeros((num_phy,num_rings)),
                        'Height':np.zeros((num_phy,num_rings))},
                'Right':{'Radii':np.zeros((num_phy,num_rings)),
                        'Axial':np.zeros((num_phy,num_rings)),
                        'Tangential':np.zeros((num_phy,num_rings)),
                        'Height':np.zeros((num_phy,num_rings))}}
p_num         = 0   # running counter
angles        = np.linspace(0,2*np.pi,num_phy) # angles for the topo maps
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
                    # Extract Topo maps
                    for eIdx in ['Left', 'Right']:
                        # Extract Topo maps
                        topo_txt_file = os.path.join(tempExtractFolder,'clientdb',fIdx,kIdx,'Topo',f'{eIdx}Topo_Meas_1.txt')
                        if os.path.exists(topo_txt_file):
                            # extract the three type of maps: Axial, Tangential, Height
                            print(f'[Info][TopoMapExtractors] topo txt file {eIdx} found')
                            map_type = ['Radii','Axial','Tangential','Height']
                            with open(topo_txt_file,'r',encoding='utf-16') as file:
                                lines = file.readlines()
                            for mIdx in map_type:
                                for lIdx in lines:
                                    c_map = re.findall(mIdx+'_\d+',lIdx)
                                    if len(c_map)>0:
                                        c_map     = c_map[0] # map type
                                        # Find the line number
                                        lNum      = int(c_map.replace(mIdx+'_',''))
                                        line_txt  = lIdx.replace(mIdx+f'_{lNum}= ','').replace('\n','')
                                        # Extract all numbers
                                        nums = re.findall(r'[+-]?\d+\.?\d+',line_txt)
                                        # Assign num to the map
                                        for nIdx in range(num_rings):
                                            val = float(nums[nIdx])
                                            if val in invalid_set:
                                                maps_radial[eIdx][mIdx][lNum,nIdx] = None
                                            else:
                                                maps_radial[eIdx][mIdx][lNum,nIdx] = val
                                            # translate to cartesian
                                            rad = maps_radial[eIdx]['Radii'][lNum,nIdx]
                                            if rad in invalid_set:
                                                x[lNum,nIdx] = None
                                                y[lNum,nIdx] = None
                                            else:
                                                x[lNum,nIdx] = rad*np.cos(angles[lNum])
                                                y[lNum,nIdx] = rad*np.sin(angles[lNum])
                                # export the map
                                pd.DataFrame(maps_radial[eIdx][mIdx]).to_csv(os.path.join(topo_maps,mIdx,f'{pIdx}_{eIdx}.csv'),index=False,sep=',')
                        else:

                            print(f'[Warning][TopoMapExtractor] File {topo_txt_file} does not exist for eye: {eIdx}')
        except:
            print(f'[Error][TopoMapExtractor]s Cannot process file: {os.path.join(dataFolder,pIdx)}')
        # empty the temp folder
        shutil.rmtree(os.path.join(tempExtractFolder,'clientdb'))
        p_num+=1

        # fig = plt.figure()
        # ax  = plt.axes(projection='3d')
        # z   = maps_radial['Right']['Tangential']
        # # z   = z/(np.nanmax(z)-np.nanmin(z))
        # ax.scatter3D(x,y,z,marker='.',cmap='Pastel1')
        # fig.show()