#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 18:00:21 2020

@author: agapi
Script to create a .nii volume from 2D .tiff images.
It is possible that 3D volume has more slices than the number .tiff images.
That is because only non-empty tiff images are used.
The rest of the slices are left empty.
"""


        
import os
import glob
os.system
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import re
import nibabel as nib


img = nib.load('/home/agapi/Desktop/MasterThesis/Datasets/NIH_nii/PANCREAS_0082_0_Pancreas.nii.gz')

input_test_path = '/home/agapi/Desktop/MasterThesis/Morphsnakes/test_output/' #Location of testing images.
output_test_path = '/home/agapi/Desktop/MasterThesis/Morphsnakes/nii_test/' #Location of output images.

images_2D_filenames = glob.glob(input_test_path+'*.tiff', recursive=True) 
images_2D_filenames = sorted(images_2D_filenames, key=lambda f: int(''.join(filter(str.isdigit, f))))  

total_slices = img.header.get_data_shape()[2] 

#Create empty 512x512xslices .nii array with zeros. 
nii_array = np.zeros([512,512,total_slices],dtype=np.uint8)  


for image in images_2D_filenames:
    
    #Get the slice number from the .tiff filename
    sliceNumber = os.path.basename(image[:-5]) #Remove '.tiff.
    sliceNumber = sliceNumber.split('_') #Split at each underscore, e.g. ['PANCREAS', '0082', '0', 'Pancreas93']
    sliceNumber = sliceNumber[3] #Keep the last one that contains the slice number
    sliceNumber = re.sub('\D', '', sliceNumber) #Keep only the number from the whole string
    
    #Read the corresponding 2D image
    image_2D_array = cv2.imread(image) / 255
    
    #Insert it in the respective slice of the volume
    nii_array[:,:,int(sliceNumber)] = image_2D_array[:,:,0]


nii_array = np.rot90(nii_array, k=-1)
#plt.imshow(nii_array[:,:,115])

segmentation = nib.Nifti1Image(nii_array, img.affine, img.header)
img.header.get_data_shape() # DO AT first

#Construct a new empty header
empty_header = nib.Nifti1Header()
empty_header.get_data_shape()

#Save the .nii volume.
nib.save(segmentation, os.path.join(output_test_path,'segmentation.nii.gz'))