#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 00:42:35 2020

@author: agapi

Script for optimization after applying MorphGAC. 
Specifically, a pixel label (0 or 1) is investigated in 3 slices.
If it is 1 in the first and the last slice, but not in the middle one, then it becomes 1 too.
Similarly, if it is 0 in the first and the last, but not in the middle one, then it becomes 0 too.
"""


import os
import glob
import nibabel as nib



def read_cases(the_list=None, folder='../data/train/', masks=True):
    
    filenames = glob.glob(folder+'/*0082*.nii.gz', recursive=True) #Add pancreas to exclude label files.     
    
    for filename in filenames:
        image = nib.load(filename)
        array = image.get_fdata()
        
        return image, array



def optimize(segm_3D_array):
    
    total_slices = segm_3D_array.shape[2]   

    # iterate through slices
    for current_slice in range(0, total_slices-1):        
        
       
        first_slice = segm_3D_array[:, :,current_slice -1]
        middle_slice = segm_3D_array[:, :,current_slice]
        last_slice = segm_3D_array[:, :,current_slice + 1]        
        print(str(current_slice) +" / "+ str(total_slices))
    
        for x in range(0, middle_slice.shape[0]):
            for y in range(0, middle_slice.shape[1]):
                if (middle_slice[x,y] == 0 and first_slice[x,y] == 1 and last_slice[x,y] == 1):

                    middle_slice[x,y] = 1                   


        for x in range(0, middle_slice.shape[0]):
            for y in range(0, middle_slice.shape[1]):
                if (middle_slice[x,y] == 1 and first_slice[x,y] == 0 and last_slice[x,y] == 0):

                    middle_slice[x,y] = 0      


        segm_3D_array[:, :,current_slice] = middle_slice    
    
    return segm_3D_array


    

if __name__=='__main__':
    
    input_dir = '/home/agapi/Desktop/MasterThesis/Morphsnakes/nii_test'
    
    img, y_pred  = read_cases(folder = input_dir)  
    
    #Apply the "filling" optimization
    y_pred_optimized = optimize(y_pred)
    
    
#    y_pred_optimized = np.rot90(y_pred_optimized, k=-1)
    
    segmentation = nib.Nifti1Image(y_pred_optimized, img.affine, img.header)
    img.header.get_data_shape()
    
    #Construct a new empty header
    empty_header = nib.Nifti1Header()
    empty_header.get_data_shape()    
    
    
    nib.save(segmentation, os.path.join(input_dir,'opt_segmentation_0082.nii.gz'))

    print('Optimized segmentation saved!')    