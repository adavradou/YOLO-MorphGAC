#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:02:51 2020

@author: agapi

Script: It applies the morphological snakes (GAC) on Bounding Boxes (YOLO detections). 
The Bounding Boxes are already extracted on .txt files.
The script does the following:
    1) For each Bounding Box, reads the corresponding image (512x512).
    2) Creates an empty 512x512 image (one for each .txt file).
    3) Crops each detection/Bounding Box from the CT-image.
    4) Applies MorphGAC on each detection/Bounding Box.
    5) Puts each segmentation to the empty image.
    4) Saves the final 512x512 segmentation image for each .txt file.
"""


import os
import logging
import glob
import numpy as np
from imageio import imread
import matplotlib
from matplotlib import pyplot as plt
import re
import SimpleITK as sitk
import morphsnakes_DICE as ms
import scipy.misc
import cv2



def visual_callback_2d(background, name, output_dir, fig=None):
    """
    Returns a callback than can be passed as the argument `iter_callback`
    of `morphological_geodesic_active_contour` and
    `morphological_chan_vese` for visualizing the evolution
    of the levelsets. Only works for 2D images.
    
    Parameters
    ----------
    background : (M, N) array
        Image to be plotted as the background of the visual evolution.
    fig : matplotlib.figure.Figure
        Figure where results will be drawn. If not given, a new figure
        will be created.
    
    Returns
    -------
    callback : Python function
        A function that receives a levelset and updates the current plot
        accordingly. This can be passed as the `iter_callback` argument of
        `morphological_geodesic_active_contour` and
        `morphological_chan_vese`.
    
    """
    
    # Prepare the visual environment.
    if fig is None:
        fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(background, cmap=plt.cm.gray)

    ax2 = fig.add_subplot(1, 2, 2)
    ax_u = ax2.imshow(np.zeros_like(background), vmin=0, vmax=1)
    plt.pause(0.001)

    def callback(levelset, iteration, iterations_num): 
        

        if ax1.collections:
            del ax1.collections[0]
        ax1.contour(levelset, [0.5], colors='r')
        ax_u.set_data(levelset)
        fig.canvas.draw()
        plt.pause(0.001)
        
        #For the final iteration do:
        if iteration == iterations_num-1:
            plt.savefig(os.path.join(output_dir, str(name[:-5]) + ".png"))
            
    return callback
    


def do_nothing(background, name, output_dir, fig=None):
    """
    Literally, it does nothing. Used to run MorphGAC.
    """  

    def callback(levelset, iteration, iterations_num):         
        pass
            
    return callback
    



def read_filenames(input_path, output_path, BBox_path):
    """
    Finds and sorts the respective files in the paths.
    """
    images = glob.glob(input_path+'*_0082_*.tiff', recursive=True) 
    images = sorted(images, key=lambda f: int(''.join(filter(str.isdigit, f))))    
 
    boundingBoxes = glob.glob(BBox_path+'*.txt', recursive=True) 
    boundingBoxes = sorted(boundingBoxes, key=lambda f: int(''.join(filter(str.isdigit, f))))
       
    return images, boundingBoxes

           
def get_BB_coordinates(BBarray, margin):
    """
    Gets as input the Bounding Box array and a margin parameter.
    Margin parameter is optional and is the number of pixels from the actual Bounding Box.
    Returns the Bounding Box coordinates.
    """
    
    x_1 = int(BBarray[0]) - margin #-10 
    x_2 = x_1 + int(BBarray[2]) + 2* margin #+ 20 
    y_1 = int(BBarray[1]) - margin #- 10 
    y_2 = y_1 + int(BBarray[3]) + 2* margin #+ 20    
    
    return x_1, x_2, y_1, y_2



if __name__=='__main__':
    
    #Initialize the paths
    input_image_path = '/home/agapi/Desktop/MasterThesis/Datasets/NIH_tiff_16bit_range_-200_300_rotated/' #Location of testing images.
    output_image_path = '/home/agapi/Desktop/MasterThesis/Morphsnakes/results/' #Location of output images.
    BB_path = '/home/agapi/Desktop/MasterThesis/Yolo_pancreas/BoundingBoxes_NIH_test_-200_300/'

    [test_filenames, BB_filenames] = read_filenames(input_image_path, output_image_path, BB_path)
        
        
    for BB in BB_filenames:    
        
        #Create empty 512x512 2D array with zeros. 
        final_image = np.zeros([512,512],dtype=np.uint8) 
        
        f = open(BB, "r") #Read .txt file containing the bounding box coordinates.

        for bounding_box in f:    
            
            #Read respective image
            image_name = os.path.join(input_image_path, os.path.basename(BB))
            image_name = image_name[:-4] + ".tiff"
            original_image_array = cv2.imread(image_name) /255.0     
            
            bounding_box = bounding_box[:-2] #Remove '/n' characters from array.
            bounding_box = np.fromstring(bounding_box, dtype=np.int, sep=' ') #Convert string array to np.array.
            
            # Get bounding box  (Change 0 to another value to add margin).
            [x1, x2, y1, y2] = get_BB_coordinates(bounding_box, 0)
                                             
            #Crops each detection/bounding box.
            test_image_cropped = original_image_array[y1:y2, x1:x2]        
            
            #Apply MorphGAC on the cropped image.
            print('Running: MorphGAC...')        
            img = test_image_cropped[:,:,0]              
            gimg = ms.inverse_gaussian_gradient(img, alpha=1000, sigma=2)   
                     
            # Initialization of the level set from the center of the Bounding Box.
            init_ls = ms.circle_level_set(img.shape, (int(img.shape[0]/2), int(img.shape[1]/2)), 2)        
            filename = os.path.basename(image_name)           
            
            # Callback for visual plotting        
#            callback = visual_callback_2d(img, filename, output_image_path) #Uncomment to plot figures.
            callback = do_nothing(img, filename, output_image_path)        
            
        
            # Apply the MorphGAC algorithm for each Bounding Box. 
            BB_segmentation = ms.morphological_geodesic_active_contour(gimg, iterations=50, init_level_set=init_ls,
                                                     smoothing=1, threshold=0.31,
                                                     balloon=1, iter_callback = callback)
                                
            #Put the segmentation to its original position in the 512x512 image.
            for x in range(x1, x2):
                for y in range(y1, y2):
                    final_image[y,x] = BB_segmentation[y-y1, x-x1]       
                    
            #Save images in tiff format. 
            scipy.misc.imsave(os.path.join(output_image_path, os.path.basename(image_name)), final_image*255)
                    
   
