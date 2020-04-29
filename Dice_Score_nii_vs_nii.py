#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 22:55:10 2020

@author: agapi

Script to calculate the Dice score between the real label and the predicted segmentation.
In addition, the 20 best and worst predictions respectively are plotted and saved as .png.
"""

import os
import numpy as np
from imageio import imread
import matplotlib
from matplotlib import pyplot as plt
import SimpleITK as sitk
import glob
import matplotlib.gridspec as gridspec
plt.switch_backend('agg')
from skimage.measure import find_contours
import nibabel as nib


def make_2D_plots(X, y, y_pred, n_best=20, n_worst=20):
    #PLotting the results'
    img_rows = X.shape[1]
    img_cols = img_rows
    axis =  tuple( range(1, X.ndim ) )
    scores = numpy_dice(y, y_pred, axis=axis )
    sort_ind = np.argsort( scores )[::-1]
    indice = np.nonzero( y.sum(axis=axis) )[0]
    #Add some best and worst predictions
    img_list = []
    count = 1
    for ind in sort_ind:
        if ind in indice:
            img_list.append(ind)
            count+=1
        if count>n_best:
            break

    segm_pred = y_pred[img_list].reshape(-1,img_rows, img_cols)
    img = X[img_list].reshape(-1,img_rows, img_cols)
    segm = y[img_list].reshape(-1, img_rows, img_cols).astype('float32')       
    
    n_cols= 4
    n_rows = int( np.ceil(len(img)/n_cols) )

    fig = plt.figure(figsize=[ 4*n_cols, int(4*n_rows)] )
    gs = gridspec.GridSpec( n_rows , n_cols )

    for mm in range( len(img) ):

        ax = fig.add_subplot(gs[mm])
        ax.imshow(img[mm], cmap='gray')
        contours = find_contours(segm[mm], 0.01, fully_connected='high')
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='r')

        contours = find_contours(segm_pred[mm], 0.01, fully_connected='high')
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='b')

        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1)  # aspect ratio of 1

    fig.savefig('../Morphsnakes/2D_plots/best_predictions_train.png', bbox_inches='tight', dpi=300 )
        

    img_list = []
    count = 1
    for ind in sort_ind[::-1]:
        if ind in indice:
            img_list.append(ind)
            count+=1
        if count>n_worst:
            break

    segm_pred = y_pred[img_list].reshape(-1,img_rows, img_cols)
    img = X[img_list].reshape(-1,img_rows, img_cols)
    segm = y[img_list].reshape(-1, img_rows, img_cols).astype('float32')


    n_cols= 4
    n_rows = int( np.ceil(len(img)/n_cols) )

    fig = plt.figure(figsize=[ 4*n_cols, int(4*n_rows)] )
    gs = gridspec.GridSpec( n_rows , n_cols )

    for mm in range( len(img) ):

        ax = fig.add_subplot(gs[mm])
        ax.imshow(img[mm], cmap='gray')
        contours = find_contours(segm[mm], 0.01, fully_connected='high')
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='r')

        contours = find_contours(segm_pred[mm], 0.01, fully_connected='high')
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='b')

        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1)  # aspect ratio of 1

    fig.savefig('../Morphsnakes/2D_plots/worst_predictions_train.png', bbox_inches='tight', dpi=300 )
    
    
    
def make_3D_plots(background, filename, fig = None):
   
    from mpl_toolkits.mplot3d import Axes3D
    # PyMCubes package is required for `visual_callback_3d`
    try:
        import mcubes
    except ImportError:
        raise ImportError("PyMCubes is required for 3D `visual_callback_3d`")
    
    # Prepare the visual environment.
    if fig is None:
        fig = plt.figure()
    fig.clf()
    ax = fig.add_subplot(111, projection='3d')

    if ax.collections:
        del ax.collections[0]

    coords, triangles = mcubes.marching_cubes(background, 0.5)
    ax.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2],
                    triangles=triangles)
    plt.pause(0.1)
    fig.savefig(os.path.join('/home/agapi/Desktop/MasterThesis/Morphsnakes', filename + ".png"))




def numpy_dice(y_true, y_pred, axis=None, smooth=1.0):

    intersection = y_true*y_pred
    return ( 2. * intersection.sum(axis=axis) +smooth)/ (np.sum(y_true, axis=axis) + np.sum(y_pred, axis=axis) +smooth )



def check_predictions(data, true_label, prediction, plot):

    if not os.path.isdir('../Morphsnakes/2D_plots'):
        os.mkdir('../Morphsnakes/2D_plots')
  
    print('Accuracy:', numpy_dice(true_label, prediction))
	
    #Create 2D plots with best and worst predictions
    make_2D_plots(data, true_label, prediction)
    
    
    #Create the 3D plot of the real label and the prediction respectively.

#    for i in range(3):
#        true_label = np.transpose(true_label, (1, 2, 0))
#        for angle in range(4):
#            true_label = np.rot90(true_label)
#            make_3D_plots(true_label, "true_label_3D_plot" + str(i) + str(angle))

#    for i in range(3):
#        prediction = np.transpose(prediction, (1, 2, 0))
#        for angle in range(4):
#            prediction = np.rot90(prediction)
#            make_3D_plots(prediction, "predicted_label_3D_plot" + str(i) + str(angle))
            
#    make_3D_plots(np.rot90(prediction, k=1), "predicted_label_3D_plot")



def read_cases(the_list=None, folder='../data/train/', masks=True):
    
    filenames = glob.glob(folder+'/*0082*.nii.gz', recursive=True) #Add pancreas to exclude label files.     
    
    for filename in filenames:
        itkimage = sitk.ReadImage(filename)
        itkimage = sitk.Flip(itkimage, [False, True, False]) #Flip images, because they are shown reversed.
        img = sitk.GetArrayFromImage(itkimage)
        return img



if __name__=='__main__':

    x_data = read_cases(folder = '/home/agapi/Desktop/MasterThesis/Datasets/NIH_nii')
    y_true = read_cases(folder = '/home/agapi/Desktop/MasterThesis/Datasets/NIH_labels_edited')
    y_pred = read_cases(folder = '/home/agapi/Desktop/MasterThesis/Morphsnakes/nii_test')  
       
    check_predictions(x_data, y_true, y_pred, True)
