from __future__ import print_function

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 10:25:37 2018

@author: Neuroimmunology Unit
"""

# -*- coding: utf-8 -*-
"""
Created on Sunday Dec. 24th
============================================================


 ***NEED TO INSTALL SCIPY???
 
 
 Try reducing network size
 

@author: Tiger
"""


import tensorflow as tf
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from natsort import natsort_keygen, ns
from skimage import measure
import pickle as pickle
import os
import zipfile
import scipy
import cv2 as cv
from natsort import natsort_keygen, ns

from plot_functions import *
from data_functions import *
#from post_process_functions import *
from data_functions import *
from split_im_to_patches import *
from UNet import *
import glob, os


import tkinter
from tkinter import filedialog
import os
    
truth = 0

# Initialize everything with specific random seeds for repeatability
tf.reset_default_graph() 
tf.set_random_seed(1); np.random.seed(1)

"""  Network Begins:
"""
## for saving
#s_path = 'J:/DATA_2017-2018/Optic_nerve/EAE_miR_AAV2/2018.08.07/ON_11/Checkpoints/3rd_run_SHOWCASE/'
s_path = './Checkpoints/2nd_OPTIC_NERVE_run_full_dataset/'
s_path = './Checkpoints/2nd_OPTIC_NERVE_run_CONTINUE_PATCHES/'

#num_check = 400000
num_check = 200000
#s_path = './Checkpoints/3rd_OPTIC_NERVE_large_network/'


resize = 0
im_scale = 0.300

tf_size = 1024
## for input
#input_path = 'J:/DATA_2017-2018/Optic_nerve/EAE_miR_AAV2/2018.08.07/ON_11/'
#input_path = 'J:/DATA_2017-2018/Optic_nerve/EAE_miR_AAV2/2018.08.16/EAE_A3/'
#input_path = './2018.08.16/EAE_A3/'
#input_path = './Training Data/'

# in 2018


""" load mean and std """  
mean_arr = load_pkl('', 'mean_arr.pkl')
std_arr = load_pkl('', 'std_arr.pkl')
               

# Variable Declaration
x = tf.placeholder('float32', shape=[None, tf_size, tf_size, 3], name='InputImage')
y_ = tf.placeholder('float32', shape=[None, tf_size, tf_size, 2], name='CorrectLabel')
training = tf.placeholder(tf.bool, name='training')
weight_matrix = tf.placeholder('float32', shape=[None, tf_size, tf_size, 2], name = 'weighted_labels')


""" Creates network and cost function"""
y, y_b, L1, L2, L3, L4, L5, L6, L7, L8, L9, L9_conv, L10, L11, logits, softMaxed = create_network_SMALL(x, y_, training)
#y, y_b, L1, L2, L3, L4, L5, L6, L7, L8, L9, L9_conv, L10, L11, logits, softMaxed = create_network(x, y_, training)
accuracy, jaccard, train_step, cross_entropy, loss, cross_entropy, original = costOptm(y, y_b, logits, weight_matrix, weight_mat=True)

sess = tf.InteractiveSession()

""" TO LOAD OLD CHECKPOINT """
saver = tf.train.Saver()
checkpoint = '_' + str(num_check)
saver.restore(sess, s_path + 'check' + checkpoint)
    


#root = tkinter.Tk()
#sav_dir = filedialog.askdirectory(parent=root, initialdir="/Users/Neuroimmunology Unit/Anaconda3/AI stuff/MyelinUNet/Source/",
#                                        title='Please select saving directory')
#sav_dir = sav_dir + '/'

""" Select multiple folders for analysis AND creates new subfolder for results output """
root = tkinter.Tk()
# get input folders
another_folder = 'y';
list_folder = []
input_path = "/Users/Neuroimmunology Unit/Anaconda3/AI stuff/MyelinUNet/Source/"
while(another_folder == 'y'):
    input_path = filedialog.askdirectory(parent=root, initialdir= input_path,
                                        title='Please select input directory')
    input_path = input_path + '/'
    
    another_folder = input();   # currently hangs forever
    #another_folder = 'y';

    list_folder.append(input_path)
        

""" Loop through all the folders and do the analysis!!!"""
for input_path in list_folder:
    foldername = input_path.split('/')[-2]
    sav_dir = input_path + '/' + foldername + '_analytic_results'
 
    """ Load filenames from zip """
    images = glob.glob(os.path.join(input_path,'*.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    examples = [dict(input=i,truth=i.replace('.tif','truth.tif')) for i in images]
    
    if truth:
        images = glob.glob(os.path.join(input_path,'*_pos_input.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
        images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
        examples = [dict(input=i,truth=i.replace('input.tif','truth.tif')) for i in images]

    try:
        # Create target Directory
        os.mkdir(sav_dir)
        print("Directory " , sav_dir ,  " Created ") 
    except FileExistsError:
        print("Directory " , sav_dir ,  " already exists")
        
    sav_dir = sav_dir + '/'
    
    # Required to initialize all
    batch_size = 1;
    
    batch_x = []; batch_y = [];
    weights = [];
    
    plot_jaccard = [];
    
    output_stack = [];
    output_stack_masked = [];
    all_PPV = [];
    input_im_stack = [];
    for i in range(len(examples)):
            
            input_name = examples[i]['input']
            input_im = np.asarray(Image.open(input_name), dtype=np.float32)
           
            """ NEED TO CONVERT TO np.uint8 if the original input is np.uint16!!!"""
            # NORMALIZED BECAUSE IMAGE IS uint16 ==> do same when actually running images!!!
            input_im = np.asarray(Image.open(input_name))
            if input_im.dtype == 'uint16':
                input_im = np.asarray(input_im, dtype=np.float32)
                input_im = cv.normalize(input_im, 0, 255, cv.NORM_MINMAX)
                input_im = input_im * 255
                
            input_im = np.asarray(input_im, dtype= np.float32)
            
                    
            size_whole = input_im.shape[0]
            
            size = int(size_whole) # 4775 and 6157 for the newest one
            if resize:
                size = int((size * im_scale) / 0.45) # 4775 and 6157 for the newest one
                input_im = resize_adaptive(Image.fromarray(input_im), size, method=Image.BICUBIC)
                input_im = np.asarray(input_im, dtype=np.float32)

            
            # convert to single channel
            if len(input_im.shape) > 2:
                input_im = input_im[:, :, 1]   # take the green channel
                
            # scale to between 0 -- 255
            if input_im.max() > 255:
                input_im = input_im / (input_im.max()/255)
            input_save = np.copy(input_im)
                           
            
            """ Divide input into patches if larger than tf_size, later put patches together
                OR
                pad image to be tf_size so can have input of any size
            """
            patches = np.zeros(1)
            if input_im.shape[0] > tf_size or input_im.shape[1] > tf_size:
                patches = patchify(input_im, patch_shape=(tf_size,tf_size), overlap=10)
                if not type(patches) is np.ndarray:
                    patches = np.array(patches)
            elif input_im.shape[0] < tf_size or input_im.shape[1] < tf_size:
                
                delta_w = tf_size - input_im.shape[0]
                delta_h = tf_size - input_im.shape[1]
                top, bottom = delta_h//2, delta_h-(delta_h//2)
                left, right = delta_w//2, delta_w-(delta_w//2)
                
                color = [0, 0, 0]
                input_im = cv.copyMakeBorder(input_im, top, bottom, left, right, cv.BORDER_CONSTANT,
                    value=color)
                
                #cv.imshow("image", new_im)
                       
                     
                    

            """ if trying to run with test images and not new images, load the truth """
            if truth:
                truth_name = examples[i]['truth']
                truth_tmp = np.asarray(Image.open(truth_name), dtype=np.float32)
                       
                """ convert truth to 2 channel image """
                if "_neg_" in truth_name:
                    truth_im = np.zeros(np.shape(truth_tmp) + (2,))
                    truth_im[:, :, 0] = np.ones(np.shape(truth_tmp))   # background
                    truth_im[:, :, 1] = np.zeros(np.shape(truth_tmp))   # blebs
                else:
                    channel_1 = np.copy(truth_tmp)
                    channel_1[channel_1 == 0] = 1
                    channel_1[channel_1 == 255] = 0
                            
                    channel_2 = np.copy(truth_tmp)
                    channel_2[channel_2 == 255] = 1   
                    
                    truth_im = np.zeros(np.shape(truth_tmp) + (2,))
                    truth_im[:, :, 0] = channel_2   # background
                    truth_im[:, :, 1] = channel_1   # blebs
                    
                    # some reasons values are switched in Barbara's images
                    if "_BARBARA_" in truth_name:
                        truth_im[:, :, 0] = channel_1   # background
                        truth_im[:, :, 1] = channel_2   # blebs
                        
                
                blebs_label = np.copy(truth_im[:, :, 1])
    
                """ Get spatial AND class weighting mask for truth_im """
                sp_weighted_labels = spatial_weight(blebs_label,edgeFalloff=10,background=0.01,approximate=True)
                
                """ OR DO class weighting ONLY """
                #c_weighted_labels = class_weight(blebs_label, loss, weight=10.0)        
                
                """ Create a matrix of weighted labels """
                weighted_labels = np.copy(truth_im)
                weighted_labels[:, :, 1] = sp_weighted_labels
    
                batch_y.append(truth_im)
                weights.append(weighted_labels)
                
            else:
                batch_y.append(np.zeros([tf_size,tf_size,2]))
                weights.append(np.zeros([tf_size,tf_size,2]))
    
            
            
            """ if do split patches, then:
                (1) need to skip any patches that are empty to save time
                (2) analyze each patch image individually, then recombine
            """
            feed_dict = []
            if patches.any():
                seg_output_patches = np.zeros(np.shape(patches))
                idx = 0
                for input_im in patches:
                    
                    #input_im = patches[idx]
                    input_RGB = np.zeros(np.shape(input_im) + (3,))
                    input_RGB[:, :, 1] = input_im
                    input_RGB = np.asarray(input_RGB, dtype=np.uint8)
                    input_im = input_RGB

                    # SKIP IF NOTHING IS IN THE IMAGE
                    input_im[input_im < 50] = 0
                    if np.count_nonzero(input_im) < 1000:
                        seg_train = np.zeros(np.shape(patches[0])) 
                        #seg_train[0:100, 0:500] = 25;   # for debugging
                        seg_output_patches[idx, :, :] = seg_train
                        idx = idx + 1     
                        print("skipped")
                        continue

                    #plt.figure(8); plt.imshow(input_im); plt.pause(1)
                    #plt.figure(9); plt.imshow(seg_train); plt.pause(2)
                
                    """ maybe remove normalization??? """
                    input_im = normalize_im(input_im, mean_arr, std_arr) 
                        
                    """ set inputs and truth """
                    batch_x.append(input_im)
            
                            
                    """ Feed into training loop """
                    feed_dict = {x:batch_x, y_:batch_y, training:1, weight_matrix:weights}                 
                    output_train = softMaxed.eval(feed_dict=feed_dict)
                    seg_train = np.argmax(output_train, axis = -1)[0] 
                    seg_train[0:50, 0:50] = 1;   # for debugging
                    seg_output_patches[idx, :, :] = seg_train
                    idx = idx + 1
                    batch_x = []
                    
                k = q
                seg_train = collect(seg_output_patches, (input_save.shape[0], input_save.shape[1]), overlap=10)
            else:
                    input_RGB = np.zeros(np.shape(input_im) + (3,))
                    input_RGB[:, :, 1] = input_im
                    input_RGB = np.asarray(input_RGB, dtype=np.uint8)
                    input_im = input_RGB

                    """ maybe remove normalization??? """
                    input_im = normalize_im(input_im, mean_arr, std_arr) 
                        
                    """ set inputs and truth """
                    batch_x.append(input_im)
            
                    """ Feed into training loop """
                    feed_dict = {x:batch_x, y_:batch_y, training:1, weight_matrix:weights}                 
                    output_train = softMaxed.eval(feed_dict=feed_dict)
                    seg_train = np.argmax(output_train, axis = -1)[0] 
                
                
                
            """ Makes into RGB image """
            input_RGB = np.zeros(np.shape(input_save) + (3,))
            input_RGB[:, :, 1] = input_save
            input_RGB = np.asarray(input_RGB, dtype=np.uint8)
            input_save = input_RGB

            
            
            
            """ Plot for debug """
            if truth:
                plt.figure(1); 
                plt.subplot(221); plt.imshow(np.asarray(input_im, dtype = np.uint8)); plt.title('Input');
                plt.subplot(222); plt.imshow(sp_weighted_labels); plt.title('weighted');    plt.pause(0.005)
                plt.subplot(223); plt.imshow(channel_1); plt.title('background');
                plt.subplot(224); plt.imshow(channel_2); plt.title('blebs');
    
            print('Analyzed: %d of total: %d' %(i + 1, len(examples)))
               
               
            plt.close(2)
           
            """ Training Jaccard """
            jacc_t = jaccard.eval(feed_dict=feed_dict)
            plot_jaccard.append(jacc_t)           
                                  
            """ Plot outputs """
                         
            #plt.figure(2);
            plt.figure(num=2, figsize=(40, 40), dpi=80, facecolor='w', edgecolor='k')
            if truth:  # plot truth
                plt.subplot(121); plt.imshow(truth_tmp); plt.title('Truth');
            else:  # or just plot the input image
                plt.subplot(121); plt.imshow(np.asarray(input_save, dtype=np.uint8)); plt.title('Input');
                
            filename_split = input_name.split('\\')[-1]
            filename_split = filename_split.split('.')[0]
                
            plt.subplot(122); plt.imshow(seg_train); plt.title('Output');                            
            plt.savefig(sav_dir + filename_split + '_' + str(i) + '_compare_output.png', bbox_inches='tight')
                  
            batch_x = []; batch_y = []; weights = [];
                  
            #plt.imsave(sav_dir + filename_split + '_' + str(i) + '_output_mask.tif', (seg_train), cmap='binary_r')
            
    
            """ Compute accuracy """
            if truth:
                overlap_im = seg_train + truth_im[:, :, 1]
                binary_overlap = overlap_im > 0
                labelled = measure.label(binary_overlap)
                cc_overlap = measure.regionprops(labelled, intensity_image=overlap_im)
    
                """ (1) Find # True Positives identified (overlapped) """
                masked = np.zeros(seg_train.shape)
                all_no_overlap = np.zeros(seg_train.shape)
                truth_no_overlap = np.zeros(seg_train.shape)   # ALL False Negatives
                seg_no_overlap = np.zeros(seg_train.shape)     # All False Positives
                for M in range(len(cc_overlap)):
                    overlap_val = cc_overlap[M]['MaxIntensity']
                    overlap_coords = cc_overlap[M]['coords']
                    if overlap_val > 1:    # if there is overlap
                        for T in range(len(overlap_coords)):
                            masked[overlap_coords[T,0], overlap_coords[T,1]] = overlap_im[overlap_coords[T,0], overlap_coords[T,1]]   # TRUE POSITIVES
                    else:  # no overlap
                        for T in range(len(overlap_coords)):
                            all_no_overlap[overlap_coords[T,0], overlap_coords[T,1]] = overlap_im[overlap_coords[T,0], overlap_coords[T,1]]     
                            truth_no_overlap[overlap_coords[T,0], overlap_coords[T,1]] = overlap_im[overlap_coords[T,0], overlap_coords[T,1]]     
                            seg_no_overlap[overlap_coords[T,0], overlap_coords[T,1]] = overlap_im[overlap_coords[T,0], overlap_coords[T,1]]     
    
                labelled = measure.label(masked)
                cc_masked_TPs = measure.regionprops(labelled)
                TP = len(cc_masked_TPs)
                         
                labelled = measure.label(truth_no_overlap)
                cc_truth_FNs = measure.regionprops(labelled)
                FN = len(cc_truth_FNs)
    
                labelled = measure.label(seg_no_overlap)
                cc_seg_FPs = measure.regionprops(labelled)
                FP = len(cc_seg_FPs)
                            
                PPV = TP/(TP + FP)
                    
                print("PPV value for image %d is: %.3f" %(i + 1, PPV))            
                all_PPV.append(PPV)
    
    
            """ Save as 3D stack """
            if len(output_stack) == 0:
                output_stack = seg_train
                #output_stack_masked = seg_train_masked
                input_im_stack = input_save
            else:
                """ Keep this if want to keep all the outputs as single stack """
                #output_stack = np.dstack([output_stack, seg_train])
                ##output_stack_masked = np.dstack([output_stack_masked, seg_train_masked])
                #input_im_stack = np.dstack([input_im_stack, input_save])

            """ save individual tiffs as well """    
            plt.imsave(sav_dir + filename_split + '_' + str(i) + '_input_im.tif', (input_save))
            plt.imsave(sav_dir + filename_split + '_' + str(i) + '_output_seg.tif', (seg_train))
            
    
    
    
    """ Pre-processing """
    # 1) get more data (and good data)
    # 2) overlay seg masks and binarize to get better segmentations???
        
    """ Post-processing """
    """ (1) removes all things that do not appear in > 5 slices!!!"""
    all_seg, all_blebs, all_eliminated = slice_thresh(output_stack, slice_size=5)
    
    filename_split = filename_split.split('_z')[0]
    
    """ (2) DO THRESHOLDING TO SHRINK SEGMENTATION SIZE, but do THRESH ON 3D array!!! """
    save_input_im_stack = np.copy(input_im_stack)
    input_im_stack[all_blebs == 0] = 0     # FIRST MASK THE ORIGINAL IMAGE
    
    # then do thresholding
    from skimage import filters
    val = filters.threshold_otsu(input_im_stack)
    mask = input_im_stack > val
    
    # closes image to make less noisy """
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    blebs_opened_masked = cv.morphologyEx(np.asarray(mask, dtype=np.uint8), cv.MORPH_CLOSE, kernel)
    
    """ apply slice THRESH again"""
    all_seg_THRESH, all_blebs_THRESH, all_eliminated_THRESH = slice_thresh(blebs_opened_masked, slice_size=5)
    
    """ (3) Find vectors of movement and eliminate blobs that migrate """
    final_bleb_matrix, elim_matrix = distance_thresh(all_blebs_THRESH, average_thresh=15, max_thresh=15)
    
    
    print("Saving input images")
    input_im_stack_m_tiffs = convert_matrix_to_multipage_tiff(save_input_im_stack)
    imsave(sav_dir + "1) " +  filename_split + '_z' + '_input_stack.tif', input_im_stack_m_tiffs)
    
    print("Saving post-processed slice threshed images")
    all_seg_m_tiffs = convert_matrix_to_multipage_tiff(all_seg)
    imsave(sav_dir + "2) " + filename_split + '_z' + '_ORIGINAL_post-processed.tif', all_seg_m_tiffs)
    all_blebs_m_tiffs = convert_matrix_to_multipage_tiff(all_blebs)
    imsave(sav_dir + "3) " + filename_split + '_z' + '_BLEBS_post-processed.tif', all_blebs_m_tiffs)
    all_eliminated_m_tiffs = convert_matrix_to_multipage_tiff(all_eliminated)
    imsave(sav_dir + "4) " + filename_split + '_z' + '_ELIM_post-processed.tif', all_eliminated_m_tiffs)
    
    
    print("Saving post-processed intensity threshed images")
    all_blebs_THRESH_m_tiffs = convert_matrix_to_multipage_tiff(all_blebs_THRESH)
    imsave(sav_dir + "5) " + filename_split + '_z' + '_THRESH_and_SLICED_post-processed.tif', all_blebs_THRESH_m_tiffs)
    
    
    print("Saving post-processed distance thresheded images")
    final_bleb_m_tiffs = convert_matrix_to_multipage_tiff(final_bleb_matrix)
    imsave(sav_dir + "6) " + filename_split + '_z' + '_DISTANCE_THRESHED_post-processed.tif', final_bleb_m_tiffs)
    elim_matrix_m_tiffs = convert_matrix_to_multipage_tiff(elim_matrix)
    imsave(sav_dir + "7) " + filename_split + '_z' + '_DISTANCE_THRESHED_elimed_post-processed.tif', elim_matrix_m_tiffs)
    
    
        
        
        
    
    """ Pseudo-local thresholding (applies Otsu to each individual bleb) """
    #binary_overlap = all_blebs_THRESH > 0
    #labelled = measure.label(binary_overlap)
    #cc_overlap = measure.regionprops(labelled)
    #
    #total_blebs = 0
    #pseudo_threshed_stack = np.zeros(np.shape(input_im_stack))
    #for bleb in cc_overlap:
    #    cur_bleb_coords = bleb['coords']
    #    cur_bleb_mask = convert_vox_to_matrix(cur_bleb_coords, np.zeros(output_stack.shape))
    #    
    #    val = filters.threshold_otsu(cur_bleb_mask)
    #    mask = cur_bleb_mask > val    
    #    
    #    pseudo_threshed_stack = pseudo_threshed_stack + mask
    #
    #    total_blebs = total_blebs + 1    
    #        
    #    print("Total analyzed: " + str(total_blebs) + "of total blebs: " + str(len(cc_overlap)))
        
    
    """ Plotting as interactive scroller """
    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, final_bleb_matrix)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()

