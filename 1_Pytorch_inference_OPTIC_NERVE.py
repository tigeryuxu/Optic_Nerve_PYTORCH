# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 10:25:37 2018

@author: Tiger
"""

#import tensorflow as tf
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from natsort import natsort_keygen, ns
from skimage import measure
import scipy
import cv2 as cv
from natsort import natsort_keygen, ns

#from UNet import *
#from UNet_3D import *
import glob, os
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order

#from csbdeep.internals import predict
import tifffile as tifffile
import tkinter
from tkinter import filedialog


""" Required to allow correct GPU usage ==> or else crashes """
# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# tf.keras.backend.set_session(tf.Session(config=config))


import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from functional.plot_functions_CLEANED import *
from functional.data_functions_CLEANED import *
from functional.data_functions_3D import *
from functional.tracker import *
from functional.IO_func import *


from layers.UNet_pytorch_online import *
from layers.unet_nested import *
from layers.unet3_3D import *
from layers.switchable_BN import *

from losses_pytorch.HD_loss import *

from UNet_functions_PYTORCH import *


torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.enabled = True  # new thing? what do? must be True

""" Define GPU to use """
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


""" Import network """
#unet = UNet_online()

"""  Network Begins: """

s_path = './(1) optic_nerve_5x5_medium/'; HD = 0; alpha = 0; nested = 0; b_norm = 1; sps_bool = 0; sp_weight_bool = 0;



""" AMOUNT OF EDGE TO ELIMINATE 
    scaling???
"""

overlap_percent = 0.25
input_size = 1024
# depth = 32
num_truth_class = 2

""" TO LOAD OLD CHECKPOINT """
# Read in file names
onlyfiles_check = glob.glob(os.path.join(s_path,'check_*'))
onlyfiles_check.sort(key = natsort_key1)

""" Find last checkpoint """       
last_file = onlyfiles_check[-1]
split = last_file.split('check_')[-1]
num_check = split.split('.')
checkpoint = num_check[0]
checkpoint = 'check_' + checkpoint
num_check = int(num_check[0])

check = torch.load(s_path + checkpoint, map_location=device)
tracker = check['tracker']

# unet = check['model_type']
# unet.load_state_dict(check['model_state_dict'])
# unet.to(device)
# unet.eval()
# #unet.training # check if mode set correctly


unet = check['model_type']; unet.load_state_dict(check['model_state_dict'])
unet.to(device); unet.eval()
#unet.training # check if mode set correctly

print('parameters:', sum(param.numel() for param in unet.parameters()))


# """ load mean and std """  
#input_path = './normalize/'
mean_arr = tracker.mean_arr
std_arr = tracker.std_arr


""" Select multiple folders for analysis AND creates new subfolder for results output """
root = tkinter.Tk()
# get input folders
another_folder = 'y';
list_folder = []
input_path = "./"

initial_dir = '/media/user/storage/Data/(4) Optic nerve project/Optic Nerve/EAE_miR_AAV2/'

#input_path = "/Users/Neuroimmunology Unit/Anaconda3/AI stuff/MyelinUNet/Source/"
while(another_folder == 'y'):
    input_path = filedialog.askdirectory(parent=root, initialdir= initial_dir,
                                        title='Please select input directory')
    input_path = input_path + '/'
    
    print('Do you want to select another folder? (y/n)')
    another_folder = input();   # currently hangs forever
    #another_folder = 'y';

    list_folder.append(input_path)
    initial_dir = input_path
        

""" Loop through all the folders and do the analysis!!!"""
for input_path in list_folder:
    foldername = input_path.split('/')[-2]
    sav_dir = input_path + '/' + foldername + '_output_optic_nerve_5x5_medium_' + str(num_check)

    """ For testing ILASTIK images """
    images = glob.glob(os.path.join(input_path,'*.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    examples = [dict(input=i,truth=i.replace('.tif','_truth.tif'), ilastik=i.replace('.tif','_single_Object Predictions_.tiff')) for i in images]


    # images = glob.glob(os.path.join(input_path,'*_single_channel.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    # images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    # examples = [dict(input=i,truth=i.replace('_single_channel.tif','_truth.tif'), ilastik=i.replace('_single_channel.tif','_single_Object Predictions_.tiff')) for i in images]
     
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
         
    
         """ TRY INFERENCE WITH PATCH-BASED analysis from TORCHIO """
         with torch.set_grad_enabled(False):  # saves GPU RAM            
            input_name = examples[i]['input']            
            input_im = open_image_sequence_to_3D(input_name, width_max='default', height_max='default', depth='default')
            input_im = np.squeeze(input_im)
            
            #input_im = tifffile.imread(input_name)
            #input_im = input_im[:, :, 1]  
            ### only get channel 2 (green)
            if len(input_im.shape) >= 3 and input_im.shape[-1] == 3:
                input_im = input_im[:, :, 1]
                

   
            """ Analyze each block with offset in all directions """
            
            # Display the image
            #max_im = plot_max(input_im, ax=0)
            
            print('Starting inference on volume: ' + str(i) + ' of total: ' + str(len(examples)))
            #plot_max(input_im)
            
            segmentation = UNet_inference_by_subparts_PYTORCH_2D(unet, device, input_im, overlap_percent, quad_size=input_size,
                                                      mean_arr=mean_arr, std_arr=std_arr, num_truth_class=num_truth_class,
                                                      skip_top=0)
           
            segmentation[segmentation > 0] = 255
            #plot_max(segmentation)
            filename = input_name.split('/')[-1]
            filename = filename.split('.')[0:-1]
            filename = '.'.join(filename)
            

            segmentation = np.asarray(segmentation, np.uint8)
            imsave(sav_dir + filename + '_' + str(int(i)) +'_segmentation.tif', segmentation)
            segmentation[segmentation > 0] = 1
            
            input_im = np.asarray(input_im, np.uint8)
            imsave(sav_dir + filename + '_' + str(int(i)) +'_input_im.tif', input_im)
            
            
            """ Load in truth data for comparison!!! sens + precision """
            # truth_name = examples[i]['truth']
            
            # truth_im = open_image_sequence_to_3D(truth_name, width_max='default', height_max='default', depth='default')
            # truth_im[truth_im > 0] = 1                                   
            
            # truth_im_cleaned = clean_edges(truth_im, extra_z=1, extra_xy=3, skip_top=1)
                                            
            # TP, FN, FP, truth_im_cleaned, cleaned_seg = find_TP_FP_FN_from_seg(segmentation, truth_im_cleaned, size_limit=5)

                 
            # #plot_max(truth_im_cleaned)
            # #plot_max(cleaned_seg)
            

            # if TP + FN == 0: TP;
            # else: sensitivity = TP/(TP + FN);     # PPV
                   
            # if TP + FP == 0: TP;
            # else: precision = TP/(TP + FP);     # precision

            # print(filename)
            # print(str(sensitivity))
            # print(str(precision))
            
            # truth_im_cleaned = np.asarray(truth_im_cleaned, np.uint8)
            # imsave(sav_dir + filename + '_' + str(int(i)) +'_truth_im_cleaned.tif', truth_im_cleaned)            
            
            # """ Compare with ilastik () if you want to """
            # ilastik_compare = 0
            # if ilastik_compare:
                 
            #     """ Load in truth data for comparison!!! sens + precision """
            #     ilastik_name = examples[i]['ilastik']

            #     ilastik_im = open_image_sequence_to_3D(ilastik_name, width_max='default', height_max='default', depth='default')
            #     ilastik_im[ilastik_im > 0] = 1                                   
                 
            #     ilastik_im_cleaned = clean_edges(ilastik_im, depth_im, w=width, h=height, extra_z=1, extra_xy=3)
                                                   
            #     TP, FN, FP, truth_im_cleaned, cleaned_seg = find_TP_FP_FN_from_seg(ilastik_im, truth_im_cleaned, size_limit=10)
                                    
                               
            #     ilastik_im_cleaned = np.asarray(ilastik_im_cleaned, np.uint8)
            #     imsave(sav_dir + filename + '_' + str(int(i)) +'_ilastik_cleaned.tif', ilastik_im_cleaned)
                 
                      
            #     plot_max(ilastik_im_cleaned)
                 
            #     if TP + FN == 0: TP;
            #     else: sensitivity = TP/(TP + FN);     # PPV
                        
            #     if TP + FP == 0: TP;
            #     else: precision = TP/(TP + FP);     # precision
        
            #     print(str(sensitivity))
            #     print(str(precision))                                         
            

            """ Save as 3D stack """
            if len(output_stack) == 0:
                output_stack = segmentation
                #output_stack_masked = seg_train_masked
                input_im_stack = input_im
            else:
                """ Keep this if want to keep all the outputs as single stack """
                output_stack = np.dstack([output_stack, segmentation])
                ##output_stack_masked = np.dstack([output_stack_masked, seg_train_masked])
                input_im_stack = np.dstack([input_im_stack, input_im])

            # """ save individual tiffs as well """    
            # plt.imsave(sav_dir + filename + '_' + str(i) + '_input_im.tif', (input_im))
            # plt.imsave(sav_dir + filename + '_' + str(i) + '_output_seg.tif', (segmentation))
            
    
    
    
    """ Pre-processing """
    # 1) get more data (and good data)
    # 2) overlay seg masks and binarize to get better segmentations???
        
    """ Post-processing """
    """ (1) removes all things that do not appear in > 5 slices!!!"""
    all_seg, all_blebs, all_eliminated = slice_thresh(output_stack, slice_size=5)
    
    filename_split = filename.split('_z')[0]
    
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
    input_im_stack_m_tiffs = np.asarray(convert_matrix_to_multipage_tiff(save_input_im_stack), dtype=np.uint8) * 255
    imsave(sav_dir + "1) " +  filename_split + '_z' + '_input_stack.tif', input_im_stack_m_tiffs)
    
    print("Saving post-processed slice threshed images")
    all_seg_m_tiffs = np.asarray(convert_matrix_to_multipage_tiff(all_seg), dtype=np.uint8) * 255
    imsave(sav_dir + "2) " + filename_split + '_z' + '_ORIGINAL_post-processed.tif', all_seg_m_tiffs)
    all_blebs_m_tiffs = np.asarray(convert_matrix_to_multipage_tiff(all_blebs), dtype=np.uint8) * 255
    imsave(sav_dir + "3) " + filename_split + '_z' + '_BLEBS_post-processed.tif', all_blebs_m_tiffs)
    all_eliminated_m_tiffs = np.asarray(convert_matrix_to_multipage_tiff(all_eliminated), dtype=np.uint8) * 255
    imsave(sav_dir + "4) " + filename_split + '_z' + '_ELIM_post-processed.tif', all_eliminated_m_tiffs)
    
    
    print("Saving post-processed intensity threshed images")
    all_blebs_THRESH_m_tiffs = np.asarray(convert_matrix_to_multipage_tiff(all_blebs_THRESH), dtype=np.uint8) * 255
    imsave(sav_dir + "5) " + filename_split + '_z' + '_THRESH_and_SLICED_post-processed.tif', all_blebs_THRESH_m_tiffs)
    
    
    print("Saving post-processed distance thresheded images")
    final_bleb_m_tiffs = np.asarray(convert_matrix_to_multipage_tiff(final_bleb_matrix), dtype=np.uint8) * 255
    imsave(sav_dir + "6) " + filename_split + '_z' + '_DISTANCE_THRESHED_post-processed.tif', final_bleb_m_tiffs)
    elim_matrix_m_tiffs = np.asarray(convert_matrix_to_multipage_tiff(elim_matrix), dtype=np.uint8) * 255
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


