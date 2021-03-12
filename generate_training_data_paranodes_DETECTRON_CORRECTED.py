# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 09:46:29 2020

@author: tiger
"""

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


 ***NEED TO INSTALL numexpr!!!
 
@author: Tiger


"""

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

from PYTORCH_dataloader import *
from functional.plot_functions_CLEANED import *
from functional.data_functions_CLEANED import *
from functional.data_functions_3D import *
from functional.tracker import *
from functional.IO_func import *
import glob, os
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order

import tkinter
from tkinter import filedialog
import os


import tifffile as tiff
    
truth = 0

def plot_max(im, ax=0, fig_num=1):
     max_im = np.amax(im, axis=ax)
     plt.figure(fig_num); plt.imshow(max_im[:, :])
     return max_im
     

""" removes detections on the very edges of the image """
def clean_edges(im, depth, w, h, extra_z=1, extra_xy=5):
     labelled = measure.label(im)
     cc_coloc = measure.regionprops(labelled)
    
     cleaned_im = np.zeros(np.shape(im))
     for obj in cc_coloc:
         #max_val = obj['max_intensity']
         coords = obj['coords']
         
         bool_edge = 0
         for c in coords:
              if (c[0] <= 0 + extra_z or c[0] >= depth - extra_z):
                   #print('badz')
                   bool_edge = 1
                   break;
              if (c[1] <= 0 + extra_xy or c[1] >= w - extra_xy):
                   #print('badx')
                   bool_edge = 1
                   break;                                       
              if (c[2] <= 0 + extra_xy or c[2] >= h - extra_xy):
                   #print('bady')
                   bool_edge = 1
                   break;                                        
                   
                   
    
         if not bool_edge:
              #print('good')
              for obj_idx in range(len(coords)):
                   cleaned_im[coords[obj_idx,0], coords[obj_idx,1], coords[obj_idx,2]] = 1

     return cleaned_im                     
            

resize_bool = 0

### LARGE CROPS
# input_size = 256
# depth = 16   # ***OR can be 160


### SMALLER, for 2D
input_size = 320
depth = 10   # ***OR can be 160
max_proj = 1
num_truth_class = 1 + 1 # for reconstruction
multiclass = 0

name = '_TRAINING_DETECTRON_CORRECTED'


""" Load detectron """
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
import detectron2
# from detectron2.utils.logger import setup_logger
# setup_logger()
  
# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow
  
# import some common detectron2 utilities
#from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("balloon_train",)

balloon_metadata = MetadataCatalog.get("balloon_train")

### NEED TO SET THIS FOR BINARY MASK TRUTH DATA
cfg.INPUT.MASK_FORMAT = 'bitmask'
cfg.TEST.EVAL_PERIOD = 100
  
cfg.DATASETS.TEST = ("balloon_valid",)
cfg.DATALOADER.NUM_WORKERS = 2
  
  
  
from detectron2.evaluation.sem_seg_evaluation import SemSegEvaluator
from detectron2.evaluation import COCOEvaluator


from LossEvalHook import *
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR,"inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks            
  
  
  
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 200000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
  

""" Inference and evaluation """
# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0009999.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)
                          
                          

""" Select multiple folders for analysis AND creates new subfolder for results output """
root = tkinter.Tk()
# get input folders
another_folder = 'y';
list_folder = []
input_path = "./"
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
    
    sav_dir = input_path + '/' + foldername + name
 
    """ Load filenames from tiff """
    images = glob.glob(os.path.join(input_path,'*.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    #examples = [dict(input=i,truth=i.replace('_RAW_REGISTERED_substack_1_110.tif','_TRUTH_REGISTERED_substack_1_110.tif'), ilastik=i.replace('_RAW_REGISTERED_substack_1_110.tif','_Object_Predictions.tiff')) for i in images]


    try:
        # Create target Directory
        os.mkdir(sav_dir)
        print("Directory " , sav_dir ,  " Created ") 
    except FileExistsError:
        print("Directory " , sav_dir ,  " already exists")
        
    sav_dir = sav_dir + '/'
    
    # sav_dir_max_project = sav_dir + '/max/'

    # try:
    #     # Create target Directory
    #     os.mkdir(sav_dir_max_project)
    #     print("Directory " , sav_dir_max_project ,  " Created ") 
    # except FileExistsError:
    #     print("Directory " , sav_dir_max_project ,  " already exists")
        

    
    # Required to initialize all
    batch_size = 1;
    
    input_batch = []; truth_batch = [];
    weights = [];
    
    plot_jaccard = [];
    
    output_stack = [];
    output_stack_masked = [];
    all_PPV = [];
    input_im_stack = [];
    
    empty = 1


    total_samples = 0
    
    expectedLen = 10000
    overlap_percent = 0.50  
    for i in range(0, len(images), 2):
        input_name = images[i]
        input_im = np.asarray(tiff.imread(input_name), dtype=np.uint8)

        truth_name = images[i + 1]
        truth_im = np.asarray(tiff.imread(truth_name), dtype=np.float32)  ### converting directly to uint8 losses data!!!
        truth_im[truth_im > 0] = 1
        

        
        """ Analyze each block with offset in all directions """
        quad_size = input_size
        quad_depth = depth
        im_size = np.shape(input_im);
        width = im_size[1];  height = im_size[2]; depth_im = im_size[0];
          
        num_quads = np.floor(width/quad_size) * np.floor(width/quad_size) * np.floor (depth_im/quad_depth);

        quad_idx = 1;

      
        segmentation = np.zeros([depth_im, width, height])
        input_im_check = np.zeros(np.shape(input_im))
        total_blocks = 0;
        
        all_xyz = []
        for x in range(1, width + quad_size, round(quad_size - quad_size * overlap_percent)):
             if x + quad_size > width:
                  difference = (x + quad_size) - width
                  x = x - difference
                      
             for y in range(1, height + quad_size, round(quad_size - quad_size * overlap_percent)):
                  
                  if y + quad_size > height:
                       difference = (y + quad_size) - height
                       y = y - difference

                  for z in range(1, depth_im + quad_depth, round(quad_depth - quad_depth * overlap_percent)):
                      batch_x = []; batch_y = [];
                      
                      if z + quad_depth > depth_im:
                           difference = (z + quad_depth) - depth_im
                           z = z - difference

                      quad_intensity = input_im[z:z + quad_depth, x:x + quad_size, y:y + quad_size]
                      quad_truth = truth_im[z:z + quad_depth, x:x + quad_size, y:y + quad_size]

                      """ Save block """                          
                      #filename = input_name.split('\\')[-1]  # on Windows
                      filename = input_name.split('/')[-1] # on Ubuntu
                      filename = filename.split('.')[0:-1]
                      filename = '.'.join(filename)
                                                
                      filename = filename.split('RAW_REGISTERED')[0]
                      
                      """ Check if repeated """
                      skip = 0
                      for coord in all_xyz:
                           if coord == [x,y,z]:
                                skip = 1
                                break                      
                      if skip:
                           continue
                      all_xyz.append([x, y, z])  
                       
                      
                      """ If want to save images as well """
                     
                      max_quad_intensity = plot_max(quad_intensity, ax=0, fig_num=1)


                      """ Do INFERENCE """
                      grayImage = cv2.cvtColor(max_quad_intensity, cv2.COLOR_GRAY2BGR)
                      
                      outputs = predictor(grayImage)
                      #cv2_imshow(grayImage)

                      ### RANDOMLY SELECT SAMPLES TO TEST
                      from detectron2.utils.visualizer import ColorMode
                      balloon_metadata = MetadataCatalog.get("balloon_train")

                      # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
                      v = Visualizer(grayImage[:, :, ::-1],
                                            metadata=balloon_metadata,
                                            scale=0.5, 
                                            instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
                      )
                      #out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                      #cv2_imshow(out.get_image()[:, :, ::-1])


                      outputs["instances"].pred_masks.shape
                      
                      """ loop through prediction masks 
                      
                              - and figure out where it is located in the 3D volume
                      """
                      from skimage.filters import threshold_otsu
                      from skimage import measure
                      for pred in outputs["instances"].pred_masks.to("cpu"):
                          tmp_3D = np.copy(quad_intensity)
                          coords_mask = np.transpose(np.where(~pred))
                          
                          tmp_3D[:, coords_mask[:, 0], coords_mask[:, 1]]  = 0
                          
                          thresh = threshold_otsu(tmp_3D)
                          binary = tmp_3D > thresh

                          """ Clean segmentation by removing objects on the edge """
                          binary = clean_edges(binary, quad_depth, w=quad_size, h=quad_size, extra_z=0, extra_xy=5)
                          #quad_truth = cleaned_seg

                          cc = measure.regionprops(np.asarray(binary, dtype=np.uint8))
                          
                          
                          ### MIGHT NEED TO ALSO LOOP THROUGH MULTIPLE OBJECTS AND FIND LARGEST???
                          if len(cc) == 0:
                             continue;
                          elif len(cc) > 1:
                              print('many objs: PROBLEM???')
                          elif len(cc) == 1:
                              obj_coords = cc[0].coords
                              obj_coords_unscaled = cc[0].coords
                              
                              obj_coords[:, 0] = obj_coords[:, 0] + z
                              obj_coords[:, 1] = obj_coords[:, 1] + x
                              obj_coords[:, 2] = obj_coords[:, 2] + y
                              
                              
                              """ Check if was found in truth """
                              if (quad_truth[obj_coords_unscaled[:, 0], obj_coords_unscaled[:, 1], obj_coords_unscaled[:, 2]] > 0).any():
                                ### also indicate in the truth_im that this object has been detected:
                                quad_truth[obj_coords_unscaled[:, 0], obj_coords_unscaled[:, 1], obj_coords_unscaled[:, 2]] = -1
                                
                                truth_im[obj_coords[:, 0], obj_coords[:, 1], obj_coords[:, 2]] = -1
                                    
                                segmentation[obj_coords[:, 0], obj_coords[:, 1], obj_coords[:, 2]] = np.max(segmentation) + 1
                                
                              
                              #print('added obj')
                             
        # max_seg = plot_max(segmentation, ax=0, fig_num=2)
        """ Add any final objects that were NOT in TRUTH """
        binary_truth = np.copy(truth_im)
        binary_truth[binary_truth == -1] = 255
        binary_truth[binary_truth > 0] = 1
        
        label = measure.label(binary_truth)
        cc = measure.regionprops(np.asarray(label, dtype=np.uint8), intensity_image=truth_im)
        
        for obj in cc:
            coord = obj.coords

            min_val = obj.min_intensity
            if min_val > -1 and len(coord) > 1:   ### if was NOT matched before and also has a length > 1 pixel
                #truth_im[coord[:, 0], coord[:, 1], coord[:, 2]] = 
                segmentation[coord[:, 0], coord[:, 1], coord[:, 2]] = np.max(segmentation) + 1

                
        total_samples += 1
        print(total_samples)


        # plot_max(input_im, fig_num=0)
        # plot_max(truth_im, fig_num=1)
        
        
        imsave(sav_dir + filename + str(int(x)) + '_' + str(int(y)) + '_' + str(int(z)) +'_OUTPUT_SEG.tif', np.uint8(segmentation))
        
        max_seg = plot_max(segmentation, ax=0, fig_num=5)
        imsave(sav_dir + filename + str(int(x)) + '_' + str(int(y)) + '_' + str(int(z)) +'_OUTPUT_SEG_max_proj.tif', np.uint8(max_seg))      
        
        
        
        # binary_truth = np.copy(truth_im)
        # binary_truth[binary_truth == -1] = 255
        # plot_max(binary_truth, fig_num=3)
  








