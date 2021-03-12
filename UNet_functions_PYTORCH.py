#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:10:34 2020

@author: user
"""


import torch
import numpy as np
from functional.data_functions_3D import *
from functional.data_functions_CLEANED import *
from functional.plot_functions_CLEANED import *



def normalize(im, mean, std):
    return (im - mean)/std
    
    
""" Perform inference by splitting input volume into subparts """
def UNet_inference_by_subparts_PYTORCH_2D(unet, device, input_im, overlap_percent, quad_size, mean_arr, std_arr, skip_top=0, num_truth_class=1):
     im_size = np.shape(input_im);
     width = im_size[0];  height = im_size[1];
        
     segmentation = np.zeros([width, height])
     total_blocks = 0;
     all_xyz = []                                               
     

     for x in range(0, width + quad_size, round(quad_size - quad_size * overlap_percent)):
          if x + quad_size > width:
               difference = (x + quad_size) - width
               x = x - difference
                    
          for y in range(0, height + quad_size, round(quad_size - quad_size * overlap_percent)):
               
                if y + quad_size > height:
                    difference = (y + quad_size) - height
                    y = y - difference
               
                 #for z in range(0, depth_im + quad_depth, round(quad_depth - quad_depth * overlap_percent)):
                   #batch_x = []; batch_y = [];
         
                   # if z + quad_depth > depth_im:
                   #      difference = (z + quad_depth) - depth_im
                   #      z = z - difference   
                       
                """ Check if repeated """
                skip = 0
                for coord in all_xyz:
                     if coord == [x,y]:
                          skip = 1
                          break                      
                if skip:  continue
                     
                all_xyz.append([x, y])
                
                quad_intensity = input_im[x:x + quad_size, y:y + quad_size];  
                

                """ Normalization """
                quad_intensity = np.asarray(quad_intensity, dtype=np.float32)
                quad_intensity = normalize(quad_intensity, mean_arr, std_arr)

                             
                """ Analyze """
                """ set inputs and truth """
                quad_intensity = np.expand_dims(quad_intensity, axis=-1)
                batch_x = quad_intensity
                batch_x = np.moveaxis(batch_x, -1, 0)
                batch_x = np.expand_dims(batch_x, axis=0)
                
                
                #batch_y = np.zeros([1, num_truth_class, quad_depth, quad_size, quad_size])

            
                """ Convert to Tensor """
                inputs_val = torch.tensor(batch_x, dtype = torch.float, device=device, requires_grad=False)
                #labels_val = torch.tensor(batch_y, dtype = torch.long, device=device, requires_grad=False)
      
                # forward pass to check validation
                output_val = unet(inputs_val)

                """ Convert back to cpu """                                      
                output_tile = output_val.cpu().data.numpy()            
                output_tile = np.moveaxis(output_tile, 1, -1)
                seg_train = np.argmax(output_tile[0], axis=-1)  

                 
                
                """ Debug """
                # plot_max(quad_intensity)
                # plot_max(seg_train)
 
                """ Clean segmentation by removing objects on the edge """
                # if skip_top and z == 0:
                #      #print('skip top')
                #      cleaned_seg = clean_edges(seg_train, extra_z=0, extra_xy=0, skip_top=skip_top)                                             
                # else:
                #      cleaned_seg = clean_edges(seg_train, extra_z=0, extra_xy=0)
                
                """ ADD IN THE NEW SEG??? or just let it overlap??? """                         
                #segmentation[z:z + quad_depth, x:x + quad_size, y:y + quad_size] = cleaned_seg                        
                segmentation[x:x + quad_size, y:y + quad_size] = seg_train + segmentation[x:x + quad_size, y:y + quad_size]
      
                total_blocks += 1
                
                #print('inference on sublock: ')
                #print([x, y, z])
        
     return segmentation