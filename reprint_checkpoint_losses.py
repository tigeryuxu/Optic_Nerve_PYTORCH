#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 09:44:45 2021

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Sunday Dec. 24th
============================================================
@author: Tiger
"""

""" ALLOWS print out of results on compute canada """
import matplotlib
matplotlib.rc('xtick', labelsize=8)
matplotlib.rc('ytick', labelsize=8) 
#matplotlib.use('Agg')


import matplotlib.pyplot as plt
import numpy as np
import glob, os
import datetime
import time
from sklearn.model_selection import train_test_split

from natsort import natsort_keygen, ns
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order


from PYTORCH_dataloader import *
from functional.plot_functions_CLEANED import *
from functional.data_functions_CLEANED import *
from functional.data_functions_3D import *
from functional.tracker import *


from layers.UNet_pytorch_online import *
from layers.unet_nested import *
from layers.unet3_3D import *
from layers.switchable_BN import *

from losses_pytorch.HD_loss import *


""" Libraries to load """
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim



""" optional dataviewer if you want to load it """
#import napari
# with napari.gui_qt():
#     viewer = napari.view_image(seg_val)

torch.backends.cudnn.benchmark = True  ### set these options to improve speed
torch.backends.cudnn.enabled = True 

if __name__ == '__main__':
        
    """ Define GPU to use """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    """" path to checkpoints """       

    s_path = './(8) paranode_detection_5x5_medium_UNet_1e5_only_CLEANED_DATA/'; HD = 0; alpha = 0;


    """ path to input data """
    #input_path = '/media/user/storage/Data/(1) snake seg project/InternodeAnalysis_Round2/shortened_for_training_data/paranodes for RL/paranodes for RL_quads_PARANODES/'

    input_path = '/media/user/storage/Data/(1) snake seg project/InternodeAnalysis_Round2/shortened_for_training_CORRECTED/paranodes for RL/paranodes for RL_quads_PARANODES/'
    
    all_trees = [];

    """ Load filenames from tiff """
    images = glob.glob(os.path.join(input_path,'*_INPUT.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    examples = [dict(input=i,truth=i.replace('_INPUT.tif','_TRUTH.tif')) for i in images]
    counter = list(range(len(examples)))

    # ### REMOVE IMAGE 1 from training data
    idx_skip = []
    for idx, im in enumerate(examples):
        filename = im['input']
        if '491_ROI1_' in filename:
            print('skip')
            idx_skip.append(idx)

    ### USE THE EXCLUDED IMAGE AS VALIDATION/TESTING
    examples_test = examples[idx_skip[0]:idx_skip[-1]]

    examples = [i for j, i in enumerate(examples) if j not in idx_skip]
          
            
    counter = list(range(len(examples)))
    counter_val = list(range(len(examples_test)))  ### NEWLY ADDED!!!
    
    # """ load mean and std for normalization later """  
    mean_arr = np.load('./normalize/' + 'mean_VERIFIED.npy')
    std_arr = np.load('./normalize/' + 'std_VERIFIED.npy')   


    num_workers = 2;
 
    save_every_num_epochs = 1; plot_every_num_epochs = 1; validate_every_num_epochs = 1;      
    
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

    print('restoring weights from: ' + checkpoint)
    check = torch.load(s_path + checkpoint, map_location=device)
    #check = torch.load(s_path + checkpoint, map_location='cpu')
    #check = torch.load(s_path + checkpoint, map_location=device)
        
    
    tracker = check['tracker']
    loss_function = check['loss_function']

    tracker.idx_valid = counter_val
    
    #tracker.idx_valid = idx_skip   ### IF ONLY WANT 
    
    
    tracker.idx_train = []

    tracker.batch_size = 1
    tracker.train_loss_per_batch = [] 
    tracker.train_jacc_per_batch = []
    tracker.val_loss_per_batch = []; tracker.val_jacc_per_batch = []
    
    tracker.train_ce_pb = []; tracker.train_hd_pb = []; tracker.train_dc_pb = [];
    tracker.val_ce_pb = []; tracker.val_hd_pb = []; tracker.val_dc_pb = [];
 
    """ Get metrics per epoch"""
    tracker.train_loss_per_epoch = []; tracker.train_jacc_per_epoch = []
    tracker.val_loss_per_eval = []; tracker.val_jacc_per_eval = []
    tracker.plot_sens = []; tracker.plot_sens_val = [];
    tracker.plot_prec = []; tracker.plot_prec_val = [];
    tracker.lr_plot = [];
    tracker.iterations = 0;
    tracker.cur_epoch = 0;
    
    
    #tracker.


    for check_file in onlyfiles_check:      
        last_file = check_file
        """ Find last checkpoint """       
        #last_file = onlyfiles_check[-1]
        split = last_file.split('check_')[-1]
        num_check = split.split('.')
        checkpoint = num_check[0]
        checkpoint = 'check_' + checkpoint

        print('restoring weights from: ' + checkpoint)
        check = torch.load(s_path + checkpoint, map_location=device)
        #check = torch.load(s_path + checkpoint, map_location='cpu')
        #check = torch.load(s_path + checkpoint, map_location=device)

        # """ Print info """
        # tracker = check['tracker']
        # tracker.print_essential(); 
        # continue;
        
        
        unet = check['model_type']
        unet.load_state_dict(check['model_state_dict']) 
        unet.eval();   unet.to(device)

        print('parameters:', sum(param.numel() for param in unet.parameters()))  
        
        """ Clean up checkpoint file """
        del check
        torch.cuda.empty_cache()

                
        """ Create datasets for dataloader """
        training_set = Dataset_tiffs(tracker.idx_train, examples, tracker.mean_arr, tracker.std_arr,
                                               sp_weight_bool=tracker.sp_weight_bool, transforms = tracker.transforms)
        val_set = Dataset_tiffs(tracker.idx_valid, examples_test, tracker.mean_arr, tracker.std_arr,
                                          sp_weight_bool=tracker.sp_weight_bool, transforms = 0)
        
        """ Create training and validation generators"""
        val_generator = data.DataLoader(val_set, batch_size=tracker.batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=True, drop_last = True)
    
        # training_generator = data.DataLoader(training_set, batch_size=tracker.batch_size, shuffle=True, num_workers=num_workers,
        #                   pin_memory=True, drop_last=True)
        
        
        #print('Total # training images per epoch: ' + str(len(training_set)))
        print('Total # validation images: ' + str(len(val_set)))
        
    
        """ Epoch info """
        #train_steps_per_epoch = len(tracker.idx_train)/tracker.batch_size
        validation_size = len(tracker.idx_valid)
        #epoch_size = len(tracker.idx_train)    
       
        

         
         
        """ Should I keep track of loss on every single sample? and iteration? Just not plot it??? """   
        loss_val = 0; jacc_val = 0
        precision_val = 0; sensitivity_val = 0; val_idx = 0;
        iter_cur_epoch = 0;
        if tracker.cur_epoch % validate_every_num_epochs == 0:
            
              with torch.set_grad_enabled(False):  # saves GPU RAM
                  unet.eval()
                  for batch_x_val, batch_y_val, spatial_weight in val_generator:
                        
                        """ Transfer to GPU to normalize ect... """
                        inputs_val, labels_val = transfer_to_GPU(batch_x_val, batch_y_val, device, mean_arr, std_arr)
             
                        # forward pass to check validation
                        output_val = unet(inputs_val)
                        loss = loss_function(output_val, labels_val)
                        if torch.is_tensor(spatial_weight):
                               spatial_tensor = torch.tensor(spatial_weight, dtype = torch.float, device=device, requires_grad=False)          
                               weighted = loss * spatial_tensor
                               loss = torch.mean(weighted)
                        else:
                               loss = torch.mean(loss)  
          
                        """ Training loss """
                        tracker.val_loss_per_batch.append(loss.cpu().data.numpy());  # Training loss
                        loss_val += loss.cpu().data.numpy()
                                         
                        """ Calculate jaccard on GPU """
                        jacc = jacc_eval_GPU_torch(output_val, labels_val)
                        jacc = jacc.cpu().data.numpy()
                        
                        jacc_val += jacc
                        tracker.val_jacc_per_batch.append(jacc)   


                        """ Convert back to cpu """                                      
                        output_val = output_val.cpu().data.numpy()            
                        output_val = np.moveaxis(output_val, 1, -1)
                        
                        """ Calculate sensitivity + precision as other metrics ==> only ever on ONE IMAGE of a batch"""
                        batch_y_val = batch_y_val.cpu().data.numpy() 
                        seg_val = np.argmax(output_val[0], axis=-1)  
                        TP, FN, FP = find_TP_FP_FN_from_im(seg_val, batch_y_val[0])
                                       
                        if TP + FN == 0: TP;
                        else: sensitivity = TP/(TP + FN); sensitivity_val += sensitivity;    # PPV
                                       
                        if TP + FP == 0: TP;
                        else: precision = TP/(TP + FP);  precision_val += precision    # precision             
              
                        val_idx = val_idx + tracker.batch_size
                        print('Validation: ' + str(val_idx) + ' of total: ' + str(validation_size))
                
                        iter_cur_epoch += 1
                
              

              tracker.val_loss_per_eval.append(loss_val/iter_cur_epoch)
              tracker.val_jacc_per_eval.append(jacc_val/iter_cur_epoch)   
                   
              tracker.plot_prec.append(precision_val/iter_cur_epoch)
              tracker.plot_sens.append(sensitivity_val/iter_cur_epoch)   
                  
              """ Add to scheduler to do LR decay """
              #scheduler.step()
           
        """ Plot metrics every epoch """      
        if tracker.cur_epoch % plot_every_num_epochs == 0:       
             
              """ Plot sens + precision + jaccard + loss """
              plot_metric_fun(tracker.plot_sens, tracker.plot_sens_val, class_name='', metric_name='sensitivity', plot_num=30)
              plt.figure(30); plt.savefig(s_path + 'Sensitivity.png')
                    
              plot_metric_fun(tracker.plot_prec, tracker.plot_prec_val, class_name='', metric_name='precision', plot_num=31)
              plt.figure(31); plt.savefig(s_path + 'Precision.png')
           
              plot_metric_fun(tracker.train_jacc_per_epoch, tracker.val_jacc_per_eval, class_name='', metric_name='jaccard', plot_num=32)
              plt.figure(32); plt.savefig(s_path + 'Jaccard.png')
                   
                 
              plot_metric_fun(tracker.train_loss_per_epoch, tracker.val_loss_per_eval, class_name='', metric_name='loss', plot_num=33)
              plt.figure(33); plt.yscale('log'); plt.savefig(s_path + 'loss_per_epoch.png')          
                   
              
              plot_metric_fun(tracker.lr_plot, [], class_name='', metric_name='learning rate', plot_num=35)
              plt.figure(35); plt.savefig(s_path + 'lr_per_epoch.png') 
     

              """ Plot metrics per batch """                
              plot_metric_fun(tracker.train_jacc_per_batch, [], class_name='', metric_name='jaccard', plot_num=34)
              plt.figure(34); plt.savefig(s_path + 'Jaccard_per_batch.png')
                                
              plot_cost_fun(tracker.train_loss_per_batch, tracker.train_loss_per_batch)                   
              plt.figure(18); plt.savefig(s_path + 'global_loss.png')
              plt.figure(19); plt.savefig(s_path + 'detailed_loss.png')
              plt.figure(25); plt.savefig(s_path + 'global_loss_LOG.png')
                                 
              plot_depth = 8
              # output_train = output_train.cpu().data.numpy()            
              # output_train = np.moveaxis(output_train, 1, -1)              
              # seg_train = np.argmax(output_train[0], axis=-1)  
              
              # convert back to CPU
              #batch_x = batch_x.cpu().data.numpy() 
              #batch_y = batch_y.cpu().data.numpy() 
              batch_x_val = batch_x_val.cpu().data.numpy()
              
              tracker.iterations = tracker.iterations + 1
              
              plot_trainer_3D_PYTORCH(seg_val, seg_val, batch_x_val[0], batch_x_val[0], batch_y_val[0], batch_y_val[0],
                                       s_path, tracker.iterations, plot_depth=plot_depth)
                                             
             
        """ To save tracker and model (every x iterations) """
        # if tracker.cur_epoch % save_every_num_epochs == 0:           
        #       tracker.iterations += 1

        #       save_name = s_path + 'check_RERUN_' +  str(tracker.iterations)               
        #       torch.save({
        #        'tracker': tracker,

               
        #        }, save_name)
     
                

                
               
              
              
