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

""" Libraries to load """


import matplotlib.pyplot as plt
import numpy as np
import glob, os
import datetime
import time
from sklearn.model_selection import train_test_split


import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from natsort import natsort_keygen, ns
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order


from PYTORCH_dataloader import *
from functional.plot_functions_CLEANED import *
from functional.data_functions_CLEANED import *
from functional.data_functions_3D import *
from functional.tracker import *
from functional.IO_func import *


from layers.UNet_pytorch_online_2D import *
from layers.unet_nested import *
from layers.unet3_3D import *
from layers.switchable_BN import *

from losses_pytorch.HD_loss import *
import tifffile as tiff
from UNet_functions_PYTORCH import *
   
 
import re
import sps
""" optional dataviewer if you want to load it """
# import napari
# with napari.gui_qt():
#     viewer = napari.view_image(seg_val)

torch.backends.cudnn.benchmark = True  ### set these options to improve speed
torch.backends.cudnn.enabled = True 

if __name__ == '__main__':
        
    """ Define GPU to use """
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    
    """" path to checkpoints """       
    resize_z = 0
    skeletonize = 0
    HISTORICAL = 0;
    s_path = './(1) optic_nerve_5x5_medium/'; HD = 0; alpha = 0; nested = 0; b_norm = 1; sps_bool = 0; sp_weight_bool = 0;
    
    #s_path = './(2) optic_nerve_5x5_medium_sps_bool_NO_bnorm/'; HD = 0; alpha = 0; b_norm = False; sps_bool = 1; sp_weight_bool = 0;
    
    
    """ path to input data """
    input_path = '/media/user/storage/Data/(4) Optic nerve project/Optic Nerve/Training Data Full/'; 
    
    ### added validation of full image!!!
    # val_path = '/media/user/storage/Data/(1z) paranode_identification/InternodeAnalysis_Round2/shortened_for_training_CORRECTED/paranodes for RL/val_image/'
    # images_val = glob.glob(os.path.join(val_path,'*_input.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    # images_val.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting   
    # examples_val = [dict(input=i,truth=i.replace('_input.tif','_seeds.tif')) for i in images_val]
    
    """ Load filenames from tiff """
    images = glob.glob(os.path.join(input_path,'*_input.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    examples = [dict(input=i,truth=i.replace('_input.tif','_truth.tif')) for i in images]
    counter = list(range(len(examples)))
    

    # ### REMOVE IMAGE 1 from training data
    # idx_skip = []
    # for idx, im in enumerate(examples):
    #     filename = im['input']
    #     if '491_' in filename:
    #         print('skip')
    #         idx_skip.append(idx)

    # ### USE THE EXCLUDED IMAGE AS VALIDATION/TESTING
    # examples_test = examples[idx_skip[0]:idx_skip[-1]]

    #examples = [i for j, i in enumerate(examples) if j not in idx_skip]
          
            
    #counter = list(range(len(examples)))
    # counter_val = list(range(len(examples_test)))  ### NEWLY ADDED!!!
    
    
    
    
    # """ load mean and std for normalization later """  
    # mean_arr = np.load('./normalize/' + 'mean_VERIFIED.npy')
    # std_arr = np.load('./normalize/' + 'std_VERIFIED.npy')   

    """ load mean and std """  
    mean_arr = load_pkl('', 'mean_arr.pkl')
    std_arr = load_pkl('', 'std_arr.pkl')
                   
    
    num_workers = 2;
 
    save_every_num_epochs = 1; plot_every_num_epochs = 1; validate_every_num_epochs = 1;      
    
    """ TO LOAD OLD CHECKPOINT """
    # Read in file names
    onlyfiles_check = glob.glob(os.path.join(s_path,'check_*'))
    onlyfiles_check.sort(key = natsort_key1)
    
    if not onlyfiles_check:   ### if no old checkpoints found, start new network and tracker
    
        """ Hyper-parameters """
        deep_sup = False
        switch_norm = False
        
        #transforms = initialize_transforms(p=1.0)
        #transforms = initialize_transforms_simple(p=0.5)
        transforms = 0
        batch_size = 4;      
        test_size = 0.1 
        
        in_channels = 1
        

        """ Initialize network """  
        kernel_size = 5
        pad = int((kernel_size - 1)/2)
        
        if not nested:
            unet = UNet_online(in_channels=in_channels, n_classes=2, depth=5, wf=4, kernel_size = kernel_size, padding= int((kernel_size - 1)/2), 
                            batch_norm=b_norm, batch_norm_switchable=switch_norm, up_mode='upsample')
        else:
            unet = NestedUNet(num_classes=2, input_channels=in_channels, deep_sup=deep_sup, padding=pad, batch_norm_switchable=switch_norm)
        #unet = UNet_3Plus(num_classes=2, input_channels=2, kernel_size=kernel_size, padding=pad)

        unet.train()
        unet.to(device)
        print('parameters:', sum(param.numel() for param in unet.parameters()))  
        
        """ Select loss function *** unimportant if using HD loss """
        if not HD:    loss_function = torch.nn.CrossEntropyLoss(reduction='none')
        else:         loss_function = 'Haussdorf'
            

        """ Select optimizer """
        lr = 1e-5; milestones = [100]  # with AdamW slow down
        if not sps_bool:
            optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        else:            
            optimizer = sps.Sps(unet.parameters())
            
            # import sls
            # optimizer = sls.Sls(unet.parameters())
   

        """ Add scheduler """
        if not sps_bool:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
        else:
            scheduler = []
            
        """ initialize index of training set and validation set, split using size of test_size """
        idx_train, idx_valid, empty, empty = train_test_split(counter, counter, test_size=test_size, random_state=2018)
        
        """ initialize training_tracker """
        # idx_valid = counter_val
        # idx_train = counter
        
        tracker = tracker(batch_size, test_size, mean_arr, std_arr, idx_train, idx_valid, deep_sup=deep_sup, switch_norm=switch_norm, alpha=alpha, HD=HD,
                                          sp_weight_bool=sp_weight_bool, transforms=transforms, dataset=input_path)

        tracker.resize_z = resize_z

    else:             
        """ Find last checkpoint """       
        last_file = onlyfiles_check[-1]
        split = last_file.split('check_')[-1]
        num_check = split.split('.')
        checkpoint = num_check[0]
        checkpoint = 'check_' + checkpoint

        print('restoring weights from: ' + checkpoint)
        check = torch.load(s_path + checkpoint, map_location=lambda storage, loc: storage)
        #check = torch.load(s_path + checkpoint, map_location='cpu')
        #check = torch.load(s_path + checkpoint, map_location=device)
        
        tracker = check['tracker']
        
        unet = check['model_type']
        optimizer = check['optimizer_type']
        scheduler = check['scheduler_type']
        unet.load_state_dict(check['model_state_dict'])
        unet.to(device)
        
        if not sps_bool:
            optimizer.load_state_dict(check['optimizer_state_dict'])
            scheduler.load_state_dict(check['scheduler'])    
        else:
            optimizer = sps.Sps(unet.parameters())
        loss_function = check['loss_function']

        print('parameters:', sum(param.numel() for param in unet.parameters()))  
        
        """ Clean up checkpoint file """
        del check
        torch.cuda.empty_cache()


    #transforms = initialize_transforms_simple(p=0.5)

    """ Create datasets for dataloader """
    #depth = 64  ### redefine for neuron dataset
    
    training_set = Dataset_tiffs(tracker.idx_train, examples, tracker.mean_arr, tracker.std_arr,
                                           sp_weight_bool=tracker.sp_weight_bool, transforms = tracker.transforms)
    val_set = Dataset_tiffs(tracker.idx_valid, examples, tracker.mean_arr, tracker.std_arr,
                                      sp_weight_bool=tracker.sp_weight_bool, transforms = 0)
    
    """ Create training and validation generators"""
    val_generator = data.DataLoader(val_set, batch_size=tracker.batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=True, drop_last = True)

    training_generator = data.DataLoader(training_set, batch_size=tracker.batch_size, shuffle=True, num_workers=num_workers,
                      pin_memory=True, drop_last=True)
         
    print('Total # training images per epoch: ' + str(len(training_set)))
    print('Total # validation images: ' + str(len(val_set)))
    

    """ Epoch info """
    train_steps_per_epoch = len(tracker.idx_train)/tracker.batch_size
    validation_size = len(tracker.idx_valid)
    epoch_size = len(tracker.idx_train)    
   

    print('Total # training images per epoch: ' + str(len(training_set)))
    print('Total # validation images: ' + str(len(val_set)))
    
    """ Find LR, to do so must: (1) have no workers, (2) uncomment section in dataloader """  
    # from torch_lr_finder import LRFinder

    # model = unet
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-7, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    # lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    # lr_finder.range_test(training_generator, end_lr=100, num_iter=100)
    # lr_finder.plot() # to inspect the loss-learning rate graph
    # #lr_finder.reset() # to reset the model and optimizer to their initial state

    
    
    """ Start training """
    for cur_epoch in range(len(tracker.train_loss_per_epoch), 10000): 
         unet.train        
         loss_train = 0
         jacc_train = 0   
         
         

         """ check and plot params during training """             
         for param_group in optimizer.param_groups:
               #param_group['lr'] = 1e-5   # manually sets learning rate
               if not sps_bool:
                   cur_lr = param_group['lr']
               else: cur_lr = 0
               tracker.lr_plot.append(cur_lr)
               tracker.print_essential()

         unet.train()  ### set PYTORCH to training mode

         iter_cur_epoch = 0;     starter = 0    
         loss_train = 0; jacc_train = 0; ce_train = 0; dc_train = 0; hd_train = 0;
         iter_cur_epoch = 0; starter = 0;
         for batch_x, batch_y, spatial_weight in training_generator:
                starter += 1
                if starter == 1:
                    start = time.perf_counter()
                if starter == 50:
                    stop = time.perf_counter(); diff = stop - start; print(diff)

                """ Plot for debug """    
                # np_inputs = np.asarray(batch_x.numpy()[0], dtype=np.uint8)
                # np_labels = np.asarray(batch_y.numpy()[0], dtype=np.uint8)
                # np_labels[np_labels > 0] = 255
                
                # imsave(s_path + str(tracker.iterations) + '_input.tif', np_inputs)
                # imsave(s_path + str(tracker.iterations) + '_label.tif', np_labels)
                
                # in_max = plot_max(np_inputs, plot=0)
                # lb_max = plot_max(np_labels, plot=0)
                
                # imsave(s_path + str(tracker.iterations) + '_max_input.tif', in_max)
                # imsave(s_path + str(tracker.iterations) + '_max_label.tif', lb_max)                
                
                # tracker.iterations = tracker.iterations + 1    
                
                
                """ Load data ==> shape is (batch_size, num_channels, depth, height, width)
                     (1) converts to Tensor
                     (2) normalizes + appl other transforms on GPU
                     (3) ***add non-blocking???
                     ***INPUT LABELS MUST BE < 255??? or else get CudNN error
                     
                """
                inputs, labels = transfer_to_GPU(batch_x, batch_y, device, mean_arr, std_arr)

   
                """ zero the parameter gradients"""
                optimizer.zero_grad()       
                
                """ forward + backward + optimize """
                output_train = unet(inputs)
                
                
                if tracker.HD == 1:
                     loss, tracker, ce_train, dc_train, hd_train = compute_HD_loss(output_train, labels, tracker.alpha, tracker, 
                                                                                   ce_train, dc_train, hd_train, val_bool=0)
                
                else:
                    loss = loss_function(output_train, labels)
                    

                    if torch.is_tensor(spatial_weight):
                        """ Generate BALANCED spatial weighting, taking into account 2 classes """
                        background_weights = labels.cpu().data.numpy()
                        background_weights[background_weights > 0] = -1
                        background_weights[background_weights == 0] = 1
                        background_weights[background_weights == -1] = 0
                        background_weights = np.expand_dims(background_weights, axis=0)
                        background_weights = torch.tensor(background_weights, dtype = torch.float, device=device, requires_grad=False)   
                        
                        spatial_weight = spatial_weight.unsqueeze(0)
                        spatial_tensor = torch.tensor(spatial_weight, dtype = torch.float, device=device, requires_grad=False)         
                        
                        
                        true_spatial_tensor = torch.cat((background_weights, spatial_tensor))
    
                        mean_sW = torch.mean(true_spatial_tensor, dim=0)

                          
                        weighted = torch.multiply(loss, mean_sW)
                        loss = torch.mean(weighted)
                    else:
                             loss = torch.mean(loss)   
                             #loss
                
                
                loss.backward()
                if sps_bool:
                     optimizer.step(loss=loss)
                else:
                     optimizer.step()
               
                """ Training loss """
                tracker.train_loss_per_batch.append(loss.cpu().data.numpy());  # Training loss
                loss_train += loss.cpu().data.numpy()
                
   
                """ Calculate Jaccard on GPU """                 
                jacc = jacc_eval_GPU_torch(output_train, labels)
                jacc = jacc.cpu().data.numpy()
                                            
                jacc_train += jacc # Training jacc
                tracker.train_jacc_per_batch.append(jacc)
   
                tracker.iterations = tracker.iterations + 1       
                iter_cur_epoch += 1
                if tracker.iterations % 100 == 0:
                     print('Trained: %d' %(tracker.iterations))

         tracker.train_loss_per_epoch.append(loss_train/iter_cur_epoch)
         tracker.train_jacc_per_epoch.append(jacc_train/iter_cur_epoch)                       
    
         """ Should I keep track of loss on every single sample? and iteration? Just not plot it??? """   
  
         precision_vals = []; sensitivity_vals = []; val_idx = 0;
         loss_val = 0; jacc_val = 0; val_i = 0;
         iter_cur_epoch = 0;  ce_val = 0; dc_val = 0; hd_val = 0;

         if cur_epoch % validate_every_num_epochs == 0:
             
              with torch.set_grad_enabled(False):  # saves GPU RAM
                  unet.eval()
                  for batch_x_val, batch_y_val, spatial_weight in val_generator:
                        
                        """ Transfer to GPU to normalize ect... """
                        inputs_val, labels_val = transfer_to_GPU(batch_x_val, batch_y_val, device, mean_arr, std_arr)
             

                        
                        
                        output_val = unet(inputs_val)
                        if tracker.HD == 1:
                            loss, tracker, ce_val, dc_val, hd_val = compute_HD_loss(output_val, labels_val, tracker.alpha, tracker, 
                                                                                          ce_val, dc_val, hd_val, val_bool=1)
                        else:    
                            # forward pass to check validation
                            
                            loss = loss_function(output_val, labels_val)



                            if torch.is_tensor(spatial_weight):
                                """ Generate BALANCED spatial weighting, taking into account 2 classes """
                                background_weights = labels_val.cpu().data.numpy()
                                background_weights[background_weights > 0] = -1
                                background_weights[background_weights == 0] = 1
                                background_weights[background_weights == -1] = 0
                                background_weights = np.expand_dims(background_weights, axis=0)
                                background_weights = torch.tensor(background_weights, dtype = torch.float, device=device, requires_grad=False)   
                                
                                spatial_weight = spatial_weight.unsqueeze(0)
                                spatial_tensor = torch.tensor(spatial_weight, dtype = torch.float, device=device, requires_grad=False)         
                                
                                
                                true_spatial_tensor = torch.cat((background_weights, spatial_tensor))
            
                                mean_sW = torch.mean(true_spatial_tensor, dim=0)                                  
                                weighted = torch.multiply(loss, mean_sW)
                                loss = torch.mean(weighted)
                            else:
                                     loss = torch.mean(loss)   
                                     #loss


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
                    
                        
                        
                        for id_val, im_val in enumerate(output_val):
                            seg_val = np.argmax(im_val, axis=-1)
                            
                            
                            TP, FN, FP = find_TP_FP_FN_from_im(seg_val, batch_y_val[id_val])
                                           
                            if TP + FN == 0: TP;   ### Excludes images with NOTHING in ground truth
                            else: sensitivity = TP/(TP + FN); sensitivity_vals.append(sensitivity);    # PPV
                                           
                            if TP + FP == 0: TP;  ### Excludes images with no match TP and also no FP (so still could be missing something from ground truth not detected FN)
                            else: precision = TP/(TP + FP);  precision_vals.append(precision)    # precision             

                    
                            """ Plot for debug """    
                            # imsave(s_path + str(val_i) + '_max_out.tif', np.asarray(seg_val * 255, dtype=np.uint8))
    
                            # np_inputs = np.asarray(batch_x_val.numpy()[id_val], dtype=np.uint8)
                            # np_labels = np.asarray(batch_y_val[id_val], dtype=np.uint8)
                            # np_labels[np_labels > 0] = 255
                            
                            # imsave(s_path + str(val_i) + '_max_input.tif', np_inputs)
                            # imsave(s_path + str(val_i) + '_max_label.tif', np_labels)       
                            # val_i += 1



                  
                        val_idx = val_idx + tracker.batch_size
                        print('Validation: ' + str(val_idx) + ' of total: ' + str(validation_size))
                
                        iter_cur_epoch += 1
                

              tracker.val_loss_per_eval.append(loss_val/iter_cur_epoch)
              tracker.val_jacc_per_eval.append(jacc_val/iter_cur_epoch)   
                   
              tracker.plot_prec.append(np.mean(precision_vals))
              tracker.plot_sens.append(np.mean(sensitivity_vals))

  
              """ Add to scheduler to do LR decay """
              if not sps_bool:
                scheduler.step()
            

              """ calculate new alpha for next epoch """   
              if tracker.HD == 1:
                tracker.alpha = alpha_step(ce_train, dc_train, hd_train, iter_cur_epoch)
                  
         if cur_epoch % plot_every_num_epochs == 0:       
              
             
              """ Plot metrics in tracker """
              plot_tracker(tracker, s_path)
                          
              """ Plot sens + precision + jaccard + loss """
              plot_metric_fun(tracker.plot_sens, tracker.plot_sens_val, class_name='', metric_name='sensitivity', plot_num=30)
              plt.figure(30); plt.savefig(s_path + 'Sensitivity.png')
                    
              plot_metric_fun(tracker.plot_prec, tracker.plot_prec_val, class_name='', metric_name='precision', plot_num=31)
              plt.figure(31); plt.savefig(s_path + 'Precision.png')
           
            
              # """ Plot for entire VOLUME """
              # plot_metric_fun(tracker.plot_sens_val_vol, tracker.plot_sens_val_vol, class_name='', metric_name='sensitivity', plot_num=40)
              # plt.figure(40); plt.savefig(s_path + 'FULL_VAL_VOL_Sensitivity.png')
                    
              # plot_metric_fun(tracker.plot_prec_val_vol, tracker.plot_prec_val_vol, class_name='', metric_name='precision', plot_num=41)
              # plt.figure(41); plt.savefig(s_path + 'FULL_VAL_VOL_Precision.png')
           


              # plot_metric_fun(tracker.train_jacc_per_epoch, tracker.val_jacc_per_eval, class_name='', metric_name='jaccard', plot_num=32)
              # plt.figure(32); plt.savefig(s_path + 'Jaccard.png')
                   
                 
              # plot_metric_fun(tracker.train_loss_per_epoch, tracker.val_loss_per_eval, class_name='', metric_name='loss', plot_num=33)
              # plt.figure(33); plt.yscale('log'); plt.savefig(s_path + 'loss_per_epoch.png')          
                   
              
              # plot_metric_fun(tracker.lr_plot, [], class_name='', metric_name='learning rate', plot_num=35)
              # plt.figure(35); plt.savefig(s_path + 'lr_per_epoch.png') 
     

              # """ Plot metrics per batch """                
              # plot_metric_fun(tracker.train_jacc_per_batch, [], class_name='', metric_name='jaccard', plot_num=34)
              # plt.figure(34); plt.savefig(s_path + 'Jaccard_per_batch.png')
                                
              # plot_cost_fun(tracker.train_loss_per_batch, tracker.train_loss_per_batch)                   
              # plt.figure(18); plt.savefig(s_path + 'global_loss.png')
              # plt.figure(19); plt.savefig(s_path + 'detailed_loss.png')
              # plt.figure(25); plt.savefig(s_path + 'global_loss_LOG.png')
                                 
              plot_depth = 8
              output_train = output_train.cpu().data.numpy()            
              output_train = np.moveaxis(output_train, 1, -1)              
              seg_train = np.argmax(output_train[0], axis=-1)  
              
              # convert back to CPU
              batch_x = batch_x.cpu().data.numpy() 
              batch_y = batch_y.cpu().data.numpy() 
              batch_x_val = batch_x_val.cpu().data.numpy()
              
              plot_trainer_2D_PYTORCH(seg_train, seg_val * 255, batch_x[0], batch_x_val[0], batch_y[0], batch_y_val[0],
                                       s_path, tracker.iterations, plot_depth=plot_depth)
                                             
              
         """ To save (every x iterations) """
         if cur_epoch % save_every_num_epochs == 0:                          

               save_name = s_path + 'check_' +  str(tracker.iterations)               
               if not sps_bool:
                    torch.save({
                     'tracker': tracker,
     
                     'model_type': unet,
                     'optimizer_type': optimizer,
                     'scheduler_type': scheduler,
                     
                     'model_state_dict': unet.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(),
                     'loss_function': loss_function,  
                     
                     }, save_name)
  
               else:
                    torch.save({
                     'tracker': tracker,
     
                     'model_type': unet,
                     'optimizer_type': optimizer,
                     'scheduler_type': [],
                     
                     'model_state_dict': unet.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'scheduler': [],
                     'loss_function': loss_function,  
                     
                     }, save_name)           
                
 
                     
                

                
               
              
              
