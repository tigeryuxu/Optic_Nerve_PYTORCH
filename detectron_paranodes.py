#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:32:23 2021

@author: user
"""

""" Installation """
# !pip install pyyaml==5.1
# import torch, torchvision
# print(torch.__version__, torch.cuda.is_available())
# !gcc --version


# import torch
# #assert torch.__version__.startswith("1.7")
# !pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html



""" Basic setup """
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer

# !wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O input.jpg
# im = cv2.imread("./input.jpg")
# cv2_imshow(im)


""" Run pre-trained model """
# cfg = get_cfg()
# # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# predictor = DefaultPredictor(cfg)
# outputs = predictor(im)


# # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
# print(outputs["instances"].pred_classes)
# print(outputs["instances"].pred_boxes)


# # We can use `Visualizer` to draw the predictions on the image.
# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2_imshow(out.get_image()[:, :, ::-1])




""" Run with custom """

# !wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
# !unzip balloon_dataset.zip > /dev/null


from detectron2.structures import BoxMode

""" setup for paranodes """
import glob, os
from natsort import natsort_keygen, ns
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order

data_path = '/media/user/storage/Data/(1) snake seg project/InternodeAnalysis_Round2/shortened_for_training_CORRECTED/paranodes for RL/paranodes for RL_quads_PARANODES_2D/max/'


import tifffile as tiff
def get_balloon_dicts(img_dir):
    # json_file = os.path.join(img_dir, "via_region_data.json")
    # with open(json_file) as f:
    #     imgs_anns = json.load(f)
    images = glob.glob(os.path.join(img_dir,'*_INPUT_max_proj.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    examples = [dict(input=i,truth=i.replace('_INPUT_max_proj.tif','_TRUTH_max_proj.tif')) for i in images]
    #counter = list(range(len(examples)))
    
    
    dataset_dicts = []
    for idx, v in enumerate(examples):
        record = {}
      
        filename = v['input']
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        truth_name = v["truth"]
        #mask = cv2.imread(truth_name, cv2.IMREAD_GRAYSCALE)
        
        mask = tiff.imread(truth_name)
        
    
        
        from skimage import measure
        from pycocotools.mask import encode
        #labels = measure.label(mask)
        cc = measure.regionprops(mask)
        
        objs = []
        
        for node in cc:
            single_mask = np.zeros(np.shape(mask), np.uint8)
            bbox = list(node['bbox'])    ###  (min_row, min_col, max_row, max_col)
            bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
            
            coords = node['coords']
            #box_id = [0, 1, 3, 4]
            #bbox = list(np.asarray(bbox)[box_id])
            
            ### generate mask with only this single object
            #single_mask[coords[:, 0], coords[:, 1], coords[:, 2]] = 255
            single_mask[coords[:, 0], coords[:, 1]] = 255
   
            """ DEBUG PLOT """
            #cv2_imshow(single_mask)
            annos = encode(np.asarray(np.asarray(single_mask), order="F"))
            
            obj = {
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": annos,
                    "category_id": 0
                
                }
            objs.append(obj)
            # for _, anno in annos.items():
            #     assert not anno["region_attributes"]
            #     anno = anno["shape_attributes"]
            #     px = anno["all_points_x"]
            #     py = anno["all_points_y"]
            #     poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            #     poly = [p for x in poly for p in x]
    
            #     obj = {
            #         "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
            #         "bbox_mode": BoxMode.XYXY_ABS,
            #         "segmentation": [poly],
            #         "category_id": 0,
            #     }
            #     objs.append(obj)
            #print(len(coords))
            
        """ Also save if empty """
        if len(cc) == 0:
            record["annotations"] = []           
            
        else:
            record["annotations"] = objs
            #print(idx)
        
        
        dataset_dicts.append(record)
        
        
    return dataset_dicts




for d in ["train", "valid"]:
    DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts(data_path + d))
    MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
balloon_metadata = MetadataCatalog.get("balloon_train")



""" Fine tune the model on new data """



cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("balloon_train",)


### NEED TO SET THIS FOR BINARY MASK TRUTH DATA
cfg.INPUT.MASK_FORMAT = 'bitmask'

#cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = 0


#cfg = get_cfg()
#cfg.DATASETS.TEST = ("your-validation-set",) 
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

""" Visualize loaded data with randomly selected samples """
dataset_dicts = get_balloon_dicts(data_path + "/train")
for d in random.sample(dataset_dicts, 100):
#for d in dataset_dicts: 
    img = cv2.imread(d["file_name"])
    cv2_imshow(img)
    
    visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2_imshow(out.get_image()[:, :, ::-1])



os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#trainer = DefaultTrainer(cfg) 
trainer = MyTrainer(cfg)
trainer.build_evaluator(cfg, "balloon_valid", output_folder=None)
trainer.build_hooks()


trainer.resume_or_load(resume=False)
trainer.train()




# Look at training curves in tensorboard:
# %load_ext tensorboard
# %tensorboard --logdir=output/ --port=8083

### GO TO:
#http://localhost:6006/



""" Inference and evaluation """
# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0009999.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

### RANDOMLY SELECT SAMPLES TO TEST
from detectron2.utils.visualizer import ColorMode
#dataset_dicts = get_balloon_dicts(data_path + "/valid")

dataset_dicts = get_balloon_dicts('/media/user/storage/Data/(1) snake seg project/Traces files/just first 2/just first 2_quads_PARANODES_2D/max')

print('done loading data')

#for d in random.sample(dataset_dicts, 300): 
for d in dataset_dicts: 
    
    im = cv2.imread(d["file_name"])
    cv2_imshow(im)
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=balloon_metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_imshow(out.get_image()[:, :, ::-1])






