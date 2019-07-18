import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from PIL import Image ## added new
import argparse
# import cv2
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_image_directory", required=True,
	help="Input image directory name")
ap.add_argument("-o", "--output_image_directory", required=True,
	help="Output image directory name")
args = vars(ap.parse_args())

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)

from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

class CowConfig(Config):
    NAME = "cow"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 3 
    STEPS_PER_EPOCH = 125
    DETECTION_MIN_CONFIDENCE = 0.9
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 1024
    TRAIN_ROIS_PER_IMAGE = 200

config = CowConfig()

def get_width(xy):
    width = abs(xy[1] - xy[3])
    return width

def get_height(xy):
    height = abs(xy[0] - xy[2])
    return height

def get_area(xy):
    width = get_width(xy)
    height = get_height(xy)
    area = width * height
    return area

def get_biggest_box(xy_list,class_ids,specific_class_id):
    biggest_area = 0
    box_index_list = []
    for check_index,check_val in enumerate(class_ids):    
        if specific_class_id == int(check_val):
            box_index_list.append(check_index)
    for i, xy in enumerate(xy_list):
        if i in box_index_list:
            area = get_area(xy)
            if area > biggest_area:
                biggest_area = area
                biggest_xy = xy
                ix = i
    return biggest_xy, ix

def crop(image_path, coords, saved_location):
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords) ## (x1, y1, x2,y2 )
    cropped_image.save(saved_location)
    # cropped_image.show()


MODEL_DIR = os.path.join(ROOT_DIR, "logs")
MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_cow_0030.h5")

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(MODEL_PATH, by_name=True)

IMAGE_DIR = os.path.join(ROOT_DIR,'samples','images',args["input_image_directory"])
OUTPUT_DIR = os.path.join(ROOT_DIR,'samples','generated_images',args["output_image_directory"])

images = os.listdir(IMAGE_DIR)
output_images = os.listdir(OUTPUT_DIR)

print(f" Total of {len(images)} images to crop from input image directory ~")
print(f" Approximately of {len(images)*3} images to be cropped to output image directory ~")
print("\n" * 5)
i = 0 

for image_ in images:
    i+=1
    if len(output_images) > 0:
        if any(image_ in s for s in output_images):
            print(f" Skipping image number {i} ")
            continue
    image = skimage.io.imread(os.path.join(IMAGE_DIR,image_))
    if image.shape[-1] == 4:
       image = image[..., :3]
    results = model.detect([image], verbose=0)
    r = results[0]
    print("* "*20)
    print(f" Printing image number {i} ")

    if 1 in r['class_ids']:
        big_box, big_ix = get_biggest_box(r['rois'],r['class_ids'],1)
        y1, x1, y2, x2 = big_box
        crop(os.path.join(IMAGE_DIR,image_), (x1, y1, x2,y2 ), os.path.join(OUTPUT_DIR, image_))
        print(f" Printing sub image number {i}_{1} ")
        # img = image.copy()
        # cropped = image[y1:y2,x1:x2,:]
        # cv2.imshow("cropped", image)
        # cv2.imwrite(os.path.join(OUTPUT_DIR, image_), image)


    if 2 in r['class_ids']:
        big_box, big_ix = get_biggest_box(r['rois'],r['class_ids'],2)
        y1, x1, y2, x2 = big_box
        crop(os.path.join(IMAGE_DIR,image_), (x1, y1, x2,y2 ), os.path.join(OUTPUT_DIR, image_))
    
        print(f" Printing sub image number {i}_{2} ")
        # img = image.copy()
        # cropped = image[y1:y2,x1:x2,:]
        # cv2.imshow("cropped", image)
        # cv2.imwrite(os.path.join(OUTPUT_DIR, image_), image)

    if 3 in r['class_ids']:
        big_box, big_ix = get_biggest_box(r['rois'],r['class_ids'],3)
        y1, x1, y2, x2 = big_box
        crop(os.path.join(IMAGE_DIR,image_), (x1, y1, x2,y2 ), os.path.join(OUTPUT_DIR, image_))
        print(f" Printing sub image number {i}_{3} ")
        # img = image.copy()
        # cropped = image[y1:y2,x1:x2,:]
        # cv2.imshow("cropped", image)
        # cv2.imwrite(os.path.join(OUTPUT_DIR, image_), image)

    print(f" Approximately {(len(os.listdir(IMAGE_DIR))*3)-len(os.listdir(OUTPUT_DIR))} images left to be cropped")
    print("* "*20)
    print("\n" * 2)

print(" <--> Done!! Goodbye!")

    

    





