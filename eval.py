#!/usr/bin/env python
'''
    File name: eval.py
    Description: evaluate the model and write the results to csv
    Author: Iris Liu
    Date created: May 23, 2019
    Python Version: 3.5
'''
import os
import csv
from PIL import Image
import numpy as np

from util import *
from config import *
from model import *
from main import *
import glob

in_channels = 3
batch_size = 4

# exists_or_mkdir(RESULT_CSV_DIR)

# open a file
# with open(os.path.join(RESULT_CSV_DIR, 'evaluated_result.csv')) as f:
#     csv = csv.reader(f)
# src_eval_list = gen_list(SOURCE_EVAL_DIR)
# target_eval_list = gen_list(TARGET_EVAL_DIR)

# source_img_batch, source_label, num_src = input_producer(src_eval_list, in_channels, batch_size, need_shuffle=False)
# target_img_batch, target_label, num_target = input_producer(target_eval_list, in_channels, batch_size, need_shuffle=False)
source_img_batch = []
target_img_batch = []

for x in glob.glob(os.path.join(SOURCE_EVAL_DIR, '*.*')):
    source_img = Image.open(x)
    source_img_batch.append(source_img)

for x in glob.glob(os.path.join(TARGET_EVAL_DIR, '*.*')):
    target_img = Image.open(x)
    target_img_batch.append(target_img)

# source_img_batch = np.array(source_img_batch,dtype='float32')
# target_img_batch = np.array(source_img_batch,dtype='float32')

print(source_img_batch.shape)
print(target_img_batch.shape)

# compute the result
print("Computing PSNR...")
color_psnr = compute_color_psnr(source_img_batch, target_img_batch)
psnr = measure_psnr(source_img_batch, target_img_batch)

print("Color PSNR: {}".format(color_psnr))
print("Color PSNR: {}".format(psnr))

# write the result
