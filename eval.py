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
import cv2

from util import *
from config import *
from model import *
from main import *
import glob

in_channels = 3
batch_size = 4

def write_eval_result_2_csv(color_psnr, psnr):
    exists_or_mkdir(RESULT_CSV_DIR)
    # open a file
    with open(os.path.join(RESULT_CSV_DIR, 'evaluated_result.csv'), 'w') as f:
        fieldnames = ['Color PSNR', 'PSNR']
        file = csv.DictWriter(f, fieldnames = fieldnames)

        file.writeheader() # write header
        file.writerow({'Color PSNR': color_psnr, 'PSNR': psnr})

source_img_batch = []
target_img_batch = []
src_files = []
target_files = []

print('>>>> Evaluation starts')
print('Processing image batches....')
print('Source evaluating directory: {}'.format(SOURCE_EVAL_DIR))
print('Target evaluating directory: {}'.format(TARGET_EVAL_DIR))

eval_list = glob.glob(os.path.join(SOURCE_EVAL_DIR, '*.*'))
source_evaluation_sample = NUMBER_OF_SAMPLES if SAMPLE_TEST_MODE else len(eval_list)

for i, x in enumerate(sorted(eval_list)):
    if i < source_evaluation_sample:
        src_files.append(x)
        source_img = cv2.imread(x)
        source_img_batch.append(source_img)
    # source_img.close()

for x in sorted(glob.glob(os.path.join(TARGET_EVAL_DIR, '*.*'))):
    target_files.append(x)
    target_img = cv2.imread(x)
    target_img_batch.append(target_img)
    # target_img.close()

print('Num of source images: {} \nNum of target images: {}'.format(len(src_files), len(target_files)))
print('Paring Test:', '\n'+src_files[-1], '\n'+target_files[-1])

source_img_batch = np.array(source_img_batch, dtype='float32')
target_img_batch = np.array(target_img_batch, dtype='float32')

print('Number of sample in source batch: {}'.format(len(source_img_batch)))
print('Number of sample in target batch: {}'.format(len(target_img_batch)))

# compute the result
print("Computing PSNR....")
color_psnr = compute_color_psnr(source_img_batch, target_img_batch)
psnr = measure_psnr(source_img_batch, target_img_batch)

print("Color PSNR: {}".format(color_psnr))
print("PSNR: {}".format(psnr))
print('Evaluation Completed!')

# write the result
print('Writing result to csv...')
write_eval_result_2_csv(color_psnr, psnr)
