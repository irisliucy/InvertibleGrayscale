#!/usr/bin/env python
'''
    File name: config.py
    Author: Iris Liu
    Date created: May 16, 2019
    Python Version: 3.5
'''
import os

MODEL_VERSION = 'vgg19v0'

# Directories & Paths
DIR_TO_TRAIN_SET = '' # '/home/chuiyiliu3/srv/VOCdevkit/VOC2012/train_imgs'
DIR_TO_VALID_SET = '' # '/home/chuiyiliu3/srv/VOCdevkit/VOC2012/train_imgs'
DIR_TO_TEST_SET = '' # '/home/chuiyiliu3/srv/VOCdevkit/VOC2012/test_imgs'
CURRENT_DIR = '' # '/home/chuiyiliu3/srv/InvertibleGrayscale'
RESULT_STORAGE_DIR = os.path.join('./resultStorage/', MODEL_VERSION)
RESULT_CSV_DIR = os.path.join(RESULT_STORAGE_DIR, 'csv')
SOURCE_EVAL_DIR = '/home/chuiyiliu3/srv/VOCdevkit/VOC2012/test_imgs'
TARGET_EVAL_DIR = '/home/chuiyiliu3/srv/InvertibleGrayscale/resultStorage/vgg19v0/output/test_result_imgs/restored_rgb'

# GPU settings
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# Parameters
IMG_SHAPE = (256, 256)

# Sample Mode for code testing
SAMPLE_TEST_MODE = False
NUMBER_OF_SAMPLES = 20

# Noise
NOISE_MODE = 'M' # M: multiplicative noise, A: additive noise, N: None
NOISE_VAL = 0.0
NOISE_MEAN = .0 # if NOISE_MODE != 'N'
NOISE_STD = .2  # if NOISE_MODE != 'N'

# Training
DEBUG_MODE = True # Run validation

# Evaluation
# if run encoder (RGB -> Grayscale), 3 channel RGB image should be provided in the 'DIR_TO_TEST_SET'
# if run decoder (Grayscale -> RGB), 1 channel invertible grayscale image should be provided in the 'DIR_TO_TEST_SET'
RUN_Encoder = True
