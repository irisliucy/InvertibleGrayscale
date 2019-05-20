#!/usr/bin/env python
'''
    File name: config.py
    Author: Iris Liu
    Date created: May 16, 2019
    Python Version: 3.5
'''
import os

# Directories & Paths
DIR_TO_TRAIN_SET = '/home/chuiyiliu3/srv/VOCdevkit/VOC2012/train_imgs'
DIR_TO_TEST_SET = '/home/chuiyiliu3/srv/VOCdevkit/VOC2012/test_imgs'
CURRENT_DIR = '/home/chuiyiliu3/srv/InvertibleGrayscale'
# DIR_TO_TRAIN_SET = '/home/chuiyiliu3/srv/VOCdevkit/VOC2012/train_imgs'
# DIR_TO_TEST_SET = '/home/chuiyiliu3/srv/VOCdevkit/VOC2012/test_imgs'
# CURRENT_DIR = '/home/chuiyiliu3/srv/InvertibleGrayscale'

# GPU settings
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
