#!/usr/bin/env python
'''
    File name: config.py
    Author: Iris Liu
    Date created: May 16, 2019
    Python Version: 3.5
'''
import os

# Directories & Paths
DIR_TO_TRAIN_SET = ''
DIR_TO_TEST_SET = ''
CURRENT_DIR = ''

# GPU settings
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
