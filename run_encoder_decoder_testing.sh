#!/bin/bash

MODEL_VERSION = ''
DIR_TO_TEST_SET_ENCODER = ''
DIR_TO_TEST_SET_DECODER = ''
NOISE_VAL =
RUN_ENCODER =  # Run encoder option: {E: Encoder, D: Decoder, A: both}

### Change config.py
echo "Updating ${oldfile}..."
sed -i 's/MODEL_VERSION/MODEL_VERSION/g' file.txt

### Run python and output log
