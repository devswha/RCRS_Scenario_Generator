import os
import random
import shutil
import fileinput
import sys
import shutil
import subprocess
import signal
import time
import datetime

config = open("config.txt", 'r')
lines = config.readlines()

for line in lines:
    ###########################################################
    # Map data constant
    ###########################################################
    if "NUM_OF_CIVILIANS" in line:
        NUM_OF_CIVILIANS = int(line.split()[1])
    if "NUM_OF_FIRES" in line:
        NUM_OF_FIRES = int(line.split()[1])
    if "SEQUENCE_TIME_STEP" in line:
        SEQUENCE_TIME_STEP = int(line.split()[1])
    if "LIMIT_TIME_STEP" in line:
        LIMIT_TIME_STEP = int(line.split()[1])
    if "LIMIT_CIVILIAN_HP" in line:
        LIMIT_CIVILIAN_HP = int(line.split()[1])

    ###########################################################
    # Train data constant
    ###########################################################
    if "TRAIN_START_MAP_NUM" in line:
        TRAIN_START_MAP_NUM = int(line.split()[1])
    if "TRAIN_END_MAP_NUM" in line:
        TRAIN_END_MAP_NUM = int(line.split()[1])
    if "TRAIN_STEP" in line:
        TRAIN_STEP = int(line.split()[1])

    ###########################################################
    # test data constant
    ###########################################################
    if "TEST_START_MAP_NUM" in line:
        TEST_START_MAP_NUM = int(line.split()[1])
    if "TEST_END_MAP_NUM" in line:
        TEST_END_MAP_NUM = int(line.split()[1])

    ###########################################################
    # Directory data constant
    ###########################################################
    if "TRAIN_GENERATED_MAP_DIR" in line:
        TRAIN_GENERATED_MAP_DIR = os.path.abspath(line.split()[1])
    if "TRAIN_GENERATED_IMAGE_DIR" in line:
        TRAIN_GENERATED_IMAGE_DIR = os.path.abspath(line.split()[1])
    if "TEST_GENERATED_MAP_DIR" in line:
        TEST_GENERATED_MAP_DIR = os.path.abspath(line.split()[1])
    if "TEST_GENERATED_IMAGE_DIR" in line:
        TEST_GENERATED_IMAGE_DIR = os.path.abspath(line.split()[1])
    if "RESULTS_DIR" in line:
        RESULTS_DIR = os.path.abspath(line.split()[1])
    if "MODELS_DIR" in line:
        MODELS_DIR = os.path.abspath(line.split()[1])
    if "DATASET_DIR" in line:
        DATASET_DIR = os.path.abspath(line.split()[1])

SIMULATOR_DIR = os.path.abspath('./RCRS/simulator')
BASE_MAP_DIR = os.path.abspath('./RCRS/baseMap')

config.close()
