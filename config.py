import os
import random
import numpy as np
import torch

TRAIN_DIR = "C:/uddaniiii/used-car-image-classification/data/train"
TEST_DIR = "C:/uddaniiii/used-car-image-classification/data/test"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
SEED = 42
NUM_WORKERS = 0

MODEL_NAME = "test"
SAVE_DIR = f"./checkpoints/{MODEL_NAME}"