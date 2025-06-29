import os
import random
import numpy as np
import torch

TRAIN_DIR = "C:/uddaniiii/used-car-image-classification/data/train"
TEST_DIR = "C:/uddaniiii/used-car-image-classification/data/test"

IMG_SIZE = 384
BATCH_SIZE = 16
EPOCHS = 150
LOSS = "ce"
LR = 1e-4
SEED = 42
NUM_WORKERS = 0

MODEL_NAME= "tiny_vit_21m_384"
MODEL_PATH = "tiny_vit_21m_384_randaug_hem"
SAVE_DIR = f"./checkpoints/{MODEL_PATH}"

OUTPUT_CSV = f"./submit/{MODEL_PATH}.csv"
OUTPUT_NPY = f"./npy/{MODEL_PATH}.npy"