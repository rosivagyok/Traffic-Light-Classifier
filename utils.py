import cv2
import random
import numpy as np
from glob import glob
from os.path import join

# normalize feature space
def preprocess(x):
        x = np.float32(x) - np.mean(x)
        x /= x.max()-x.min()
        return x

def init_from_npy(dump_file_path):
        
    # load dataset
    dataset_npy = np.load(dump_file_path)
    initialized = True
    return dataset_npy

def load_batch(dataset_npy, batch_size, augmentation=False):

    X_batch, Y_batch = [], []

    loaded = 0

    # generate a batch of random images according to batch size
    while loaded < batch_size:
        idx = np.random.randint(0, len(dataset_npy))

        # features
        x = dataset_npy[idx][0]

        # labels
        y = dataset_npy[idx][1]

        X_batch.append(x)
        Y_batch.append(y)

        loaded += 1
    
    # preprocess batches
    X_batch = preprocess(X_batch)
    if augmentation:
        X_batch = perform_augmentation(X_batch)

    return X_batch, Y_batch

def perform_augmentation(batch):

    # random data augmentation in image batch
    for b in range(batch.shape[0]):
        if random.choice([True, False]) == True:
            batch[b] = np.fliplr(batch[b])    # flip horizontally (mirroring)
        if random.choice([True, False]) == True:
            batch[b] = np.flipud(batch[b])    # flip vertically
        if random.choice([True, False]) == True:
            batch[b] = np.rot90(batch[b], 3)  # rotate 90 degree clockwise
        if random.choice([True, False]) == True:
            batch[b] = np.rot90(batch[b], 1)  # rotate 90 degree counter-clockwise

    return batch