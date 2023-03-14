import os
import pandas as pd
import numpy as np
import librosa
import math
import tensorflow as tf
from dataset import load_to_dataframe, turn_into_data_for_model, get_sample_rate
from model import train_cnn

# train_df, test_df = load_to_dataframe('dataset/train', 'dataset/test')

train_data_value = np.load(
    './processed_data/40_1024_512train_data_value_default.npy')
train_data_target = np.load(
    './processed_data/40_1024_512train_data_target_default.npy')
test_data_value = np.load(
    './processed_data/40_1024_512test_data_value_default.npy')
test_data_target = np.load(
    './processed_data/40_1024_512test_data_target_default.npy')
