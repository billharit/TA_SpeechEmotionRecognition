import os
import pandas as pd
import numpy as np
import librosa
import math
import tensorflow as tf
from dataset import load_to_dataframe

train_df, test_df = load_to_dataframe('dataset/train', 'dataset/test')


# Set Variable for MFCC
num_mfcc = 13
n_fft = 2048
hop_length = 512
SAMPLE_RATE = 16000

train_data = {
    "labels": [],
    "mfcc": []
}

# Encode Categories
labels = {'disgust': 0, 'happy': 1, 'sad': 2,
          'neutral': 3, 'fear': 4, 'angry': 5}
train_df_encoded = train_df.replace({'Target': labels}, inplace=False)
test_df_encoded = test_df.replace({'Target': labels}, inplace=False)


# for item, row in train_df.iterrows():
#     train_data['labels'].append(train_df_encoded.iloc[item, 1])
#     signal, sample_rate = librosa.load(
#         train_df_encoded.iloc[item, 0], sr=SAMPLE_RATE)
#     mfcc = librosa.feature.mfcc(
#         y=signal, sr=SAMPLE_RATE, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
#     mfcc = mfcc.T
#     train_data["mfcc"].append(np.asarray(mfcc))
#     if item % 300 == 0:
#         print(str(math.floor(item)))

for item in range(100):
    train_data['labels'].append(train_df_encoded.iloc[item, 1])
    signal, sample_rate = librosa.load(
        train_df_encoded.iloc[item, 0], sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(
        y=signal, sr=SAMPLE_RATE, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    # print(mfcc)
    # print(mfcc.shape)
    mfcc = mfcc.T
    print(mfcc.shape)
    # print(mfcc)
    train_data["mfcc"].append(np.asarray(mfcc))
    if item % 300 == 0:
        print(str(math.floor(item)))

X = np.asarray(train_data['mfcc'])
y = np.asarray(train_data["labels"])
# X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=156)
print(X.shape)
