import os
import pandas as pd
import numpy as np
import librosa
import math
import tensorflow as tf


def load_to_dataframe(train_folder_path, test_folder_path):
    """
    Loads train and test data from the specified file folders, and returns them as pandas DataFrame objects.
    :param train_folder_path: The path to the folder containing the train data files.
    :type train_folder_path: str
    :param test_folder_path: The path to the folder containing the test data files.
    :type test_folder_path: str
    :return: A two pandas DataFrame objects, containing the train and test data, respectively.
    :rtype: pandas.core.frame.DataFrame, pandas.core.frame.DataFrame
    """
    train_path = 'dataset/train'
    test_path = 'dataset/test'
    train_dir_list = os.listdir(train_path)
    test_dir_list = os.listdir(test_path)

    train_sentiment_value = []
    test_sentiment_value = []
    train_file_path = []
    test_file_path = []

    for file in train_dir_list:
        train_file_path.append(train_path + '/' + file)
        sentiment_code = file.split('_')
        if sentiment_code[2] == 'ANG':
            train_sentiment_value.append('angry')
        elif sentiment_code[2] == 'DIS':
            train_sentiment_value.append('disgust')
        elif sentiment_code[2] == 'FEA':
            train_sentiment_value.append('fear')
        elif sentiment_code[2] == 'HAP':
            train_sentiment_value.append('happy')
        elif sentiment_code[2] == 'NEU':
            train_sentiment_value.append('neutral')
        elif sentiment_code[2] == 'SAD':
            train_sentiment_value.append('sad')
        else:
            train_sentiment_value.append('unknown')

    for file in test_dir_list:
        test_file_path.append(test_path + '/' + file)
        sentiment_code = file.split('_')
        if sentiment_code[2] == 'ANG':
            test_sentiment_value.append('angry')
        elif sentiment_code[2] == 'DIS':
            test_sentiment_value.append('disgust')
        elif sentiment_code[2] == 'FEA':
            test_sentiment_value.append('fear')
        elif sentiment_code[2] == 'HAP':
            test_sentiment_value.append('happy')
        elif sentiment_code[2] == 'NEU':
            test_sentiment_value.append('neutral')
        elif sentiment_code[2] == 'SAD':
            test_sentiment_value.append('sad')
        else:
            test_sentiment_value.append('unknown')

    train_sentiment_df = pd.DataFrame(
        {"File_Path": train_file_path, "Target": train_sentiment_value})

    test_sentiment_df = pd.DataFrame(
        {"File_Path": test_file_path, "Target": test_sentiment_value})

    return train_sentiment_df, test_sentiment_df


def get_sample_rate(audio_df):
    sample_rate_dataframe = pd.DataFrame(columns=["File_Name", "Sample_Rate"])

    for item, row in audio_df.iterrows():
        file_name = row["File_Path"]
        # print(audio_df.iloc[item, 0])
        # signal, sample_rate = librosa.load(file_name, sr=None)
        sample_rate = librosa.get_samplerate(file_name)
        duration = librosa.get_duration(path=file_name, sr=sample_rate)
        # Put all File Name and Sample Rate into one DataFrame (sample_rate_dataframe)
        new_row = {"File_Name": file_name,
                   "Sample_Rate": sample_rate, "Duration": duration}
        sample_rate_dataframe = pd.concat(
            [sample_rate_dataframe, pd.DataFrame(new_row, index=[0])])

    sample_rate_dataframe = sample_rate_dataframe.reset_index(drop=True)

    return sample_rate_dataframe


def turn_into_data_for_model(train_df, test_df, mfcc_number, fft, hop):
    # Set Variable for MFCC
    num_mfcc = mfcc_number
    SAMPLE_RATE = 16000
    n_fft = fft
    hop_length = hop

    train_data = {
        "labels": [],
        "mfcc": []
    }

    test_data = {
        "labels": [],
        "mfcc": []
    }

    # Encode Categories
    labels = {'disgust': 0, 'happy': 1, 'sad': 2,
              'neutral': 3, 'fear': 4, 'angry': 5}
    train_df_encoded = train_df.replace({'Target': labels}, inplace=False)
    test_df_encoded = test_df.replace({'Target': labels}, inplace=False)

    for item, row in train_df.iterrows():
        train_data['labels'].append(train_df_encoded.iloc[item, 1])
        signal, sample_rate = librosa.load(
            train_df_encoded.iloc[item, 0], sr=SAMPLE_RATE)
        mfcc = librosa.feature.mfcc(
            y=signal, sr=SAMPLE_RATE, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T
        train_data["mfcc"].append(np.asarray(mfcc))
        if item % 300 == 0:
            print("Train Size:" + str(math.floor(item)))

    for item, row in test_df.iterrows():
        test_data['labels'].append(test_df_encoded.iloc[item, 1])
        signal, sample_rate = librosa.load(
            test_df_encoded.iloc[item, 0], sr=SAMPLE_RATE)
        mfcc = librosa.feature.mfcc(
            y=signal, sr=SAMPLE_RATE, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T
        test_data["mfcc"].append(np.asarray(mfcc))
        if item % 300 == 0:
            print("Test Size:" + str(math.floor(item)))

    train_data_value = np.asarray(train_data['mfcc'])
    train_data_target = np.asarray(train_data["labels"])
    train_data_value = tf.keras.preprocessing.sequence.pad_sequences(
        train_data_value, maxlen=156)
    test_data_value = np.asarray(test_data['mfcc'])
    test_data_target = np.asarray(test_data["labels"])
    test_data_value = tf.keras.preprocessing.sequence.pad_sequences(
        test_data_value, maxlen=156)
    print(train_data_value.shape)
    print(test_data_value.shape)

    return train_data_value, train_data_target, test_data_value, test_data_target
