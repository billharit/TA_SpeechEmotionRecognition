import os
import pandas as pd


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
