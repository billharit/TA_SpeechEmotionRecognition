import librosa
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier

train_data_value = np.load(
    'saved_dataset/crema_data_train.npy')
train_data_target = np.load('saved_dataset/crema_data_target.npy')
test_data_value = np.load(
    'saved_dataset/crema_data_test.npy')
test_data_target = np.load('saved_dataset/crema_data_test_target.npy')


def cnn_lstm(optimizer='adam', learning_rate=0.0001):
    # def cnn_lstm():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=(
        train_data_value.shape[1], train_data_value.shape[2], 1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('elu'))
    model.add(tf.keras.layers.MaxPooling2D(
        (2, 2), strides=(2, 2), padding='same'))

    model.add(tf.keras.layers.Conv2D(64, (3, 3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('elu'))
    model.add(tf.keras.layers.MaxPooling2D(
        (2, 2), strides=(2, 2), padding='same'))

    model.add(tf.keras.layers.Conv2D(128, (2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('elu'))
    model.add(tf.keras.layers.MaxPooling2D(
        (2, 2), strides=(2, 2), padding='same'))

    model.add(tf.keras.layers.Conv2D(128, (2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('elu'))

    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)))
    model.add(tf.keras.layers.Dense(6, activation="softmax"))
    optimiser = tf.keras.optimizers.get(optimizer)
    optimiser.learning_rate.assign(learning_rate)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # model.summary()
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5)
    str_learning_rate = str(learning_rate).replace('.', '')
    csv_logger = tf.keras.callbacks.CSVLogger(
        'robust_cnn_lstm{0}_{1}.csv'.format(optimizer, str_learning_rate))
    history = model.fit(train_data_value, train_data_target, validation_data=(
        test_data_value, test_data_target), batch_size=32, epochs=90, callbacks=[csv_logger, early_stop])
    return model


def resnet_lstm_unweighted(optimizer='adam', learning_rate=0.0001):
    model = tf.keras.Sequential()
    base_model = tf.keras.applications.resnet_v2.ResNet50V2(
        include_top=False, weights=None, input_shape=(train_data_value.shape[1], train_data_value.shape[2], 1))
    model.add(base_model)
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)))
    model.add(tf.keras.layers.Dense(6, activation="softmax"))
    optimiser = tf.keras.optimizers.get(optimizer)
    optimiser.learning_rate.assign(learning_rate)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    str_learning_rate = str(learning_rate).replace('.', '')
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5)
    csv_logger = tf.keras.callbacks.CSVLogger(
        'resnet_lstm{0}_{1}.csv'.format(optimizer, str_learning_rate))
    history = model.fit(train_data_value, train_data_target, validation_data=(
        test_data_value, test_data_target), batch_size=32, epochs=90, callbacks=[csv_logger, early_stop])
    return model


def resnet_unweighted(optimizer='adam', learning_rate=0.0001):
    model = tf.keras.Sequential()
    base_model = tf.keras.applications.resnet_v2.ResNet50V2(
        include_top=False, weights=None, input_shape=(train_data_value.shape[1], train_data_value.shape[2], 1))
    model.add(base_model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(6, activation="softmax"))
    optimiser = tf.keras.optimizers.get(optimizer)
    optimiser.learning_rate.assign(learning_rate)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    str_learning_rate = str(learning_rate).replace('.', '')
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5)
    csv_logger = tf.keras.callbacks.CSVLogger(
        'resnet{0}_{1}.csv'.format(optimizer, str_learning_rate))
    history = model.fit(train_data_value, train_data_target, validation_data=(
        test_data_value, test_data_target), batch_size=32, epochs=90, callbacks=[csv_logger, early_stop])
    return model


optimizer_list = ['adam', 'rmsprop', 'sgd', 'adagrad']
learning_rate_list = [0.01, 0.001, 0.0001]

for x in optimizer_list:
    for y in learning_rate_list:
        cnn_lstm(x, y)
        resnet_lstm_unweighted(x, y)
        resnet_unweighted(x, y)

# param_grid = {
#     'optimizer': ['adam'],
#     # 'optimizer': ['adam', 'rmsprop', 'sgd', 'adagrad'],
#     'learning_rate': [0.001, 0.01]
#     # 'learning_rate': [0.001, 0.01, 0.1]

# }

# # csv_logger = tf.keras.callbacks.CSVLogger(
# #     'robust_cnn_lstm{0}_{1}.csv'.format(optimizer, str_learning_rate))

# model_estimator = KerasClassifier(
#     model=cnn_lstm, loss="sparse_categorical_crossentropy", metrics=['accuracy'],  learning_rate=0.001, optimizer='adam', epochs=1, batch_size=32)

# grid = GridSearchCV(estimator=model_estimator,
#                     param_grid=param_grid, return_train_score=True)

# # fit the grid search to the training data and log the results
# grid.fit(train_data_value, train_data_target
#          #  ,
#          #  validation_data=(test_data_value, test_data_target)
#          )
# results = grid.cv_results_

# #
# print("Best hyperparameters: ", grid.best_params_)
# print("Mean validation score: ", grid.best_score_)

# # convert the results to a Pandas dataframe and save to a CSV file
# df = pd.DataFrame(results)
# df.to_csv('grid_search_results.csv', index=False)
