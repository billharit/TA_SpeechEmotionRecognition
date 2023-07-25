import librosa
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier

train_data_value = np.load('ravdess_train_data_value.npy')
train_data_target = np.load('ravdess_train_data_target.npy')
test_data_value = np.load('ravdess_validation_data_value.npy')
test_data_target = np.load('ravdess_validation_data_target.npy')


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
        'ravdess_resnet_lstm{0}_{1}.csv'.format(optimizer, str_learning_rate))
    history = model.fit(train_data_value, train_data_target, validation_data=(
        test_data_value, test_data_target), batch_size=32, epochs=30, callbacks=[csv_logger, early_stop])
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
        'ravdess_resnet{0}_{1}.csv'.format(optimizer, str_learning_rate))
    history = model.fit(train_data_value, train_data_target, validation_data=(
        test_data_value, test_data_target), batch_size=32, epochs=30, callbacks=[csv_logger, early_stop])
    return model


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
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256)))
    model.add(tf.keras.layers.Dense(6, activation="softmax"))
    optimiser = tf.keras.optimizers.get(optimizer)
    optimiser.learning_rate.assign(learning_rate)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # model.summary()
    str_learning_rate = str(learning_rate).replace('.', '')
    csv_logger = tf.keras.callbacks.CSVLogger(
        'ravdess_robust_cnn_lstm{0}_{1}.csv'.format(optimizer, str_learning_rate))
    history = model.fit(train_data_value, train_data_target, validation_data=(
        test_data_value, test_data_target), batch_size=32, epochs=30, callbacks=[csv_logger, early_stop])
    return model


optimizer_list = ['adam', 'rmsprop', 'sgd', 'adagrad']
learning_rate_list = [0.0001]

for x in optimizer_list:
    for y in learning_rate_list:
        cnn_lstm(x, y)
        resnet_lstm_unweighted(x, y)
        resnet_unweighted(x, y)
