import librosa
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier


def resnet_unweighted(optimizer='adam', learning_rate=0.0001, dataset="x"):
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
        monitor='val_loss', patience=10)
    csv_logger = tf.keras.callbacks.CSVLogger(
        'Resnet_{}.csv'.format(dataset))
    history = model.fit(train_data_value, train_data_target, validation_data=(
        test_data_value, test_data_target), batch_size=32, epochs=90, callbacks=[csv_logger, early_stop])
    model.save("Resnet_{}".format(dataset))
    return model


# train_data_value = np.load('saved_dataset/saveeFix/standardSR_savee_data.npy')
# train_data_target = np.load(
#     'saved_dataset/saveeFix/standardSR_savee_data_target.npy')
# test_data_value = np.load('saved_dataset/saveeFix/standardSR_savee_test.npy')
# test_data_target = np.load(
#     'saved_dataset/saveeFix/standardSR_savee_test_target.npy')
# resnet_unweighted("sgd", 0.01, "savee")

# test_data_value = np.load('saved_dataset/ravdessFix/ravdess_test.npy')
# test_data_target = np.load('saved_dataset/ravdessFix/ravdess_test_target.npy')
# train_data_value = np.load(
#     'saved_dataset/ravdessFix/ravdess_data.npy')
# train_data_target = np.load(
#     'saved_dataset/ravdessFix/ravdess_data_target.npy')
# resnet_unweighted("adagrad", 0.01, "ravdess")

train_data_value = np.load(
    'saved_dataset/crema_data_train.npy')
train_data_target = np.load('saved_dataset/crema_data_target.npy')
test_data_value = np.load(
    'saved_dataset/crema_data_test.npy')
test_data_target = np.load('saved_dataset/crema_data_test_target.npy')
resnet_unweighted("adam", 0.001, "crema-d")
