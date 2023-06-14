import librosa
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier


def vggnet(optimizer='adam', learning_rate=0.0001, code=""):
    model = tf.keras.Sequential()
    base_model = tf.keras.applications.vgg16.VGG16(
        include_top=False, weights=None, input_shape=(train_data_value.shape[1], train_data_value.shape[2], 1))
    model.add(base_model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024, activation="relu"))
    model.add(tf.keras.layers.Dense(512, activation="relu"))
    model.add(tf.keras.layers.Dense(512, activation="relu"))
    # model.add(tf.keras.layers.Dropout(0.2))
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
        'logbuatvgg16/vggFC_{0}_{1}_{2}.csv'.format(code, optimizer, str_learning_rate))
    history = model.fit(train_data_value, train_data_target, validation_data=(
        test_data_value, test_data_target), batch_size=32, epochs=60, callbacks=[csv_logger, early_stop])
    return model


def incepnet(optimizer='adam', learning_rate=0.0001, code=""):
    # input_shape = (
    #     train_data_value.shape[1], train_data_value.shape[2], 1)
    # input_tensor = tf.keras.Input(
    #     shape=(train_data_value.shape[1], train_data_value.shape[2], 1))

    # inputs = tf.keras.Input(shape=input_shape)
    # reshaped_inputs = tf.keras.layers.Reshape(
    #     (train_data_value.shape[1], train_data_value.shape[2], 3))(inputs)

    model = tf.keras.Sequential()
    base_model = tf.keras.applications.inception_v3.InceptionV3(
        include_top=False, weights=None, input_shape=(train_data_value.shape[1], train_data_value.shape[2], 3))
    model.add(base_model)
    # model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
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
        'logbuatinception/inception_{0}_{1}_{2}.csv'.format(code, optimizer, str_learning_rate))
    history = model.fit(train_data_value, train_data_target, validation_data=(
        test_data_value, test_data_target), batch_size=32, epochs=60, callbacks=[csv_logger, early_stop])
    return model

# train_data_value = np.load('saved_dataset/crema_d_f32_train_data_value.npy')
# train_data_target = np.load('saved_dataset/crema_d_f32_train_data_target.npy')
# test_data_value = np.load(
#     'saved_dataset/crema_d_f32_test_data_value.npy')
# test_data_target = np.load('saved_dataset/crema_d_f32_test_data_target.npy')


optimizer_list = ['rmsprop']
learning_rate_list = [0.1, 0.01, 0.001, 0.0001]

# train_data_value = np.load('saved_dataset/crema_d_f32_train_data_value.npy')
# train_data_target = np.load('saved_dataset/crema_d_f32_train_data_target.npy')
# test_data_value = np.load(
#     'saved_dataset/crema_d_f32_test_data_value.npy')
# test_data_target = np.load('saved_dataset/crema_d_f32_test_data_target.npy')

# for x in optimizer_list:
#     for y in learning_rate_list:
#         incepnet(x,y, "Crema-D")

train_data_value = np.load('saved_dataset/saveeFix/standardSR_savee_data.npy')
train_data_target = np.load(
    'saved_dataset/saveeFix/standardSR_savee_data_target.npy')
test_data_value = np.load('saved_dataset/saveeFix/standardSR_savee_test.npy')
test_data_target = np.load(
    'saved_dataset/saveeFix/standardSR_savee_test_target.npy')
# # train_data_value = np.expand_dims(train_data_value, axis=2)
# # test_data_value = np.expand_dims(test_data_value, axis=2)
# print(train_data_value.shape)
vggnet("adam", 0.0001, "Savee")

# # vggnet('rmsprop', 0.0001, "Savee")
# for x in optimizer_list:
#     for y in learning_rate_list:
#         # incepnet(x, y, "Savee")
#         vggnet(x, y, "Savee")

# train_data_value = np.load(
#     'saved_dataset/ravdessFix/ravdess_data.npy')
# train_data_target = np.load(
#     'saved_dataset/ravdessFix/ravdess_data_target.npy')
# test_data_value = np.load('saved_dataset/ravdess_test.npy')
# test_data_target = np.load('saved_dataset/ravdess_test_target.npy')
# vggnet(x, y, "Ravdess")
# for x in optimizer_list:
#     for y in learning_rate_list:
#         # incepnet(x, y, "Ravdess")
#         vggnet(x, y, "Ravdess")

# train_data_value = np.load('saved_dataset/crema_d_f32_train_data_value.npy')
# train_data_target = np.load('saved_dataset/crema_d_f32_train_data_target.npy')
# test_data_value = np.load(
#     'saved_dataset/crema_d_f32_test_data_value.npy')
# test_data_target = np.load('saved_dataset/crema_d_f32_test_data_target.npy')
# vggnet("adam", 0.0001, "Crema-D")
# for x in optimizer_list:
#     for y in learning_rate_list:
#         # incepnet(x,y, "Crema-D")
#         vggnet(x, y, "Crema-D")
