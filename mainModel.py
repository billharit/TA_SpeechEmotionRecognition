import librosa
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier

# train_data_value = np.load('saved_dataset/40_2048_512_train_data_value.npy')
# train_data_target = np.load('saved_dataset/40_2048_512_train_data_target.npy')
# test_data_value = np.load('saved_dataset/40_2048_512_test_data_value.npy')
# test_data_target = np.load('saved_dataset/40_2048_512_test_data_target.npy')


def cnn_lstm(optimizer='adam', learning_rate=0.0001, modelName="x"):
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
    model.summary()
    checkpoint_path = "checkpoints/{0}".format(modelName)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max')
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10)
    str_learning_rate = str(learning_rate).replace('.', '')
    csv_logger = tf.keras.callbacks.CSVLogger(
        '{0}{1}_{2}.csv'.format(modelName, optimizer, str_learning_rate))
    history = model.fit(train_data_value, train_data_target, validation_data=(
        test_data_value, test_data_target), batch_size=32, epochs=90, callbacks=[csv_logger, early_stop])
    return model, history


# train_data_value = np.load('crema_d_f32_train_data_value.npy')
# train_data_target = np.load('crema_d_f32_train_data_target.npy')
# test_data_value = np.load('crema_d_f32_test_data_value.npy')
# test_data_target = np.load('crema_d_f32_test_data_target.npy')


# train_data_value = np.load(
#     'saved_dataset/ravdess_dataWithGaussianAugment_train.npy')
# train_data_target = np.load(
#     'saved_dataset/ravdess_dataWithGaussianAugment_target.npy')
# test_data_value = np.load('saved_dataset/ravdess_test.npy')
# test_data_target = np.load('saved_dataset/ravdess_test_target.npy')

# model, history = cnn_lstm(
#     learning_rate=0.0001, modelName="RobustCNN3secondLayerNormalDataH5")
# model.save("2RobustCNN3secondLayerLSTM.h5")

# train_data_value = np.load(
#     'saved_dataset/ravdess_dataWithPitchShift_train.npy')
# train_data_target = np.load(
#     'saved_dataset/ravdess_dataWithPitchShift_target.npy')
# test_data_value = np.load('saved_dataset/ravdess_test.npy')
# test_data_target = np.load('saved_dataset/ravdess_test_target.npy')

# model, history = cnn_lstm(
#     learning_rate=0.0001, modelName="ravdess_with_pitchshift")
# model.save("")

# train_data_value = np.load(
#     'saved_dataset/ravdess_dataWithTimeStretch2_train.npy')
# train_data_target = np.load(
#     'saved_dataset/ravdess_dataWithTimeStretch2_target.npy')
# test_data_value = np.load('saved_dataset/ravdess_test.npy')
# test_data_target = np.load('saved_dataset/ravdess_test_target.npy')
# model, history = cnn_lstm(
#     learning_rate=0.0001, modelName="ravdess_with_timestretch")

# train_data_value = np.load(
#     'saved_dataset/ravdess_data.npy')
# train_data_target = np.load(
#     'saved_dataset/ravdess_data_target.npy')
# test_data_value = np.load('saved_dataset/ravdess_test.npy')
# test_data_target = np.load('saved_dataset/ravdess_test_target.npy')

# model, history = cnn_lstm(
#     learning_rate=0.0001, modelName="ravdess_with_pitchshift")


train_data_value = np.load(
    'saved_dataset/crema_train_gaussian_value_padded.npy')
train_data_target = np.load(
    'saved_dataset/crema_train_gaussian_target.npy')
test_data_value = np.load('saved_dataset/crema_d_f32_test_data_value.npy')
test_data_target = np.load('saved_dataset/crema_d_f32_test_data_target.npy')

model, history = cnn_lstm(
    learning_rate=0.0001, modelName="RCL4crema-d-gaussian")

train_data_value = np.load(
    'saved_dataset/crema_train_pitch_value_padded.npy')
train_data_target = np.load(
    'saved_dataset/crema_train_pitch_target.npy')
test_data_value = np.load('saved_dataset/crema_d_f32_test_data_value.npy')
test_data_target = np.load('saved_dataset/crema_d_f32_test_data_target.npy')

model, history = cnn_lstm(
    learning_rate=0.001, modelName="RCL4crema-d-pitch")

train_data_value = np.load(
    'saved_dataset/crema_train_timeStretch_value_padded.npy')
train_data_target = np.load(
    'saved_dataset/crema_train_timeStretch_target.npy')
test_data_value = np.load('saved_dataset/crema_d_f32_test_data_value.npy')
test_data_target = np.load('saved_dataset/crema_d_f32_test_data_target.npy')

model, history = cnn_lstm(
    learning_rate=0.001, modelName="RCL4crema-d-timeStretch")


train_data_value = np.load(
    'saved_dataset/crema_d_f32_train_data_value.npy')
train_data_target = np.load(
    'saved_dataset/crema_d_f32_train_data_target.npy')
test_data_value = np.load('saved_dataset/crema_d_f32_test_data_value.npy')
test_data_target = np.load('saved_dataset/crema_d_f32_test_data_target.npy')

model, history = cnn_lstm(
    learning_rate=0.001, modelName="RCL4crema-d-")
