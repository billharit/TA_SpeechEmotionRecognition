import os
import pandas as pd
import numpy as np
import librosa
import math
import tensorflow as tf
from dataset import load_to_dataframe, turn_into_data_for_model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
train_df, test_df = load_to_dataframe('dataset/train', 'dataset/test')

# train_data_value, train_data_target, test_data_value, test_data_target = turn_into_data_for_model(
#     train_df, test_df)

# np.save("train_data_value", train_data_value)
# np.save("train_data_target", train_data_target)
# np.save("test_data_value", test_data_value)
# np.save("test_data_target", test_data_target)

train_data_value = np.load('train_data_value.npy')
train_data_target = np.load('train_data_target.npy')
test_data_value = np.load('test_data_value.npy')
test_data_target = np.load('test_data_target.npy')
print(train_data_value.shape)
print(test_data_value.shape)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs')
tensorboard_callback_snn = tf.keras.callbacks.TensorBoard(
    log_dir='./logs/modelSNN')


# model4 = tf.keras.Sequential()

# # 1st conv layer
# model4.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(
#     train_data_value.shape[1], train_data_value.shape[2], 1)))
# model4.add(tf.keras.layers.MaxPooling2D(
#     (3, 3), strides=(2, 2), padding='same'))
# model4.add(tf.keras.layers.BatchNormalization())

# # 2nd conv layer
# model4.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
# model4.add(tf.keras.layers.MaxPooling2D(
#     (3, 3), strides=(2, 2), padding='same'))
# model4.add(tf.keras.layers.BatchNormalization())

# # 3rd conv layer
# model4.add(tf.keras.layers.Conv2D(32, (2, 2), activation='relu'))


# # Flatten
# model4.add(tf.keras.layers.Flatten())

# # Reshape
# # model4.add(tf.keras.layers.Reshape((129, 13)))
# # model4.add(tf.keras.layers.Reshape((126, 32*32)))
# # model4.add(tf.keras.layers.Reshape((1312, 1)))

# model4lstm = tf.keras.Sequential()
# # LSTM
# model4lstm.add(tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(
#     None, 1184)))
# model4lstm.add(tf.keras.layers.LSTM(64))

# # Final Classifier
# model4lstm.add(tf.keras.layers.Dense(6, activation='softmax'))

# model4combined = tf.keras.Sequential()
# model4combined.add(tf.keras.layers.TimeDistributed(model4, input_shape=(
#     None, train_data_value.shape[1], train_data_value.shape[2], 1)))
# model4combined.add(model4lstm)

# optimiser = tf.keras.optimizers.Adam(learning_rate=0.0001)

# model4combined.compile(optimizer=optimiser,
#                        loss='sparse_categorical_crossentropy',
#                        metrics=['accuracy']
#                        )
# history4comb = model4combined.fit(train_data_value, train_data_target, validation_data=(
#     test_data_value, test_data_target), batch_size=32, epochs=10, callbacks=[tensorboard_callback_cnnlstm])


# model4.compile(optimizer=optimiser,
#                loss='sparse_categorical_crossentropy',
#                metrics=['accuracy'])
# # model4.summary()
# history4 = model4.fit(train_data_value, train_data_target, validation_data=(
#     test_data_value, test_data_target), batch_size=32, epochs=10, callbacks=[tensorboard_callback_cnnlstm])
