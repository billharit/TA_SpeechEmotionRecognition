import time
import tensorflow as tf
import numpy as np
import math


batch_times = []


class TimingCallback(tf.keras.callbacks.Callback):
    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        batch_end_time = time.time()
        batch_time = batch_end_time - self.batch_start_time
        batch_times.append(batch_time)


def cnn_lstm(optimizer='adam', learning_rate=0.0001, code=""):
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
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # model.summary()
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5)
    str_learning_rate = str(learning_rate).replace('.', '')
    csv_logger = tf.keras.callbacks.CSVLogger(
        'logbuatdocs/robust_cnn_lstm{0}_{1}_{2}.csv'.format(optimizer, str_learning_rate, code))
    history = model.fit(train_data_value, train_data_target, validation_data=(
        test_data_value, test_data_target), batch_size=32, epochs=4, callbacks=[TimingCallback()])
    return model


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
        test_data_value, test_data_target), batch_size=32, epochs=3, callbacks=[TimingCallback()])
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
        'ravdess_resnet_lstm{0}_{1}.csv'.format(optimizer, str_learning_rate))
    history = model.fit(train_data_value, train_data_target, validation_data=(
        test_data_value, test_data_target), batch_size=32, epochs=3, callbacks=[TimingCallback()])
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
        test_data_value, test_data_target), batch_size=32, epochs=3, callbacks=[TimingCallback()])
    return model


def SaveTimes(code, batch_timesx):
    with open('timinglogaug/{}.txt'.format(code), 'w') as file:
        for batch_time in batch_timesx:
            file.write(str(batch_time) + '\n')
    batch_timesx = []
    return batch_timesx


# train_data_value = np.load('saved_dataset/crema_d_f32_train_data_value.npy')
# train_data_target = np.load('saved_dataset/crema_d_f32_train_data_target.npy')
# test_data_value = np.load(
#     'saved_dataset/crema_d_f32_test_data_value.npy')
# test_data_target = np.load('saved_dataset/crema_d_f32_test_data_target.npy')
# cnn_lstm()
# batch_times = SaveTimes(code="CNN-LSTM_CREMAD", batch_timesx=batch_times)
# vggnet()
# batch_times = SaveTimes(code="VGG_CREMAD", batch_timesx=batch_times)
# resnet_unweighted()
# batch_times = SaveTimes(code="RESNET_CREMAD", batch_timesx=batch_times)
# resnet_lstm_unweighted()
# batch_times = SaveTimes(code="RESNET-LSTM_CREMAD", batch_timesx=batch_times)

# train_data_value = np.load('saved_dataset/saveeFix/standardSR_savee_data.npy')
# train_data_target = np.load(
#     'saved_dataset/saveeFix/standardSR_savee_data_target.npy')
# test_data_value = np.load('saved_dataset/saveeFix/standardSR_savee_test.npy')
# test_data_target = np.load(
#     'saved_dataset/saveeFix/standardSR_savee_test_target.npy')
# cnn_lstm()
# batch_times = SaveTimes(code="2CNN-LSTM_SAVEE", batch_timesx=batch_times)
# vggnet()
# batch_times = SaveTimes(code="2VGG_SAVEE", batch_timesx=batch_times)
# resnet_unweighted()
# batch_times = SaveTimes(code="2RESNET_SAVEE", batch_timesx=batch_times)
# resnet_lstm_unweighted()
# batch_times = SaveTimes(code="2RESNET-LSTM_SAVEE", batch_timesx=batch_times)

# train_data_value = np.load(
#     'saved_dataset/ravdessFix/ravdess_data.npy')
# train_data_target = np.load(
#     'saved_dataset/ravdessFix/ravdess_data_target.npy')
# test_data_value = np.load('saved_dataset/ravdess_test.npy')
# test_data_target = np.load('saved_dataset/ravdess_test_target.npy')
# cnn_lstm()
# batch_times = SaveTimes(code="CNN-LSTM_RAVDESS", batch_timesx=batch_times)
# vggnet()
# batch_times = SaveTimes(code="VGG_RAVDESS", batch_timesx=batch_times)
# resnet_unweighted()
# batch_times = SaveTimes(code="RESNET_RAVDESS", batch_timesx=batch_times)
# resnet_lstm_unweighted()
# batch_times = SaveTimes(code="RESNET-LSTM_RAVDESS", batch_timesx=batch_times)
# train_data_value1 = np.load(
#     'saved_dataset/ravdessFix/ravdess_dataWithGaussian_train.npy')
# train_data_target1 = np.load(
#     'saved_dataset/ravdessFix/ravdess_dataWithGaussian_target.npy')
# train_data_value2 = np.load(
#     'saved_dataset/ravdessFix/ravdess_dataWithTimeStretch2_train.npy')
# train_data_target2 = np.load(
#     'saved_dataset/ravdessFix/ravdess_dataWithTimeStretch2_target.npy')
# train_data_value3 = np.load(
#     'saved_dataset/ravdessFix/ravdess_dataWithPitchShift_train.npy')
# train_data_target3 = np.load(
#     'saved_dataset/ravdessFix/ravdess_dataWithPitchShift_target.npy')


# test_data_value = np.load('saved_dataset/crema_d_f32_test_data_value.npy')
# test_data_target = np.load('saved_dataset/crema_d_f32_test_data_target.npy')

# train_data_value1 = np.load(
#     'saved_dataset/crema_train_gaussian_value_padded.npy')
# train_data_target1 = np.load(
#     'saved_dataset/crema_train_gaussian_target.npy')
# train_data_value2 = np.load(
#     'saved_dataset/crema_train_timeStretch_value_padded.npy')
# train_data_target2 = np.load(
#     'saved_dataset/crema_train_timeStretch_target.npy')
# train_data_value3 = np.load(
#     'saved_dataset/crema_train_pitch_value_padded.npy')
# train_data_target3 = np.load(
#     'saved_dataset/crema_train_pitch_target.npy')

# CREMA DEEZ NAT
# train_data_value = train_data_value1
# train_data_target = train_data_target1
# cnn_lstm()
# batch_times = SaveTimes(code="Crema-G3", batch_timesx=batch_times)

# train_data_value = train_data_value2
# train_data_target = train_data_target2
# cnn_lstm()
# batch_times = SaveTimes(code="Crema-T3", batch_timesx=batch_times)

# train_data_value = train_data_value3
# train_data_target = train_data_target3
# cnn_lstm()
# batch_times = SaveTimes(code="Crema-P3", batch_timesx=batch_times)

# train_data_value = np.concatenate(
#     (train_data_value1, train_data_value2[math.floor(len(train_data_value2)/2):]), axis=0)
# train_data_target = np.concatenate(
#     (train_data_target1, train_data_target2[math.floor(len(train_data_target2)/2):]), axis=0)
# cnn_lstm()
# batch_times = SaveTimes(code="Crema-GT3", batch_timesx=batch_times)

# train_data_value = np.concatenate(
#     (train_data_value1, train_data_value3[math.floor(len(train_data_value3)/2):]), axis=0)
# train_data_target = np.concatenate(
#     (train_data_target1, train_data_target3[math.floor(len(train_data_target3)/2):]), axis=0)
# cnn_lstm()
# batch_times = SaveTimes(code="Crema-GP3", batch_timesx=batch_times)

# train_data_value = np.concatenate(
#     (train_data_value2, train_data_value3[math.floor(len(train_data_value3)/2):]), axis=0)
# train_data_target = np.concatenate(
#     (train_data_target2, train_data_target3[math.floor(len(train_data_target3)/2):]), axis=0)
# cnn_lstm()
# batch_times = SaveTimes(code="Crema-TP3", batch_timesx=batch_times)


# train_data_value = np.concatenate(
#     (train_data_value1, train_data_value2[math.floor(len(train_data_value2)/2):], train_data_value3[math.floor(len(train_data_value3)/2):]), axis=0)
# train_data_target = np.concatenate(
#     (train_data_target1, train_data_target2[math.floor(len(train_data_value2)/2):], train_data_target3[math.floor(len(train_data_target3)/2):]), axis=0)
# cnn_lstm()
# batch_times = SaveTimes(code="Crema-GTP3", batch_timesx=batch_times)

test_data_value = np.load('saved_dataset/ravdessFix/ravdess_test.npy')
test_data_target = np.load('saved_dataset/ravdessFix/ravdess_test_target.npy')
train_data_value1 = np.load(
    'saved_dataset/ravdessFix/ravdess_dataWithGaussian_train.npy')
train_data_target1 = np.load(
    'saved_dataset/ravdessFix/ravdess_dataWithGaussian_target.npy')
train_data_value2 = np.load(
    'saved_dataset/ravdessFix/ravdess_dataWithTimeStretch2_train.npy')
train_data_target2 = np.load(
    'saved_dataset/ravdessFix/ravdess_dataWithTimeStretch2_target.npy')
train_data_value3 = np.load(
    'saved_dataset/ravdessFix/ravdess_dataWithPitchShift_train.npy')
train_data_target3 = np.load(
    'saved_dataset/ravdessFix/ravdess_dataWithPitchShift_target.npy')

train_data_value = train_data_value1
train_data_target = train_data_target1
cnn_lstm()
batch_times = SaveTimes(code="Ravdess-G", batch_timesx=batch_times)

train_data_value = train_data_value2
train_data_target = train_data_target2
cnn_lstm()
batch_times = SaveTimes(code="Ravdess-T", batch_timesx=batch_times)

train_data_value = train_data_value3
train_data_target = train_data_target3
cnn_lstm()
batch_times = SaveTimes(code="Ravdess-P", batch_timesx=batch_times)

train_data_value = np.concatenate(
    (train_data_value1, train_data_value2[math.floor(len(train_data_value2)/2):]), axis=0)
train_data_target = np.concatenate(
    (train_data_target1, train_data_target2[math.floor(len(train_data_target2)/2):]), axis=0)
cnn_lstm()
batch_times = SaveTimes(code="Ravdess-GT", batch_timesx=batch_times)

train_data_value = np.concatenate(
    (train_data_value1, train_data_value3[math.floor(len(train_data_value3)/2):]), axis=0)
train_data_target = np.concatenate(
    (train_data_target1, train_data_target3[math.floor(len(train_data_target3)/2):]), axis=0)
cnn_lstm()
batch_times = SaveTimes(code="Ravdess-GP", batch_timesx=batch_times)

train_data_value = np.concatenate(
    (train_data_value2, train_data_value3[math.floor(len(train_data_value3)/2):]), axis=0)
train_data_target = np.concatenate(
    (train_data_target2, train_data_target3[math.floor(len(train_data_target3)/2):]), axis=0)
cnn_lstm()
batch_times = SaveTimes(code="Ravdess-TP", batch_timesx=batch_times)


train_data_value = np.concatenate(
    (train_data_value1, train_data_value2[math.floor(len(train_data_value2)/2):], train_data_value3[math.floor(len(train_data_value3)/2):]), axis=0)
train_data_target = np.concatenate(
    (train_data_target1, train_data_target2[math.floor(len(train_data_value2)/2):], train_data_target3[math.floor(len(train_data_target3)/2):]), axis=0)
cnn_lstm()
batch_times = SaveTimes(code="Ravdess-GTP", batch_timesx=batch_times)
