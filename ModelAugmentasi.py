import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
import math


class StopTrainingOnAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, target_accuracy):
        super(StopTrainingOnAccuracyCallback, self).__init__()
        self.target_accuracy = target_accuracy

    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_accuracy') >= self.target_accuracy:
            print(
                f"\nTarget accuracy of {self.target_accuracy} reached. Stopping training.")
            self.model.stop_training = True


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
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256)))
    model.add(tf.keras.layers.Dropout(0.4))
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

    str_learning_rate = str(learning_rate).replace('.', '')
    csv_logger = tf.keras.callbacks.CSVLogger(
        '{0}{1}_{2}.csv'.format(modelName, optimizer, str_learning_rate))
    # model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #     '{}'.format(modelName), monitor='val_accuracy', mode='max', save_best_only=True)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', verbose=1, patience=1, mode='max', baseline=0.3014)
    callback = StopTrainingOnAccuracyCallback(0.81)
    history = model.fit(train_data_value, train_data_target, validation_data=(
        test_data_value, test_data_target), batch_size=32, epochs=90, callbacks=[csv_logger, callback])
    return model, history


def vggnet(optimizer='adam', learning_rate=0.0001, code=""):
    model = tf.keras.Sequential()
    base_model = tf.keras.applications.vgg16.VGG16(
        include_top=False, weights=None, input_shape=(train_data_value.shape[1], train_data_value.shape[2], 1))
    model.add(base_model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, activation="relu"))
    model.add(tf.keras.layers.Dense(4096, activation="relu"))
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
        'logbuataugmentasi/{0}_{1}_{2}.csv'.format(code, optimizer, str_learning_rate))
    print("{}{}{}".format(code, optimizer, str_learning_rate))
    history = model.fit(train_data_value, train_data_target, validation_data=(
        test_data_value, test_data_target), batch_size=32, epochs=60, callbacks=[csv_logger, early_stop])
    print("{}{}{}".format(code, optimizer, str_learning_rate))

    return model, history


def resnet_unweighted(optimizer='adam', learning_rate=0.0001, code=""):
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
        'logbuataugmentasi/{0}{1}_{2}.csv'.format(code, optimizer, str_learning_rate))
    print("{}{}{}".format(code, optimizer, str_learning_rate))
    history = model.fit(train_data_value, train_data_target, validation_data=(
        test_data_value, test_data_target), batch_size=32, epochs=60, callbacks=[csv_logger, early_stop])
    print("{}{}{}".format(code, optimizer, str_learning_rate))

    return model, history


def resnet_lstm(optimizer='adam', learning_rate=0.0001, code=""):
    model = tf.keras.Sequential()
    base_model = tf.keras.applications.resnet_v2.ResNet50V2(
        include_top=False, weights=None, input_shape=(train_data_value.shape[1], train_data_value.shape[2], 1))
    model.add(base_model)
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256)))
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
        'logbuataugmentasi/LSTM{0}{1}_{2}.csv'.format(code, optimizer, str_learning_rate))
    print("{}{}{}".format(code, optimizer, str_learning_rate))
    history = model.fit(train_data_value, train_data_target, validation_data=(
        test_data_value, test_data_target), batch_size=32, epochs=60, callbacks=[early_stop, csv_logger])
    print("{}{}{}".format(code, optimizer, str_learning_rate))

    return model, history


train_data_value1 = np.load(
    'saved_dataset/saveeFix/standardSR_savee_dataWithGaussianNoise_train.npy')
train_data_target1 = np.load(
    'saved_dataset/saveeFix/standardSR_savee_dataWithGaussianNoise_target.npy')
train_data_value2 = np.load(
    'saved_dataset/saveeFix/standardSR_savee_dataWithTimeStretch_train.npy')
train_data_target2 = np.load(
    'saved_dataset/saveeFix/standardSR_savee_dataWithTimeStretch_target.npy')
train_data_value3 = np.load(
    'saved_dataset/saveeFix/standardSR_savee_dataWithPitchShift_train.npy')
train_data_target3 = np.load(
    'saved_dataset/saveeFix/standardSR_savee_dataWithPitchShift_target.npy')
test_data_value = np.load(
    'saved_dataset/saveeFix/standardSR_savee_validation.npy')
test_data_target = np.load(
    'saved_dataset/saveeFix/standardSR_savee_validation_target.npy')
train_data_value = np.load('saved_dataset/saveeFix/standardSR_savee_data.npy')
train_data_target = np.load(
    'saved_dataset/saveeFix/standardSR_savee_data_target.npy')

NAMANYE = "ZZaResNetSavee_Ori"
model, history = resnet_unweighted("sgd", 0.01, NAMANYE)
NAMANYE = "ZZbResNetSavee_Ori"
model, history = resnet_unweighted("sgd", 0.01, NAMANYE)
NAMANYE = "ZZcResNetSavee_Ori"
model, history = resnet_unweighted("sgd", 0.01, NAMANYE)
NAMANYE = "ZZdResNetSavee_Ori"
model, history = resnet_unweighted("sgd", 0.01, NAMANYE)
NAMANYE = "ZZeResNetSavee_Ori"
model, history = resnet_unweighted("sgd", 0.01, NAMANYE)
NAMANYE = "ZZfResNetSavee_Ori"
model, history = resnet_unweighted("sgd", 0.01, NAMANYE)

# train_data_value = train_data_value1
# train_data_target = train_data_target1
# NAMANYE = "ZZResNetSavee_GN"
# model, history = resnet_unweighted("sgd", 0.01, NAMANYE)

# train_data_value = train_data_value2
# train_data_target = train_data_target2
# NAMANYE = "ZZResNetSavee_TS"
# model, history = resnet_unweighted("sgd", 0.01, NAMANYE)

# train_data_value = train_data_value3
# train_data_target = train_data_target3
# NAMANYE = "ZZResNetSavee_PS"
# model, history = resnet_unweighted("sgd", 0.01, NAMANYE)


# tf.keras.backend.clear_session()
# train_data_value = np.concatenate(
#     (train_data_value1,  train_data_value2[math.floor(len(train_data_value2)/2):]), axis=0)
# train_data_target = np.concatenate(
#     (train_data_target1, train_data_target2[math.floor(len(train_data_target2)/2):]), axis=0)

# NAMANYE = "ZZResNetSavee_GN-TS"
# model, history = resnet_unweighted("sgd", 0.01, NAMANYE)

# train_data_value = np.concatenate(
#     (train_data_value1,  train_data_value3[math.floor(len(train_data_value3)/2):]), axis=0)
# train_data_target = np.concatenate(
#     (train_data_target1, train_data_target3[math.floor(len(train_data_target3)/2):]), axis=0)

# NAMANYE = "ZZResNetSavee_GN-PS"
# model, history = resnet_unweighted("sgd", 0.01, NAMANYE)

# train_data_value = np.concatenate(
#     (train_data_value2,  train_data_value3[math.floor(len(train_data_value3)/2):]), axis=0)
# train_data_target = np.concatenate(
#     (train_data_target2, train_data_target3[math.floor(len(train_data_target3)/2):]), axis=0)

# NAMANYE = "ZZResNetSavee_TS-PS"
# model, history = resnet_unweighted("sgd", 0.01, NAMANYE)

# train_data_value = np.concatenate(
#     (train_data_value1,  train_data_value2[math.floor(len(train_data_value2)/2):],  train_data_value3[math.floor(len(train_data_value3)/2):]), axis=0)
# train_data_target = np.concatenate(
#     (train_data_target1,  train_data_target2[math.floor(len(train_data_target2)/2):], train_data_target3[math.floor(len(train_data_target3)/2):]), axis=0)

# tf.keras.backend.clear_session()
# NAMANYE = "ZZResNetSavee_GN-TS-PS"
# model, history = resnet_unweighted("sgd", 0.01, NAMANYE)


# test_data_value = np.load('saved_dataset/ravdess_validation.npy')
# test_data_target = np.load('saved_dataset/ravdess_validation_target.npy')
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


# train_data_value = train_data_value1
# train_data_target = train_data_target1
# NAMANYE = "ResNetLSTMRavdess_GN"
# model, history = resnet_lstm("adagrad", 0.01, NAMANYE)

# train_data_value = train_data_value2
# train_data_target = train_data_target2
# NAMANYE = "ResNetLSTMRavdess_TS"
# model, history = resnet_lstm("adagrad", 0.01, NAMANYE)

# # train_data_value = train_data_value3
# train_data_target = train_data_target3
# NAMANYE = "ResNetLSTMRavdess_PS"
# model, history = resnet_lstm("adagrad", 0.01, NAMANYE)

# tf.keras.backend.clear_session()
# train_data_value = np.concatenate(
#     (train_data_value1,  train_data_value2[math.floor(len(train_data_value2)/2):]), axis=0)
# train_data_target = np.concatenate(
#     (train_data_target1, train_data_target2[math.floor(len(train_data_target2)/2):]), axis=0)

# NAMANYE = "ResNetLSTMRavdess_GN-TS"
# model, history = resnet_lstm("adagrad", 0.01, NAMANYE)

# train_data_value = np.concatenate(
#     (train_data_value1,  train_data_value3[math.floor(len(train_data_value3)/2):]), axis=0)
# train_data_target = np.concatenate(
#     (train_data_target1, train_data_target3[math.floor(len(train_data_target3)/2):]), axis=0)

# NAMANYE = "ResNetLSTMRavdess_GN-PS"
# model, history = resnet_lstm("adagrad", 0.01, NAMANYE)

# train_data_value = np.concatenate(
#     (train_data_value2,  train_data_value3[math.floor(len(train_data_value3)/2):]), axis=0)
# train_data_target = np.concatenate(
#     (train_data_target2, train_data_target3[math.floor(len(train_data_target3)/2):]), axis=0)

# NAMANYE = "ResNetLSTMRavdess_TS-PS"
# model, history = resnet_lstm("adagrad", 0.01, NAMANYE)

# tf.keras.backend.clear_session()
# train_data_value = np.concatenate(
#     (train_data_value1,  train_data_value2[math.floor(len(train_data_value2)/2):],  train_data_value3[math.floor(len(train_data_value3)/2):]), axis=0)
# train_data_target = np.concatenate(
#     (train_data_target1,  train_data_target2[math.floor(len(train_data_target2)/2):], train_data_target3[math.floor(len(train_data_target3)/2):]), axis=0)

# NAMANYE = "ResNetLSTMRavdess_GN-TS-PS"
# model, history = resnet_lstm("adagrad", 0.01, NAMANYE)


# GATAU
# train_data_value = np.concatenate(
#     (train_data_value1,  train_data_value3[math.floor(len(train_data_value3)/2):]), axis=0)
# train_data_target = np.concatenate(
#     (train_data_target1, train_data_target3[math.floor(len(train_data_target3)/2):]), axis=0)

# num_samples = train_data_value.shape[0]
# indices = np.random.permutation(num_samples)
# shuffled_train_data_value = train_data_value[indices]
# shuffled_train_data_target = train_data_target[indices]

# train_data_value = shuffled_train_data_value
# train_data_target = shuffled_train_data_target
# NAMANYE = "SAVEE_GN-PS4"
# model, history = cnn_lstm("adam", 0.0001, NAMANYE)
# model.save(NAMANYE)


# CREMA-D
test_data_value = np.load(
    'saved_dataset/crema_d_f32_validation_data_value.npy')
test_data_target = np.load(
    'saved_dataset/crema_d_f32_validation_data_target.npy')

train_data_value1 = np.load(
    'saved_dataset/crema_train_gaussian_value_padded.npy')
train_data_target1 = np.load(
    'saved_dataset/crema_train_gaussian_target.npy')
train_data_value2 = np.load(
    'saved_dataset/crema_train_timeStretch_value_padded.npy')
train_data_target2 = np.load(
    'saved_dataset/crema_train_timeStretch_target.npy')
train_data_value3 = np.load(
    'saved_dataset/crema_train_pitch_value_padded.npy')
train_data_target3 = np.load(
    'saved_dataset/crema_train_pitch_target.npy')

# train_data_value = train_data_value1
# train_data_target = train_data_target1
# NAMANYE = "ResNetLSTMCrema-D_GN"
# model, history = resnet_lstm("rmsprop", 0.001, NAMANYE)

# train_data_value = train_data_value2
# train_data_target = train_data_target2
# NAMANYE = "ResNetLSTMCrema-D_TS"
# model, history = resnet_lstm("rmsprop", 0.001, NAMANYE)

# train_data_value = train_data_value3
# train_data_target = train_data_target3
# NAMANYE = "ResNetLSTMCrema-D_PS"
# model, history = resnet_lstm("rmsprop", 0.001, NAMANYE)

# tf.keras.backend.clear_session()
# train_data_value = np.concatenate(
#     (train_data_value1,  train_data_value2[math.floor(len(train_data_value2)/2):]), axis=0)
# train_data_target = np.concatenate(
#     (train_data_target1, train_data_target2[math.floor(len(train_data_target2)/2):]), axis=0)

# NAMANYE = "ResNetLSTMCrema-D_GN-TS"
# model, history = resnet_lstm("rmsprop", 0.001, NAMANYE)

# train_data_value = np.concatenate(
#     (train_data_value1,  train_data_value3[math.floor(len(train_data_value3)/2):]), axis=0)
# train_data_target = np.concatenate(
#     (train_data_target1, train_data_target3[math.floor(len(train_data_target3)/2):]), axis=0)

# NAMANYE = "ResNetLSTMCrema-D_GN-PS"
# model, history = resnet_lstm("rmsprop", 0.001, NAMANYE)

# train_data_value = np.concatenate(
#     (train_data_value2,  train_data_value3[math.floor(len(train_data_value3)/2):]), axis=0)
# train_data_target = np.concatenate(
#     (train_data_target2, train_data_target3[math.floor(len(train_data_target3)/2):]), axis=0)

# NAMANYE = "ResNetLSTMCrema-D_TS-PS"
# model, history = resnet_lstm("rmsprop", 0.001, NAMANYE)
# tf.keras.backend.clear_session()

# train_data_value = np.concatenate(
#     (train_data_value1,  train_data_value2[math.floor(len(train_data_value2)/2):],  train_data_value3[math.floor(len(train_data_value3)/2):]), axis=0)
# train_data_target = np.concatenate(
#     (train_data_target1,  train_data_target2[math.floor(len(train_data_target2)/2):], train_data_target3[math.floor(len(train_data_target3)/2):]), axis=0)

# NAMANYE = "ResNetLSTMCrema-D_GN-TS-PS"
# model, history = resnet_lstm("rmsprop", 0.001, NAMANYE)
