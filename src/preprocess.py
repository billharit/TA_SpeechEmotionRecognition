import tensorflow as tf
import numpy as np


def cnn_lstm(optimizer='adam', learning_rate=0.001, code="", nm="", nf="", hl="", half=False):
    # def cnn_lstm():

    model = tf.keras.Sequential()
    if half:
        model.add(tf.keras.layers.Conv2D(64, (2, 2), input_shape=(
            train_data_value.shape[1], train_data_value.shape[2], 1)))
    else:
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
    if half:
        pass
    else:
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
    model.summary()
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10)
    str_learning_rate = str(learning_rate).replace('.', '')
    csv_logger = tf.keras.callbacks.CSVLogger(
        'preprocess_logs/{}_{}_{}_{}.csv'.format(code, nm, nf, hl))
    print("TRAINING FOR DATA {}_{}_{}_{}".format(code, nm, nf, hl))
    # model.summary()
    history = model.fit(train_data_value, train_data_target, validation_data=(
        test_data_value, test_data_target), batch_size=32, epochs=60, callbacks=[csv_logger])
    return model


num_mfcc_list = [26, 40]
n_fft_list = [800, 1024, 2048]

# SAVEE
for x in num_mfcc_list:
    for y in n_fft_list:
        firsthop = int(y/2)
        secondhop = int(y/4)
        if x == 26:
            train_data_value = np.load(
                'preprocess_dataset/SAVEE/{}-{}-{}_data.npy'.format(x, y, firsthop))
            train_data_target = np.load(
                'preprocess_dataset/SAVEE/{}-{}-{}_data_target.npy'.format(x, y, firsthop))
            test_data_value = np.load(
                'preprocess_dataset/SAVEE/{}-{}-{}_test.npy'.format(x, y, firsthop))
            test_data_target = np.load(
                'preprocess_dataset/SAVEE/{}-{}-{}_test_target.npy'.format(x, y, firsthop))
            cnn_lstm(code="SAVEE", nm=x, nf=y, hl=firsthop, half=True)
            train_data_value = np.load(
                'preprocess_dataset/SAVEE/{}-{}-{}_data.npy'.format(x, y, secondhop))
            train_data_target = np.load(
                'preprocess_dataset/SAVEE/{}-{}-{}_data_target.npy'.format(x, y, secondhop))
            test_data_value = np.load(
                'preprocess_dataset/SAVEE/{}-{}-{}_test.npy'.format(x, y, secondhop))
            test_data_target = np.load(
                'preprocess_dataset/SAVEE/{}-{}-{}_test_target.npy'.format(x, y, secondhop))
            cnn_lstm(code="SAVEE", nm=x, nf=y, hl=secondhop, half=True)
        else:
            train_data_value = np.load(
                'preprocess_dataset/SAVEE/{}-{}-{}_data.npy'.format(x, y, firsthop))
            train_data_target = np.load(
                'preprocess_dataset/SAVEE/{}-{}-{}_data_target.npy'.format(x, y, firsthop))
            test_data_value = np.load(
                'preprocess_dataset/SAVEE/{}-{}-{}_test.npy'.format(x, y, firsthop))
            test_data_target = np.load(
                'preprocess_dataset/SAVEE/{}-{}-{}_test_target.npy'.format(x, y, firsthop))
            cnn_lstm(code="SAVEE", nm=x, nf=y, hl=firsthop)
            train_data_value = np.load(
                'preprocess_dataset/SAVEE/{}-{}-{}_data.npy'.format(x, y, secondhop))
            train_data_target = np.load(
                'preprocess_dataset/SAVEE/{}-{}-{}_data_target.npy'.format(x, y, secondhop))
            test_data_value = np.load(
                'preprocess_dataset/SAVEE/{}-{}-{}_test.npy'.format(x, y, secondhop))
            test_data_target = np.load(
                'preprocess_dataset/SAVEE/{}-{}-{}_test_target.npy'.format(x, y, secondhop))
            cnn_lstm(code="SAVEE", nm=x, nf=y, hl=secondhop)


# RAVDESS
for x in num_mfcc_list:
    for y in n_fft_list:
        firsthop = int(y/2)
        secondhop = int(y/4)
        if x == 26:
            train_data_value = np.load(
                'preprocess_dataset/RAVDESS/{}-{}-{}_data.npy'.format(x, y, firsthop))
            train_data_target = np.load(
                'preprocess_dataset/RAVDESS/{}-{}-{}_data_target.npy'.format(x, y, firsthop))
            test_data_value = np.load(
                'preprocess_dataset/RAVDESS/{}-{}-{}_test.npy'.format(x, y, firsthop))
            test_data_target = np.load(
                'preprocess_dataset/RAVDESS/{}-{}-{}_test_target.npy'.format(x, y, firsthop))
            cnn_lstm(code="RAVDESS", nm=x, nf=y, hl=firsthop, half=True)
            train_data_value = np.load(
                'preprocess_dataset/RAVDESS/{}-{}-{}_data.npy'.format(x, y, secondhop))
            train_data_target = np.load(
                'preprocess_dataset/RAVDESS/{}-{}-{}_data_target.npy'.format(x, y, secondhop))
            test_data_value = np.load(
                'preprocess_dataset/RAVDESS/{}-{}-{}_test.npy'.format(x, y, secondhop))
            test_data_target = np.load(
                'preprocess_dataset/RAVDESS/{}-{}-{}_test_target.npy'.format(x, y, secondhop))
            cnn_lstm(code="RAVDESS", nm=x, nf=y, hl=secondhop, half=True)
        else:
            train_data_value = np.load(
                'preprocess_dataset/RAVDESS/{}-{}-{}_data.npy'.format(x, y, firsthop))
            train_data_target = np.load(
                'preprocess_dataset/RAVDESS/{}-{}-{}_data_target.npy'.format(x, y, firsthop))
            test_data_value = np.load(
                'preprocess_dataset/RAVDESS/{}-{}-{}_test.npy'.format(x, y, firsthop))
            test_data_target = np.load(
                'preprocess_dataset/RAVDESS/{}-{}-{}_test_target.npy'.format(x, y, firsthop))
            cnn_lstm(code="RAVDESS", nm=x, nf=y, hl=firsthop)
            train_data_value = np.load(
                'preprocess_dataset/RAVDESS/{}-{}-{}_data.npy'.format(x, y, secondhop))
            train_data_target = np.load(
                'preprocess_dataset/RAVDESS/{}-{}-{}_data_target.npy'.format(x, y, secondhop))
            test_data_value = np.load(
                'preprocess_dataset/RAVDESS/{}-{}-{}_test.npy'.format(x, y, secondhop))
            test_data_target = np.load(
                'preprocess_dataset/RAVDESS/{}-{}-{}_test_target.npy'.format(x, y, secondhop))
            cnn_lstm(code="RAVDESS", nm=x, nf=y, hl=secondhop)

# CREMA-D
for x in num_mfcc_list:
    for y in n_fft_list:
        firsthop = int(y/2)
        secondhop = int(y/4)
        if x == 26:
            train_data_value = np.load(
                'preprocess_dataset/CREMA-D/{}-{}-{}_data.npy'.format(x, y, firsthop))
            train_data_target = np.load(
                'preprocess_dataset/CREMA-D/{}-{}-{}_data_target.npy'.format(x, y, firsthop))
            test_data_value = np.load(
                'preprocess_dataset/CREMA-D/{}-{}-{}_test.npy'.format(x, y, firsthop))
            test_data_target = np.load(
                'preprocess_dataset/CREMA-D/{}-{}-{}_test_target.npy'.format(x, y, firsthop))
            cnn_lstm(code="CREMA-D", nm=x, nf=y, hl=firsthop, half=True)
            train_data_value = np.load(
                'preprocess_dataset/CREMA-D/{}-{}-{}_data.npy'.format(x, y, secondhop))
            train_data_target = np.load(
                'preprocess_dataset/CREMA-D/{}-{}-{}_data_target.npy'.format(x, y, secondhop))
            test_data_value = np.load(
                'preprocess_dataset/CREMA-D/{}-{}-{}_test.npy'.format(x, y, secondhop))
            test_data_target = np.load(
                'preprocess_dataset/CREMA-D/{}-{}-{}_test_target.npy'.format(x, y, secondhop))
            cnn_lstm(code="CREMA-D", nm=x, nf=y, hl=secondhop, half=True)
        else:
            train_data_value = np.load(
                'preprocess_dataset/CREMA-D/{}-{}-{}_data.npy'.format(x, y, firsthop))
            train_data_target = np.load(
                'preprocess_dataset/CREMA-D/{}-{}-{}_data_target.npy'.format(x, y, firsthop))
            test_data_value = np.load(
                'preprocess_dataset/CREMA-D/{}-{}-{}_test.npy'.format(x, y, firsthop))
            test_data_target = np.load(
                'preprocess_dataset/CREMA-D/{}-{}-{}_test_target.npy'.format(x, y, firsthop))
            cnn_lstm(code="CREMA-D", nm=x, nf=y, hl=firsthop)
            train_data_value = np.load(
                'preprocess_dataset/CREMA-D/{}-{}-{}_data.npy'.format(x, y, secondhop))
            train_data_target = np.load(
                'preprocess_dataset/CREMA-D/{}-{}-{}_data_target.npy'.format(x, y, secondhop))
            test_data_value = np.load(
                'preprocess_dataset/CREMA-D/{}-{}-{}_test.npy'.format(x, y, secondhop))
            test_data_target = np.load(
                'preprocess_dataset/CREMA-D/{}-{}-{}_test_target.npy'.format(x, y, secondhop))
            cnn_lstm(code="CREMA-D", nm=x, nf=y, hl=secondhop)
