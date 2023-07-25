import tensorflow as tf


def train_snn(train_data_value, train_data_target, test_data_value, test_data_target):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(
            train_data_value.shape[1], train_data_value.shape[2])),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax'),
    ])
    optimiser = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    tensorboard_callback_snn = tf.keras.callbacks.TensorBoard(
        log_dir='./logs/modelSNN')
    history = model.fit(train_data_value, train_data_target,
                        batch_size=32,
                        epochs=10,
                        validation_data=(test_data_value, test_data_target),
                        callbacks=[tensorboard_callback_snn])

    return history


def train_cnn(train_data_value, train_data_target, test_data_value, test_data_target, callbackdir):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(
        train_data_value.shape[1], train_data_value.shape[2], 1)))
    model.add(tf.keras.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(tf.keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(
        (2, 2), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # flatten output and feed it into dense layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(6, activation='softmax'))
    optimiser = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.summary()
    tensorboard_callback_cnn = tf.keras.callbacks.TensorBoard(
        log_dir=callbackdir)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_data_value, train_data_target, validation_data=(
        test_data_value, test_data_target), batch_size=32, epochs=50, callbacks=[tensorboard_callback_cnn])
    return history


def train_lstm(train_data_value, train_data_target, test_data_value, test_data_target):
    model = tf.keras.Sequential()
    # 2 LSTM layers
    model.add(tf.keras.layers.LSTM(64, input_shape=(
        train_data_value.shape[1], train_data_value.shape[2]), return_sequences=True))
    model.add(tf.keras.layers.LSTM(64))

    # dense layer
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(6, activation='softmax'))
    optimiser = tf.keras.optimizers.Adam(learning_rate=0.0001)
    tensorboard_callback_lstm = tf.keras.callbacks.TensorBoard(
        log_dir='./logs/modelLSTM')
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(train_data_value, train_data_target, validation_data=(
        test_data_value, test_data_target), batch_size=32, epochs=10, callbacks=[tensorboard_callback_lstm])
    return history


def train_cnnlstm(train_data_value, train_data_target, test_data_value, test_data_target):
    tensorboard_callback_cnnlstm = tf.keras.callbacks.TensorBoard(
        log_dir='./logs/modelCNNLSTM')
