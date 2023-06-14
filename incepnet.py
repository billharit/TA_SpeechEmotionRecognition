import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, concatenate

# Define the Inception module


def inception_module(x, filters):
    # 1x1 convolution
    conv1 = Conv2D(filters[0], (1, 1), padding='same', activation='relu')(x)

    # 3x3 convolution
    conv3 = Conv2D(filters[1], (3, 3), padding='same', activation='relu')(x)

    # 5x5 convolution
    conv5 = Conv2D(filters[2], (5, 5), padding='same', activation='relu')(x)

    # Max pooling
    pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_conv = Conv2D(filters[3], (1, 1),
                       padding='same', activation='relu')(pool)

    # Concatenate all the branches
    merged = concatenate([conv1, conv3, conv5, pool_conv], axis=-1)

    return merged

# Build the InceptionNet model for SER


def build_inception_net(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # First Inception module
    x = inception_module(inputs, [16, 16, 16, 16])

    # Second Inception module
    x = inception_module(x, [32, 32, 32, 32])

    # Flatten the output
    x = Flatten()(x)

    # Fully connected layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=x)

    return model


# Define the input shape and number of classes
input_shape = (308, 40, 1)
num_classes = 10  # Change this according to your SER dataset

# Build the InceptionNet model
model = build_inception_net(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()
