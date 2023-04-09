import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

loaded_model = tf.keras.models.load_model("2RobustCNNLSTM.h5")
test_data_value = np.load('crema_d_f32_test_data_value.npy')
test_data_target = np.load('crema_d_f32_test_data_target.npy')

layer_outputs = [layer.output for layer in loaded_model.layers]
convolution_model_layer0 = tf.keras.models.Model(
    inputs=loaded_model.input, outputs=layer_outputs[0])
convolution_model_layer1 = tf.keras.models.Model(
    inputs=loaded_model.input, outputs=layer_outputs[4])
convolution_model_layer2 = tf.keras.models.Model(
    inputs=loaded_model.input, outputs=layer_outputs[8])
convolution_model_layer3 = tf.keras.models.Model(
    inputs=loaded_model.input, outputs=layer_outputs[12])

layer_output0 = convolution_model_layer0.predict(test_data_value)
np.save("layer_output0", layer_output0)
layer_output1 = convolution_model_layer1.predict(test_data_value)
np.save("layer_output1", layer_output1)
layer_output2 = convolution_model_layer2.predict(test_data_value)
np.save("layer_output2", layer_output2)
layer_output3 = convolution_model_layer3.predict(test_data_value)
np.save("layer_output3", layer_output3)
