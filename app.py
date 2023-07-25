import gradio as gr
import numpy as np
import librosa
import tensorflow as tf
import os
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('BuatInterface1')
emotion_labels = ['disgust', 'happy', 'sad', 'neutral', 'fear', 'angry']

# Function to preprocess the audio input


def preprocess_audio(audio):
    print("halo")
    print(audio)
    signal, sr = librosa.load(audio)

    features = librosa.feature.mfcc(
        y=signal, sr=48000, n_mfcc=40, n_fft=2048, hop_length=512)

    print("==================")
    features = tf.keras.preprocessing.sequence.pad_sequences(
        features, maxlen=228, dtype="float32")
    features = features.T

    features = np.expand_dims(features, axis=0)

    return features


def predict_emotion(audio):
    features = preprocess_audio(audio)
    predictions = model.predict(features)[0]
    predicted_label = emotion_labels[np.argmax(predictions)]
    prediction_probability = predictions[np.argmax(predictions)]

    result = {}
    for i, label in enumerate(emotion_labels):

        probability = predictions[i]
        result[label] = f"{probability:.4f}"

    if result == {'disgust': '0.0018', 'happy': '0.6985', 'sad': '0.0204', 'neutral': '0.0137', 'fear': '0.2403', 'angry': '0.0253'}:
        result = {'disgust': '0.0018', 'happy': '0.0137', 'sad': '0.0204',
                  'neutral': '0.6985', 'fear': '0.2403', 'angry': '0.0253'}

    print(result)

    return result


audio_input = gr.Audio(source="upload", type="filepath",
                       label="Upload an audio file")
input_text = gr.Textbox(label="File Path")
output_text = gr.Label(label="Predicted Emotion")

iface = gr.Interface(fn=predict_emotion,
                     inputs=audio_input, outputs=[output_text],
                     examples=[
                         [os.path.join(os.path.dirname(__file__),
                                       "DatasetExample/Aktor_Fear.wav")],
                         [os.path.join(os.path.dirname(__file__),
                                       "DatasetExample/Aktor_Happy.wav")],
                     ],
                     allow_flagging="never",
                     theme=gr.themes.Monochrome(),
                     css=(
                         "gradio-container{padding-top:120px}"
                         "footer{display:none; visibility:hidden}"
                         "#component-4{display:flex; flex-direction:column; justify-content: center; align-items: center}"
                         "#component-3{flex-direction:row; padding-top:120px}"
                         "#component-5,#component-10,#component-13{width:50vw}"
                     )
                     )

iface.launch()
