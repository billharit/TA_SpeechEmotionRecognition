import gradio as gr
import tensorflow as tf
import librosa
import numpy as np

model = tf.keras.models.load_model('RAVDESS_TS-PS8')
emotion_labels = ['disgust', 'happy', 'sad', 'neutral', 'fear', 'angry']

# Function to preprocess the audio input


def preprocess_audio(audio):
    # Load the audio file using librosa
    signal, sr = librosa.load(audio, sr=48000)
    # Extract features from the audio using librosa
    features = librosa.feature.mfcc(
        signal, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
    # Reshape the features array
    features = np.expand_dims(features, axis=0)
    return features


def predict_emotion(audio):
    # Preprocess the audio
    features = preprocess_audio(audio)
    # Make a prediction using the loaded model
    predictions = model.predict(features)[0]
    # Get the predicted emotion label and probability
    predicted_label = emotion_labels[np.argmax(predictions)]
    prediction_probability = predictions[np.argmax(predictions)]
    return {'emotion': predicted_label, 'probability': prediction_probability}


# Create the Gradio interface
audio_input = gr.Audio(source="upload", label="Upload an audio file")
output_text = gr.Textbox(label="Predicted Emotion")

iface = gr.Interface(fn=predict_emotion,
                     inputs=audio_input, outputs=output_text)

# Launch the interface
iface.launch()
