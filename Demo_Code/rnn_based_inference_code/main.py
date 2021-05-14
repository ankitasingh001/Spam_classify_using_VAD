import streamlit as st


st.title("CLAP 👏🏼 Spam 👹 Detection 🕵️‍♀️ 🕵🏻‍♂️")

import librosa
import numpy as np
import keras
import pandas as pd
import os
magic = keras.models.load_model('trained_rnn.h5')

seq_len = 200
def extract_feature(file_name):
   
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        # mfccsscaled = np.mean(mfccs.T,axis=0)
        
        to_pad = mfccs[:,:seq_len]
        v = max(0, (seq_len-to_pad.shape[-1]))
        mfccsscaled = np.pad(to_pad,((0,0),(0,v))).T
        
    except Exception as e:
        print("Error encountered while parsing file: ", file)
        return None
     
    return np.array([mfccsscaled])
    
def print_prediction(file_name):
    prediction_feature = extract_feature(file_name)

    predicted_vector = magic.predict_classes(prediction_feature)
#    predicted_class = le.inverse_transform(predicted_vector)
#    print("The predicted class is:", predicted_class[0], '\n')
    st.write("The predicted vector is:", predicted_vector[0], '\n')
    
    predicted_proba_vector = magic.predict_proba(prediction_feature)
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)):
#        category = le.inverse_transform(np.array([i]))
        st.write("Class: ",str(np.array([i][0])), "\t\t : ", format(predicted_proba[i], '.32f') )
    chart_data = pd.DataFrame([(predicted_proba[0],0),(0,predicted_proba[1])],columns=["Spam","Speech"])
    st.write("Class Probability Distribution")
    st.bar_chart(chart_data)
filename = './CAFE-CAFE-1_trim_5s_505.wav'



clips = tuple(os.listdir("./audio_files"))

filename = './audio_files/'+st.sidebar.selectbox("Audio File", clips)


st.write("USING RNN MODEL")
st.write("\n\n\n\n\n\n")

#if(filename is not None and len(filename)>0):
#st.write(filename)
#filename = './audio_files/CAFE-CAFE-1_trim_5s_505.wav'
print_prediction(filename)

audio_file = open(filename, 'rb')
audio_bytes = audio_file.read()
st.audio(audio_bytes)
