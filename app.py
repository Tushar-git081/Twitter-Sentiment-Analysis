import pickle
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

## Load the Tensorflow Model for Prediction
model=load_model('model.h5')

with open('tokenizer.pkl','rb') as file:
    tokenizer=pickle.load(file)

st.title('Twitter Tweets Sentiment Analysis')
tweet=st.text_area('Enter the Tweet: ')

if st.button('Predict Sentiment') and tweet.strip():
    sequences=tokenizer.texts_to_sequences([tweet])
    sequences=pad_sequences(sequences,padding='post',maxlen=99)
    prediction = model.predict(sequences)
    predicted_class=np.argmax(prediction,axis=1)[0]

    sentiment_map={0:'Negative',1:'Neutral',2:'Positive'}
    
    st.write('Sentiments',sentiment_map[predicted_class])