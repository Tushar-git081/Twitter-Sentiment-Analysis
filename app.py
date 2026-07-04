import pickle
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from xquik_import import XquikImportError, load_xquik_texts

## Load the Tensorflow Model for Prediction
model = load_model('model.h5')

with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

st.title('Twitter Tweets Sentiment Analysis')
st.caption('Paste one tweet or upload a Xquik CSV, JSON, or JSONL export.')

xquik_upload = st.file_uploader(
    'Optional Xquik export',
    type=['csv', 'json', 'jsonl'],
)
imported_tweets = []
if xquik_upload is not None:
    try:
        imported_tweets = load_xquik_texts(xquik_upload)
        st.success(f'Loaded {len(imported_tweets)} tweets from the export.')
    except XquikImportError as error:
        st.error(str(error))

selected_tweet = ''
if imported_tweets:
    selected_tweet = st.selectbox('Choose an imported tweet', imported_tweets)

tweet = st.text_area('Enter the Tweet: ', value=selected_tweet)

if st.button('Predict Sentiment'):
    if not tweet.strip():
        st.warning('Enter or choose a tweet before predicting sentiment.')
    else:
        sequences = tokenizer.texts_to_sequences([tweet])
        sequences = pad_sequences(sequences, padding='post', maxlen=99)
        prediction = model.predict(sequences)
        predicted_class = np.argmax(prediction, axis=1)[0]

        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

        st.write('Sentiment', sentiment_map[predicted_class])
