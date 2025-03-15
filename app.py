import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import streamlit as st
# Load the IMDB dataset
word_index = imdb.get_word_index()
reversed_word_index = {value: key for key, value in word_index.items()}


# Load the Pretrained Model
model = load_model('simple_rnn_model_imdb.h5')

# Use me helper functions
def decode_review(encoded_text):
    """Function to decode the review from the encoded text
    Args:
    -----
    encoded_text : list of integers
    
    Returns:
    --------
    decoded_text : str
    """
    
    return ' '.join(encoded_text.get(i - 3, '?') for i in encoded_text)

def preprocess_text(text: str, max_length: int = 500): 
    """Function to preprocess the User input text to predict on it 

    Args:
        text (str): User input text
        max_length (int): Maximum length of the text

    Returns:
        padded_review: a text with a uniform length
    """
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=max_length)
    return padded_review

# Create a function to predict on the user input text
def predict_sentiment(text: str, model, max_length: int = 500):
    """Function to predict the sentiment of the user input text

    Args:
        text (str): User input text
        model : Pretrained model
        max_length (int) (not required): Maximum length of the text

    Returns:
        sentiment : str
        Prediction : float
    """
    padded_review = preprocess_text(text, max_length=max_length)
    prediction = model.predict(padded_review)
    sentiment ="Positive" if prediction[0][0] >= 0.5 else "Negative"
    return sentiment, prediction[0][0]

# Create the streamlit app

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative.")
st.sidebar.header("Navigation")
st.sidebar.markdown("Created by [Ahmed Yusri](https://www.linkedin.com/in/ahmed-yusri-499a67313)")
st.sidebar.image("images.png")

# User input
user_review = st.text_area("Moive Review")

if st.button("Classify"):
    # Process the user input
    preprocess_input = preprocess_text(user_review)
    
    # Make a model prediction for the given user input
    sentiment, prediction = predict_sentiment(user_review, model)
    
    # Display the Results
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Predictions: {prediction}")
    
else:
    st.write("Please enter movie review")
