# -*- coding: utf-8 -*-
"""
Created

To deploy Tweet classification model
"""

import pickle
import re

import streamlit as st

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from tensorflow.keras.models import load_model

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def data_preprocessing(text):
    # Remove all mentions
    text = re.sub(r'@\w+', '', text)

    # Remove all URLs
    text = re.sub(r'http\S+', '', text)

    # Remove all symbols from the 'tweet' column
    text = re.sub(r'[^\w\s]', '', text)

    # Remove words that come immediately after numbers (e.g., "19th")
    text = re.sub(r'\d+\w*', '', text)

    # Remove all numeric digits (this will take care of any standalone digits that are left)
    text = re.sub(r'\d+', '', text)

    # Remove extra whitespace: leading and trailing whitespace
    text = text.strip()

    # Replace multiple whitespace characters (\s+) with a single space
    text = re.sub(r'\s+', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # Removal of stopwords
    text = ' '.join(word for word in text.split() if word not in stop_words)

    text = ' '.join(stemmer.stem(word) for word in text.split())

    # Return the processed text
    return text


count_vectorizer_path = r"vectorizer.pkl"
count_vectorizer = pickle.load(open(count_vectorizer_path, 'rb'))

# Load the model from the HDF5 file
model_path = r"tf_model.h5"
model = load_model(model_path)


# Function to analyze text
def analyze_text(input_text):
    # Check if the input text is empty
    if not input_text:
        return {"Veracity of Tweet": "Unknown"}

    # Preprocess the input text
    preprocessed_text = data_preprocessing(input_text)

    # Transform the preprocessed text using the trained count_vectorizer
    text_feature_vector = count_vectorizer.transform([preprocessed_text])
    text_features = text_feature_vector.toarray()

    # Predict label for the input text
    text_predictions = model.predict(text_features)

    # Extract the predicted class
    predicted_class = int(text_predictions[0] > 0.5)

    # Customize output based on model predictions
    result_mapping = {0: "False Tweet - News is not genuine", 1: "True Tweet - News is authentic"}

    result = result_mapping.get(predicted_class, "Unknown")

    return {"Veracity of Tweet": result}


# Streamlit UI

st.set_page_config(
    page_title="News Verification",
    page_icon="🔍✔️")

st.title("Tweet Classification")
st.text("Enter Tweet:")

input_text = st.text_area("Text Entry", height=200)
result = analyze_text(input_text)

# Display the results
st.text(f"Veracity of Tweet: {result['Veracity of Tweet']}")
