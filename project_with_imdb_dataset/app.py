import streamlit as st
import re

from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

word_index = imdb.get_word_index()

st.title("IMDB Movie Review Classification")

#loading model

model = load_model("model.keras")

#getting review text from user
review_text = st.text_input("Enter moview review to classify:")

def preprocess_review(sample_review:str):
    sample_review = sample_review.lower()
    sample_review = re.sub('[^a-zA-Z]', ' ', sample_review)
    sample_review_words = sample_review.split(" ")
    sample_review_seq = [word_index.get(word , -1 ) + 3 for word in sample_review_words]
    padded_sample_review_seq = sequence.pad_sequences([sample_review_seq], maxlen=500)
    return padded_sample_review_seq

def predict_review_class(sample_review):
    review_seq = preprocess_review(sample_review)
    y_pred = model.predict(review_seq)
    pred_val = y_pred[0][0]

    if pred_val > 0.4:
        return ("Positive Review", pred_val)
    else:
        return ("Negative Review", pred_val)

if st.button("Classify"):

    sentiment, prob = predict_review_class(review_text)
    st.write(f"Prediction_probability: {prob}")
    st.write(sentiment)
    




