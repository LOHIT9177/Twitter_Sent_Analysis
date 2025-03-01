import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained model
model_path = "Sentiment Analysis for Tweets.pkl"
vectorizer_path = r"D:\sentiment_analysis\tfidf_vectorizer.pkl" 

with open(model_path, "rb") as file:
    model = pickle.load(file)

with open(vectorizer_path, "rb") as file:
    vectorizer = pickle.load(file)



# Streamlit UI
st.title("Sentiment Analysis for Tweets")
st.write("Enter a tweet to analyze its sentiment:")

# User input
tweet_input = st.text_area("Enter Tweet Here")

if st.button("Analyze Sentiment"):
    if tweet_input:
        try:
            # Transform input using the same vectorizer
            transformed_tweet = vectorizer.transform([tweet_input])
            prediction = model.predict(transformed_tweet)
            sentiment = "Positive 😊" if prediction[0] == 1 else "Negative 😞"

            st.success(f"Sentiment: {sentiment}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a tweet before analyzing.")
