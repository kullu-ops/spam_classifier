import streamlit as st
import pickle
import string
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

ps = PorterStemmer()

# load saved model
model = pickle.load(open("model.pkl","rb"))
tfidf = pickle.load(open("vectorizer.pkl","rb"))

def preprocess(text):

    text = text.lower()

    text = text.translate(str.maketrans('', '', string.punctuation))

    text = re.sub(r'\d+', '', text)

    words = text.split()

    stop_words = set(stopwords.words('english'))

    words = [word for word in words if word not in stop_words]

    words = [ps.stem(word) for word in words]

    return " ".join(words)


# UI
st.title("📧 Email Spam Classifier")

message = st.text_area("Enter your message")

if st.button("Predict"):

    processed = preprocess(message)

    vector_input = tfidf.transform([processed])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.error("🚨 Spam Message")

    else:
        st.success("✅ Not Spam (Ham)")
