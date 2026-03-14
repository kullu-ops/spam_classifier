import pandas as pd
import string
import re
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

nltk.download('stopwords')

ps = PorterStemmer()

# preprocessing function
def preprocess(text):

    text = text.lower()

    text = text.translate(str.maketrans('', '', string.punctuation))

    text = re.sub(r'\d+', '', text)

    words = text.split()

    stop_words = set(stopwords.words('english'))

    words = [word for word in words if word not in stop_words]

    words = [ps.stem(word) for word in words]

    return " ".join(words)


# load dataset
df = pd.read_csv("spam.csv", encoding="latin1")

df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace=True)

df.rename(columns={'v1':'target','v2':'text'}, inplace=True)

df['target'] = df['target'].map({'ham':0,'spam':1})

df['text'] = df['text'].apply(preprocess)

# vectorization
tfidf = TfidfVectorizer()

X = tfidf.fit_transform(df['text'])

y = df['target']

# train model
model = RandomForestClassifier(n_estimators=200)

model.fit(X,y)

# save model
pickle.dump(model, open("model.pkl","wb"))

# save vectorizer
pickle.dump(tfidf, open("vectorizer.pkl","wb"))

print("Model Saved Successfully")
