# 📧 Spam Message Classifier

A **Machine Learning based Spam Detection Web App** built using **Natural Language Processing (NLP)** and deployed using **Streamlit**.

The application predicts whether a message is **Spam or Ham (Not Spam)** using a trained classification model.

---

## 🚀 Project Overview

Spam messages are a common problem in email and SMS communication.
This project uses **machine learning and text processing techniques** to automatically detect spam messages.

The model processes text messages, converts them into numerical features using **TF-IDF vectorization**, and then predicts whether the message is spam or not.

---

## 🧠 Machine Learning Pipeline

```
Text Message
     ↓
Text Preprocessing
(lowercase, remove punctuation, remove stopwords, stemming)
     ↓
TF-IDF Vectorization
     ↓
SVM Classification Model
     ↓
Prediction: Spam / Ham
```

---

## 📊 Model Performance

| Model                            | Accuracy        |
| -------------------------------- | --------------- |
| Naive Bayes                      | ~95%            |
| Logistic Regression              | ~96%            |
| **Random Forest Classifier**     | **~96% (Best)** |
|  Support Vector Machine (SVM)    | ~94%            |

The **SVM model** was selected as the final model due to its superior performance.

---

## 🛠️ Technologies Used

* Python
* Scikit-learn
* Natural Language Processing (NLP)
* TF-IDF Vectorization
* Streamlit
* Pandas
* NLTK

---

## 📁 Project Structure

```
spam-classifier
│
├── app.py              # Streamlit application
├── train_model.py      # Model training script
├── model.pkl           # Saved trained model
├── vectorizer.pkl      # Saved TF-IDF vectorizer
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

---

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/yourusername/spam-classifier.git
```

Navigate to the project folder:

```
cd spam-classifier
```

Install required libraries:

```
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the Streamlit application:

```
streamlit run app.py
```

The app will open in your browser.

---

## 🌐 Live Demo

You can access the deployed application here:

```
(https://spamclassifier-dworzksaiga23bufgid27d.streamlit.app/)
```

---

## ✨ Features

* Detects **Spam vs Ham messages**
* Real-time text classification
* Interactive **Streamlit web interface**
* NLP preprocessing pipeline
* TF-IDF feature extraction
* Machine learning classification model

---

## 📌 Example Spam Messages

Try these examples in the app:

```
WIN a free iPhone now! Click the link to claim your prize.
```

```
Congratulations! You have been selected for a $1000 cash reward.
```

---

## 📈 Future Improvements

* Add deep learning models (LSTM / BERT)
* Add dataset visualizations
* Deploy using Docker
* Improve UI with advanced Streamlit components

---

## 👨‍💻 Author

**Saksham**

Machine Learning Enthusiast
Passionate about AI, NLP, and building intelligent applications.

---

⭐ If you like this project, please consider **starring the repository**!
