import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load and preprocess dataset
@st.cache
def load_data(file_path, sample_size=5000):
    data = pd.read_csv(file_path, encoding='latin-1', header=None)
    data.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
    sample_data = data.sample(sample_size, random_state=42).reset_index(drop=True)
    sample_data = sample_data[['target', 'text']]
    sample_data['target'] = sample_data['target'].map({0: 0, 4: 1})
    sample_data.drop_duplicates(subset='text', inplace=True)
    return sample_data


# Prepare model
@st.cache(allow_output_mutation=True)
def train_model(data):
    X = data['text']
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    return model, vectorizer, X_test_tfidf, y_test

# App layout
st.title("Sentiment Analysis Web App")
st.write("Upload the Sentiment140 dataset and test the model on custom text.")

# File upload
uploaded_file = st.file_uploader("Upload Sentiment140 dataset (CSV)", type="csv")
if uploaded_file is not None:
    st.success("File uploaded successfully!")
    sample_data = load_data(uploaded_file)

    # Display dataset
    st.write("### Dataset Sample")
    st.dataframe(sample_data.head())

    # Train model
    st.write("### Training the Model")
    model, vectorizer, X_test_tfidf, y_test = train_model(sample_data)
    y_pred = model.predict(X_test_tfidf)

    # Visualize metrics
    st.write("### Model Metrics")
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Accuracy:** {accuracy:.2f}")

    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    st.pyplot(plt)

    # Input for prediction
    st.write("### Test the Model")
    user_input = st.text_area("Enter a sentence for sentiment analysis:")
    if user_input:
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        st.write(f"**Prediction:** {sentiment}")