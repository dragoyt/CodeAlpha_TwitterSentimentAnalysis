# Twitter Sentiment Analysis Web App

This project demonstrates a **Sentiment Analysis** model built using Python and deployed as a web app using **Streamlit**. It utilizes the **Sentiment140** dataset, which contains 1,600,000 tweets labeled for sentiment (positive or negative).

## Features
- Preprocess and train a sentiment analysis model on the Sentiment140 dataset.
- Visualize metrics such as accuracy and confusion matrix.
- Predict the sentiment (positive or negative) of custom text entered by the user.
- Web-based interactive UI for easy testing and exploration.

---

## Dataset
The dataset is the **Sentiment140 dataset**, containing tweets with the following fields:
- `target`: Sentiment polarity (0 = negative, 4 = positive).
- `text`: Text of the tweet.
- Additional fields like `ids`, `date`, `flag`, and `user` (not used in this project).

You can download the dataset from the [official source](http://help.sentiment140.com/).

---

## Tech Stack
- **Python**: For building the machine learning model.
- **Streamlit**: For creating the web-based interface.
- **Libraries**:
  - `pandas`, `numpy`: For data handling and preprocessing.
  - `scikit-learn`: For machine learning model building and evaluation.
  - `matplotlib`, `seaborn`: For data visualization.

---

## Installation and Setup
### 1. Clone the Repository
```bash
git clone https://github.com/your-username/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
```

### 2. Install Dependencies
Ensure you have Python 3.8 or above installed. Run the following command to install required packages:
```bash
pip install -r requirements.txt
```

### 3. Add the Dataset
Download the Sentiment140 dataset (`train.csv`) and place it in the project directory.

### 4. Run the App
Launch the Streamlit app:
```bash
streamlit run sentiment_app.py
```

The app will open in your default web browser. You can also access it at `http://localhost:8501`.

---

## Usage
1. Upload the **Sentiment140** dataset in the web app.
2. View the data distribution and model evaluation metrics.
3. Enter custom text to predict its sentiment (Positive or Negative).

---

## Screenshots
### Dataset Preview
![Dataset Preview](path-to-dataset-preview-image)

### Confusion Matrix
![Confusion Matrix](path-to-confusion-matrix-image)

### Sentiment Prediction
![Prediction](path-to-prediction-image)

---

## Future Improvements
- Add support for neutral sentiment.
- Implement more advanced models like LSTM or BERT for better accuracy.
- Enable real-time Twitter data analysis using Twitter API.

---

## License
This project is licensed under the [MIT License](LICENSE).
"""
