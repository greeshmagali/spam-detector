# 📩 SMS Spam Message Detection

A machine learning project that classifies SMS messages as **Spam** or **Not Spam** using **TF-IDF text vectorization** and multiple machine learning algorithms.

The trained model is also deployed as a **Flask REST API on Render**, allowing predictions through a public API endpoint.

## 🚀 Live API

**Base URL:**
https://spam-detector-um9j.onrender.com/

The API is publicly deployed and can be used to make spam predictions.

### Prediction Endpoint

```text
POST /predict
```

Example request:

```json
{
    "message": "Congratulations! You won a FREE cash prize! Click now to claim your reward!"
}
```

Example response:

```json
{
    "message": "Congratulations! You won a FREE cash prize! Click now to claim your reward!",
    "prediction": "SPAM",
    "confidence": 72.0
}
```

## 📌 Project Overview

SMS spam messages are unwanted messages that may contain advertisements, scams, fake offers, or suspicious links.

This project uses machine learning to automatically classify messages into:

* **SPAM**
* **NOT SPAM**

The project includes data preprocessing, TF-IDF feature extraction, model comparison, cross-validation, hyperparameter tuning, evaluation, error analysis, and deployment.

## 🎯 Objectives

* Clean and prepare the SMS dataset
* Convert text messages into numerical features using TF-IDF
* Train multiple machine learning models
* Compare models using 5-fold cross-validation
* Select the best model based on F1 score
* Tune the selected model using GridSearchCV
* Evaluate the final model on an unseen test set
* Analyze misclassified messages
* Save the trained model
* Build a Flask REST API
* Deploy the API using Render

## 📊 Dataset

The project uses the **SMS Spam Collection dataset** containing SMS messages labeled as:

* `ham` → Not Spam
* `spam` → Spam

Dataset columns used:

| Column | Description   |
| ------ | ------------- |
| `v1`   | Message label |
| `v2`   | SMS message   |

The dataset contains approximately **5,500+ SMS messages**.

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Flask
* Gunicorn
* Render

## 🔄 Machine Learning Workflow

```text
SMS Dataset
     ↓
Data Cleaning
     ↓
Remove Duplicates / Missing Values
     ↓
Train-Test Split
     ↓
TF-IDF Vectorization
     ↓
5-Fold Cross-Validation
     ↓
Model Comparison
     ↓
Select Best Model
     ↓
GridSearchCV Hyperparameter Tuning
     ↓
Final Test Evaluation
     ↓
Save Trained Model
     ↓
Flask REST API
     ↓
Render Deployment
```

## 🤖 Machine Learning Models

Three classification algorithms were compared:

### 1. Logistic Regression

Used as a strong baseline model for text classification.

### 2. Multinomial Naive Bayes

A commonly used algorithm for text classification problems.

### 3. Random Forest

An ensemble learning algorithm used to compare performance with the other models.

## 📝 TF-IDF

**TF-IDF (Term Frequency-Inverse Document Frequency)** converts text messages into numerical feature vectors.

The project uses:

* Lowercase conversion
* English stop-word removal
* Unigrams and bigrams
* Maximum feature limits

TF-IDF is applied inside a Scikit-learn Pipeline to avoid data leakage during cross-validation.

## 🔍 Model Selection

The models are compared using **5-fold cross-validation** with **F1 score**.

The model with the highest mean cross-validation F1 score is selected for further hyperparameter tuning.

This prevents choosing a model simply because of its performance on the final test set.

## ⚙️ Hyperparameter Tuning

After selecting the best model, **GridSearchCV** is used to find suitable hyperparameters.

Examples include:

* Logistic Regression → `C`
* Naive Bayes → `alpha`
* Random Forest → `n_estimators`, `max_depth`
* TF-IDF → `max_features`, `ngram_range`

## 📈 Model Evaluation

The final model is evaluated on an unseen test set using:

* Accuracy
* Precision
* Recall
* F1 Score
* Classification Report
* Confusion Matrix

The project also saves:

```text
results/
├── confusion_matrix.png
├── model_comparison.png
├── metrics.csv
└── misclassified_messages.csv
```

## 🔎 Error Analysis

Misclassified messages are saved in:

```text
results/misclassified_messages.csv
```

This allows incorrect predictions to be inspected and analyzed.

## 🌐 Flask API

The trained Scikit-learn pipeline is saved as:

```text
models/spam_model.pkl
```

Flask loads the saved model and provides a REST API endpoint:

```text
POST /predict
```

The API accepts a message and returns:

* Message
* Prediction
* Confidence

### Example

Request:

```json
{
    "message": "Hey, are you coming to college tomorrow?"
}
```

Response:

```json
{
    "message": "Hey, are you coming to college tomorrow?",
    "prediction": "NOT SPAM",
    "confidence": 100.0
}
```

## ☁️ Deployment

The Flask API is deployed on **Render**.

### Deployment Stack

```text
Scikit-learn Model
        ↓
Flask REST API
        ↓
Gunicorn
        ↓
Render
        ↓
Public API
```

Live application:

https://spam-detector-um9j.onrender.com/

## 📁 Project Structure

```text
spam-detector/
│
├── data/
│   └── spam.csv
│
├── models/
│   └── spam_model.pkl
│
├── results/
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   ├── metrics.csv
│   └── misclassified_messages.csv
│
├── train.py
├── predict.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ▶️ Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/greeshmagali/spam-detector.git
cd spam-detector
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python train.py
```

This generates the trained model and evaluation results.

### 4. Run the prediction script

```bash
python predict.py
```

Enter an SMS message when prompted.

### 5. Run the Flask API

```bash
python app.py
```

The local API will run at:

```text
http://127.0.0.1:5000
```

## 🧪 API Testing

Example PowerShell request:

```powershell
Invoke-RestMethod -Uri "https://spam-detector-um9j.onrender.com/predict" `
-Method Post `
-ContentType "application/json" `
-Body '{"message":"Congratulations! You won a FREE cash prize! Click now to claim your reward!"}'
```

Example result:

```text
Prediction: SPAM
Confidence: 72%
```

Another tested example:

```text
Message: Hey, are you coming to college tomorrow?

Prediction: NOT SPAM
Confidence: 100%
```

## 📚 What I Learned

Through this project, I practiced:

* Data cleaning with Pandas
* Text feature extraction using TF-IDF
* Classification algorithms
* Train-test splitting
* Stratified sampling
* 5-fold cross-validation
* Model selection using F1 score
* Hyperparameter tuning with GridSearchCV
* Classification metrics
* Confusion matrix
* Error analysis
* Saving and loading ML models
* Building REST APIs with Flask
* Deploying ML applications with Render
* Testing APIs using PowerShell

## 🔮 Future Improvements

* Add a web-based user interface
* Improve handling of ambiguous messages
* Experiment with additional text features
* Try advanced NLP/deep learning approaches
* Add automated API testing
* Monitor model performance after deployment

## 👩‍💻 Author

**Greeshma Gali**

B.Tech – Computer Science and Engineering (AI & ML)

GitHub:
https://github.com/greeshmagali
