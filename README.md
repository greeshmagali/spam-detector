# 📩 Spam Message Detection

A machine learning project that classifies SMS messages as **Spam** or **Not Spam (Ham)** using TF-IDF text vectorization and machine learning.

The project compares multiple machine learning algorithms using **TF-IDF**, selects the best model using **5-fold cross-validation**, tunes its hyperparameters using **GridSearchCV**, and evaluates the final model on an unseen test dataset.

---

## 📌 Project Overview

Spam messages are unwanted messages that may contain advertisements, fake offers, prizes, or suspicious links.

This project builds a machine learning system that learns patterns from previously labeled SMS messages and predicts whether a new message is:

* 🟢 **Not Spam (Ham)**
* 🔴 **Spam**

The project also performs error analysis by saving the messages that the model classified incorrectly.

---

## 🎯 Objectives

* Clean and prepare the SMS dataset
* Convert text messages into numerical features using **TF-IDF**
* Compare multiple machine learning models
* Select the best model using **5-fold cross-validation**
* Tune the selected model using **GridSearchCV**
* Evaluate the final model using:

  * Accuracy
  * Precision
  * Recall
  * F1 Score
* Generate a confusion matrix
* Analyze misclassified messages
* Save the trained model
* Predict new messages using a command-line tool

---

## 📊 Dataset

The project uses the **SMS Spam Collection** dataset.

The dataset contains SMS messages labeled as:

* `ham` → Not Spam
* `spam` → Spam

The main columns used are:

| Column | Description   |
| ------ | ------------- |
| `v1`   | Message label |
| `v2`   | SMS message   |

The unused columns from the original CSV are removed during preprocessing.

---

## 🛠️ Technologies Used

* **Python**
* **NumPy**
* **Pandas**
* **Scikit-learn**
* **Matplotlib**
* **TF-IDF**
* **Logistic Regression**
* **Multinomial Naive Bayes**
* **Random Forest**
* **Cross-Validation**
* **GridSearchCV**

---

## 🔄 Machine Learning Workflow

```text
SMS Dataset
     ↓
Data Cleaning
     ↓
Remove Missing Values
     ↓
Remove Duplicate Rows
     ↓
Convert Labels
     ↓
Train/Test Split
     ↓
TF-IDF Vectorization
     ↓
Compare 3 ML Models
     ↓
5-Fold Cross-Validation
     ↓
Select Best Model
     ↓
Hyperparameter Tuning
     ↓
Final Test Evaluation
     ↓
Save Model
     ↓
Predict New Messages
```

---

## 🤖 Machine Learning Models

Three different algorithms are compared:

### 1. Logistic Regression

A classification algorithm used to predict whether a message belongs to the Spam or Not Spam class.

### 2. Multinomial Naive Bayes

A machine learning algorithm commonly used for text classification.

### 3. Random Forest

An ensemble model that combines multiple decision trees to make predictions.

All three models use the same **TF-IDF text representation**.

---

## 🔤 TF-IDF

TF-IDF stands for **Term Frequency–Inverse Document Frequency**.

It converts text messages into numerical values that machine learning algorithms can understand.

The project uses:

```python
TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)
)
```

This allows the model to use:

* Individual words
* Two-word combinations
* Up to 5,000 features

---

## 🔍 Model Selection

Instead of choosing a model using the test set, the project uses **5-fold cross-validation** on the training data.

The models are compared using **F1 Score**.

The model with the highest mean cross-validation F1 score is selected for further tuning.

This helps avoid selecting a model simply because it performed well on one particular test set.

---

## ⚙️ Hyperparameter Tuning

After selecting the best model, **GridSearchCV** is used to find better hyperparameter values.

The search uses:

* 5-fold cross-validation
* F1 score

The TF-IDF settings and model-specific parameters are tested together.

---

## 📈 Model Evaluation

The final selected model is evaluated on the test dataset using:

### Accuracy

Percentage of total predictions that are correct.

### Precision

Of the messages predicted as Spam, how many were actually Spam?

### Recall

Of the actual Spam messages, how many did the model detect?

### F1 Score

A combined measure of Precision and Recall.

### Confusion Matrix

Shows:

* True Negatives
* False Positives
* False Negatives
* True Positives

---

## 📂 Project Structure

```text
spam_detector/
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
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
```

### 2. Open the project folder

```bash
cd spam_detector
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python train.py
```

This will:

* Train and compare the models
* Perform cross-validation
* Tune the selected model
* Evaluate it on the test set
* Generate result files
* Save the trained model

The trained model will be saved as:

```text
models/spam_model.pkl
```

---

## 🧪 Test a New Message

Run:

```bash
python predict.py
```

Then enter a message:

```text
Enter a message: Congratulations! You won a free prize!
```

Example output:

```text
========================================
Message: Congratulations! You won a free prize!
Prediction: SPAM
Confidence: 98.45 %
========================================
```

You can also provide the message directly from the command line:

```bash
python predict.py "Congratulations! You won a free prize!"
```

---

## 🔎 Error Analysis

The project saves incorrectly classified messages in:

```text
results/misclassified_messages.csv
```

This allows the model's mistakes to be inspected and analyzed.

For example:

```text
message | actual | predicted
```

This helps identify the types of messages that are difficult for the model to classify.

---

## 📊 Results

After running `train.py`, the following files are generated:

### Confusion Matrix

```text
results/confusion_matrix.png
```

Shows how many messages were correctly and incorrectly classified.

### Model Comparison

```text
results/model_comparison.png
```

Shows the mean cross-validation F1 score of the three models.

### Metrics

```text
results/metrics.csv
```

Contains the final test-set:

* Accuracy
* Precision
* Recall
* F1 Score

### Misclassified Messages

```text
results/misclassified_messages.csv
```

Contains the messages that were incorrectly classified by the final model.

---

## 💡 Key Learning Outcomes

Through this project, I practiced:

* Text preprocessing
* TF-IDF vectorization
* Binary classification
* Logistic Regression
* Naive Bayes
* Random Forest
* Train/test splitting
* 5-fold cross-validation
* Hyperparameter tuning
* Pipeline-based machine learning
* Model evaluation
* Confusion matrix analysis
* Error analysis
* Model serialization
* Building a command-line prediction tool

---

## 🔮 Future Improvements

Possible future improvements include:

* Testing additional NLP preprocessing techniques
* Trying additional machine learning algorithms
* Handling class imbalance with additional techniques
* Adding a web interface
* Deploying the trained model as an API
* Monitoring model performance on new messages

---

## 👩‍💻 Author

**Greeshma Gali**

B.Tech – Computer Science and Engineering (AI & ML)

Interested in **Artificial Intelligence, Machine Learning, Python, and Backend Development**.
