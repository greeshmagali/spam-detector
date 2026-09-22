# ============================================================
# SPAM MESSAGE DETECTION
# ============================================================

import os #os helps Python work with folders and files on your computer.
import pickle #pickle is used to save our trained ML model.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# ============================================================
# 1. CREATE REQUIRED FOLDERS
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

data = pd.read_csv(
    "data/spam.csv",
    encoding="latin-1"
)

print("Original dataset shape:")
print(data.shape)


# ============================================================
# 3. SELECT REQUIRED COLUMNS
# ============================================================

data = data[["v1", "v2"]]

data.columns = [
    "label",
    "message"
]


# ============================================================
# 4. CHECK MISSING VALUES
# ============================================================

print("\nMissing values:")
print(data.isnull().sum())


# Remove missing rows

data = data.dropna()


# ============================================================
# 5. REMOVE DUPLICATES
# ============================================================

print("\nDuplicate rows:")
print(data.duplicated().sum())

data = data.drop_duplicates()

print("\nDataset shape after removing duplicates:")
print(data.shape)


# ============================================================
# 6. CONVERT LABELS
# ============================================================

# ham  -> 0
# spam -> 1

data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})


# ============================================================
# 7. BASIC DATA ANALYSIS
# ============================================================

print("\nClass distribution:")
print(data["label"].value_counts())


data["message_length"] = data["message"].apply(len)

print("\nAverage message length:")
print(
    data.groupby("label")["message_length"].mean()
)


# ============================================================
# 8. SPLIT FEATURES AND TARGET
# ============================================================

X = data["message"]

y = data["label"]


# ============================================================
# 9. TRAIN / TEST SPLIT
# ============================================================

# The test set will remain untouched until the final evaluation.

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 10. CREATE ML PIPELINES
# ============================================================

# TF-IDF is inside the Pipeline.
#
# This is important because during cross-validation,
# TF-IDF is fitted separately on each training fold.
#
# This prevents data leakage.

models = {

    "Logistic Regression": Pipeline([

        (
            "tfidf", #The "tfidf" is simply a name for this step. 

            TfidfVectorizer( #TF-IDF is used to convert text messages into numbers so that a machine-learning model can understand them.

                lowercase=True, #Make all letters small, so FREE and free count as the same word.

                stop_words="english", #Throw away boring words like the, is, and, a.

                max_features=5000, #Only remember the 5,000 most useful words.

                ngram_range=(1, 2) #Only remember the 5,000 most useful words.

            )
        ),

        (
            "model", #Step 2 gets the name "model"

            LogisticRegression(
                max_iter=1000    #Step 2 is the Logistic Regression guesser. max_iter=1000 lets it practice up to 1000 times to learn.
            )
        )

    ]),


    "Naive Bayes": Pipeline([

        (
            "tfidf", #Step 1 gets the name "tfidf"

            TfidfVectorizer(

                lowercase=True, #Make all letters small, so FREE and free count as the same word.

                stop_words="english", #Throw away boring words like the, is, and, a.

                max_features=5000, #Only remember the 5,000 most useful words.

                ngram_range=(1, 2)  #Only remember the 5,000 most useful words.

            )
        ),

        (
            "model", #Step 2 gets the name "model"

            MultinomialNB()
        )

    ]),


    "Random Forest": Pipeline([

        (
            "tfidf", #Step 1 gets the name "tfidf"

            TfidfVectorizer(

                lowercase=True, #Make all letters small, so FREE and free count as the same word.

                stop_words="english", #Throw away boring words like the, is, and, a.

                max_features=5000, #Only remember the 5,000 most useful words.

                ngram_range=(1, 2) #Only remember the 5,000 most useful words.

            )
        ),

        (
            "model", #Step 2 gets the name "model"

            RandomForestClassifier(

                n_estimators=200,   #Step 2 is the Random Forest guesser. n_estimators=200 means 200 small trees vote. 
 
                random_state=42  #random_state=42 keeps results the same each time.

            )
        )

    ])

}


# ============================================================
# 11. CROSS-VALIDATION MODEL SELECTION                               purpose:   Which model performs best?
# ============================================================

print("\n")
print("=" * 60)
print("5-FOLD CROSS-VALIDATION")
print("=" * 60)


cv_results = [] #Make an empty list     Initially: cv_results = []  


#Later we will put results inside it:
'''
cv_results = [
    {"Model": "Logistic Regression", "Mean CV F1": 0.95},
    {"Model": "Naive Bayes", "Mean CV F1": 0.94},
    {"Model": "Random Forest", "Mean CV F1": 0.93}
]
'''

for name, pipeline in models.items():

    scores = cross_val_score(

        pipeline,  #This is the model we are currently testing.                          
                                                                                              #scores =[0.945, 0.952, 0.938, 0.961, 0.949]
        X_train, #This is the training messages.

        y_train,  #This contains the correct answers for those messages.

        cv=5,  #So we are doing 5-fold cross-validation.

        scoring="f1", #This tells Scikit-learn: Use F1 score to judge how well the model performed.

        n_jobs=-1 #This tells Scikit-learn: Use all available CPU processing power to do the calculation faster. It mainly helps the cross-validation run faster.

    )


    mean_score = scores.mean()


    print("\nModel:", name)

    print(
        "F1 scores:",
        np.round(scores, 4)
    )

    print(
        "Mean F1:",
        round(mean_score, 4)
    )


    cv_results.append({

        "Model": name,

        "Mean CV F1": mean_score

    })


cv_df = pd.DataFrame(cv_results)


# ============================================================
# 12. SELECT BEST MODEL
# ============================================================

best_model_name = cv_df.loc[
    cv_df["Mean CV F1"].idxmax(), #idxmax() means: Find the position/index where the biggest value is located.
    "Model"
]


print("\n")
print("=" * 60)
print("BEST MODEL")
print("=" * 60)

print(
    "Selected model:",
    best_model_name
)


best_pipeline = models[
    best_model_name
]


# ============================================================
# 13. HYPERPARAMETER TUNING
# ============================================================

print("\n")
print("=" * 60)
print("HYPERPARAMETER TUNING")   #Now this code asks: What settings should I use inside that chosen model to make it work better?
print("=" * 60)


if best_model_name == "Logistic Regression":

    param_grid = {   #param_grid is basically a list of different settings that we want Python to try. ---"Try these different settings and find which combination works best."

        "tfidf__max_features": [ #We want to try: 3000 words, 5000 words
            3000,
            5000
        ],

        "tfidf__ngram_range": [   #Use: single words + two-word combinations
            (1, 1),
            (1, 2)
        ],

        "model__C": [  #C controls how strongly Logistic Regression is regularized.
            0.1,
            1,
            10
        ]

    }


elif best_model_name == "Naive Bayes":

    param_grid = {

        "tfidf__max_features": [
            3000,
            5000
        ],

        "tfidf__ngram_range": [
            (1, 1),
            (1, 2)
        ],

        "model__alpha": [ #alpha is a Naive Bayes parameter used for smoothing.It helps the model handle words that are rare or haven't appeared in some training examples
            0.1,
            0.5,
            1.0
        ]

    }


else:

    param_grid = {

        "tfidf__max_features": [
            3000,
            5000
        ],

        "tfidf__ngram_range": [
            (1, 1),
            (1, 2)
        ],

        "model__n_estimators": [  #How many decision trees should the Random Forest use?
            100,
            200
        ],

        "model__max_depth": [  #This controls how deep each decision tree can grow.
            None,
            20
        ]

    }


grid_search = GridSearchCV(   # GridSearchCV means: Try different parameter combinations and find the one that performs best.

    best_pipeline,

    param_grid,

    cv=5,

    scoring="f1",

    n_jobs=-1

)


grid_search.fit(
    X_train,
    y_train      #Start trying all the parameter combinations using the training data.
)


print("\nBest parameters:")

print(
    grid_search.best_params_   #best_params_ tells you: Which combination of settings gave the best CV F1?
)


'''
For example, it might print:

{
 'model__C': 1,
 'tfidf__max_features': 5000,
 'tfidf__ngram_range': (1, 2)
}
'''

print("\nBest CV F1:")

print(
    round(
        grid_search.best_score_,  #best_score_ means: The highest average F1 score found during the grid search.
        4
    )
)


'''
For example:
Best CV F1:
0.9562

The round(..., 4) just keeps 4 decimal places.
'''


# ============================================================
# 14. FINAL MODEL
# ============================================================

final_model = grid_search.best_estimator_   # What is best_estimator_?   Give me the actual Pipeline that uses those best settings.


# ============================================================
# 15. FINAL TEST SET EVALUATION
# ============================================================

# IMPORTANT:
# The test set is used ONLY here.

y_pred = final_model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    y_pred
)


precision = precision_score(
    y_test,
    y_pred
)


recall = recall_score(
    y_test,
    y_pred
)


f1 = f1_score(
    y_test,
    y_pred
)


print("\n")
print("=" * 60)
print("FINAL TEST SET RESULTS")
print("=" * 60)


print(
    "Accuracy :",
    round(accuracy, 4)
)

print(
    "Precision:",
    round(precision, 4)
)

print(
    "Recall   :",
    round(recall, 4)
)

print(
    "F1 Score :",
    round(f1, 4)
)


# ============================================================
# 16. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")  #It compares: y_test → The REAL answers, y_pred → The MODEL'S answers

print(

    classification_report(

        y_test,

        y_pred,

        target_names=[
            "Ham",
            "Spam"
        ]

    )

)


# ============================================================
# 17. SAVE METRICS
# ============================================================

metrics = pd.DataFrame({

    "Metric": [

        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"

    ],

    "Score": [

        accuracy,
        precision,
        recall,
        f1

    ]

})


metrics.to_csv(
    "results/metrics.csv",
    index=False
)


# ============================================================
# 18. SAVE CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test, #actual
    y_pred #predicted
)


print("\nConfusion Matrix:")

print(cm)


disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=[
        "Ham",
        "Spam"
    ]

)


disp.plot()


plt.title(
    "Spam Detection - Confusion Matrix"
)


plt.tight_layout()


plt.savefig(
    "results/confusion_matrix.png",
    dpi=300
)


plt.show()


# ============================================================
# 19. SAVE MODEL COMPARISON CHART
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plt.bar(

    cv_df["Model"],

    cv_df["Mean CV F1"]

)


plt.title(
    "Model Comparison Using 5-Fold Cross-Validation"
)


plt.ylabel(
    "Mean F1 Score"
)


plt.ylim(
    0,
    1
)


plt.xticks(
    rotation=15
)


plt.tight_layout()


plt.savefig(
    "results/model_comparison.png",
    dpi=300
)


plt.show()


# ============================================================
# 20. ERROR ANALYSIS
# ============================================================

test_results = pd.DataFrame({

    "message": X_test,

    "actual": y_test,

    "predicted": y_pred

})


errors = test_results[
    test_results["actual"]
    !=
    test_results["predicted"]
]


print("\n")
print("=" * 60)
print("MISCLASSIFIED MESSAGES")
print("=" * 60)


print(
    errors.head(10)
)


errors.to_csv(
    "results/misclassified_messages.csv",
    index=False
)


# ============================================================
# 21. SAVE TRAINED MODEL
# ============================================================

with open("models/spam_model.pkl","wb") as file:

    pickle.dump( final_model,file)


print("\n")
print("=" * 60)
print("PROJECT COMPLETED")
print("=" * 60)


print(
    "\nModel saved to:"
)

print(
    "models/spam_model.pkl"
)


print(
    "\nMetrics saved to:"
)

print(
    "results/metrics.csv"
)


print(
    "\nCharts saved to:"
)

print(
    "results/"
)