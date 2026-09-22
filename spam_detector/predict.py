# ============================================================
# SPAM MESSAGE PREDICTOR
# ============================================================

import pickle
import sys #Import Python's sys module so we can work with command-line inputs.


# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================

with open("models/spam_model.pkl","rb") as file:
   model = pickle.load(file)


# ============================================================
# 2. GET MESSAGE
# ============================================================

if len(sys.argv) > 1: #sys.argv is a list containing the things you typed in the terminal when running Python.
    # len(sys.argv) means: Count how many items are inside sys.argv.

    message = " ".join(
        sys.argv[1:]
    )

else:

    message = input(
        "Enter a message: "
    )


# ============================================================
# 3. PREDICT
# ============================================================

prediction = model.predict(
    [message]
)[0]


probabilities = model.predict_proba(
    [message]
)[0]


# ============================================================
# 4. DISPLAY RESULT
# ============================================================

if prediction == 1:

    result = "SPAM"

    confidence = probabilities[1]

else:

    result = "NOT SPAM"

    confidence = probabilities[0]


print("\n")
print("=" * 40)

print(
    "Message:",
    message
)

print(
    "Prediction:",
    result
)

print(
    "Confidence:",
    round(
        confidence * 100,
        2
    ),
    "%"
)

print("=" * 40)