import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

with open("models/spam_model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return "Spam Message Detection API is running!"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    message = data["message"]

    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]

    if prediction == 1:
        result = "SPAM"
        confidence = probabilities[1]
    else:
        result = "NOT SPAM"
        confidence = probabilities[0]

    return jsonify({
        "message": message,
        "prediction": result,
        "confidence": round(confidence * 100, 2)
    })


if __name__ == "__main__":
    app.run()