"""
Simple Flask API for predicting Stack Overflow tags from titles.
"""
import joblib
from flasgger import Swagger
from flask import Flask, jsonify, request
from prometheus_flask_exporter import PrometheusMetrics

from remla.data.pre_processing import text_prepare

app = Flask(__name__)
Swagger(app)
PrometheusMetrics(app)

vectorizer = joblib.load("assets/outputs/tfidf-vectorizer.joblib")
model = joblib.load("assets/models/classifier_tfidf.joblib")
mlb_classifier = joblib.load("assets/models/mlb_classifier.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.get_json()
    title = input_data.get("title")
    processed_title = text_prepare(title)
    prediction = model.predict(vectorizer.transform([processed_title]))
    result = mlb_classifier.inverse_transform(prediction)

    return jsonify(
        {"result": result, "classifier": "logistic classifier", "title": title}
    )


if __name__ == "__main__":
    app.run("0.0.0.0", port=5000, debug=False)
