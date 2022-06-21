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

model = joblib.load("assets/models/TfIdfModel.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.get_json()
    title = input_data.get("title")
    processed_title = text_prepare(title)
    prediction = model.predict([processed_title])
    result = model._mlb.inverse_transform(prediction)

    return jsonify(
        {"result": result, "classifier": "logistic classifier", "title": title}
    )


if __name__ == "__main__":
    app.run("0.0.0.0", port=5000, debug=False)  # nosec B104
