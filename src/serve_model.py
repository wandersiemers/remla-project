"""
Flask API of the SMS Spam detection model model.
"""
import joblib
from flask import Flask, jsonify, request
from flasgger import Swagger

from preprocessing import text_prepare
from prometheus_flask_exporter import PrometheusMetrics


app = Flask(__name__)
swagger = Swagger(app)
PrometheusMetrics(app)

vectorizer = joblib.load('assets/outputs/tfidf-vectorizer.joblib')
model = joblib.load('assets/models/classifier_tfidf.joblib')
mlb_classifier = joblib.load('assets/models/mlb_classifier.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict whether an SMS is Spam.
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: message to be classified.
          required: True
          schema:
            type: object
            required: sms
            properties:
                sms:
                    type: string
                    example: This is an example of an SMS.
    responses:
      200:
        description: "The result of the classification: 'spam' or 'ham'."
    """
    input_data = request.get_json()
    title = input_data.get('title')
    processed_title = text_prepare(title)
    prediction = model.predict(vectorizer.transform([processed_title]))
    result = mlb_classifier.inverse_transform(prediction)

    return jsonify({
        "result": result,
        "classifier": "logistic classifier",
        "title": title
    })


@app.route('/dumbpredict', methods=['POST'])
def dumb_predict():
    """
    Predict whether a given SMS is Spam or Ham (dumb model: always predicts 'ham').
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: message to be classified.
          required: True
          schema:
            type: object
            required: sms
            properties:
                sms:
                    type: string
                    example: This is an example of an SMS.
    responses:
      200:
        description: "The result of the classification: 'spam' or 'ham'."
    """
    input_data = request.get_json()
    sms = input_data.get('sms')

    return jsonify({
        "result": "Spam",
        "classifier": "decision tree",
        "sms": sms
    })


if __name__ == '__main__':
    clf = joblib.load('assets/models/classifier_tfidf.joblib')
    app.run('0.0.0.0', port=5000, debug=False)
