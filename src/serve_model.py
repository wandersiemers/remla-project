"""
Flask API of the SMS Spam detection model model.
"""
import joblib
from flask import Flask, jsonify, request
from flasgger import Swagger

from preprocessing import text_prepare
from model.tf_idf import tfidf_features

app = Flask(__name__)
swagger = Swagger(app)


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
    print(processed_title)
    features, _, _, _ = tfidf_features(processed_title, [], [])
    model = joblib.load('assets/models/classifier_tfidf.joblib')
    prediction = model.predict(features)
    print(prediction)

    return jsonify({
        "result": prediction,
        "classifier": "decision tree",
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
    app.run(port=8080, debug=True)
