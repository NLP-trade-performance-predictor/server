from flask import Flask, jsonify, request
from flask_cors import CORS
from model import interpret, pre_process, predict
from db import save, get_metrics
import datetime


app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def get_prediction():
    headline = request.json['headline']
    input_ids, attention_mask = pre_process([headline])
    predicted_label_id, predicted_label_name, prob = predict(input_ids, attention_mask)

    response = {'date': datetime.datetime.now(),
                'predicted_class_id': predicted_label_id,
                'predicted_class_name': predicted_label_name,
                'gold_class_id': -1,
                'prob': prob}

    save(response)

    response['interpretation'] = interpret(headline)

    del response['_id']
    del response['date']
    del response['gold_class_id']

    return jsonify(response)


@app.route('/metric', methods=['GET'])
def metrics():
    accuracy, precision, recall, f1_score, cm = get_metrics()

    response = {'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'cm': cm.tolist()}

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=False)