from flask import Flask, jsonify
from transformers import BertTokenizer
from fuckingshit HeadlinesStocksDataset

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    return jsonify({'class_id': '0', 'class_name': 'down'})


def pre_process(headlines):

    encodings = tokenizer(
        headlines, truncation=True, padding=True)

    data = {key: torch.tensor(val[idx])
            for key, val in encodings.items()}

    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)

    return input_ids, attention_mask


def predict(input_ids, attention_mask):
    outputs = model(
        input_ids, attention_mask=attention_mask).logits.squeeze()
    # for accuracy calculation
    y_pred_softmax = torch.log_softmax(outputs, dim=1)
    _, predicted_labels = torch.max(y_pred_softmax, dim=1)

    return predicted_labels, y_pred_softmax


@app.route('/', methods=['POST'])
def get_prediction(headlines):
    input_ids, attention_mask = pre_process(headlines)
    predicted_labels, y_pred_softmax = predict(input_ids, attention_mask)
    return headlines


if __name__ == '__main__':
    app.run()
