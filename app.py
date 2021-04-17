from flask import Flask, jsonify, request
from transformers import BertTokenizer, BertForSequenceClassification
from HeadlinesDataset import HeadlinesStocksDataset
import torch


PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
MODEL_PATH = './model'

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

labels = ['down', 'same', 'up']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)


def pre_process(headlines):
    encodings = tokenizer(
        headlines, truncation=True, padding=True)

    data = {key: torch.tensor(val[0])
            for key, val in encodings.items()}

    input_ids = data['input_ids'].unsqueeze(dim=0).to(device)
    attention_mask = data['attention_mask'].unsqueeze(dim=0).to(device)

    return input_ids, attention_mask


def predict(input_ids, attention_mask) -> (torch.tensor, torch.tensor):
    outputs = model(
        input_ids, attention_mask=attention_mask).logits.squeeze()
    y_pred_softmax = torch.log_softmax(outputs, dim=0)
    _, predicted_labels = torch.max(y_pred_softmax, dim=0)

    return predicted_labels, y_pred_softmax


@app.route('/predict', methods=['POST'])
def get_prediction():
    headlines = request.json['headlines']
    input_ids, attention_mask = pre_process(headlines)
    predicted_labels, y_pred_softmax = predict(input_ids, attention_mask)

    label_id = predicted_labels.squeeze().item()
    label = labels[label_id]
    return jsonify({'class_id': str(label_id), 'class_name': label})


if __name__ == '__main__':
    app.run(debug=False)
