from transformers import BertTokenizer, BertForSequenceClassification
import torch


PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
MODEL_PATH = './model'

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

labels = ['down', 'same', 'up']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pre_process(headline):
    encodings = tokenizer(
        headline, truncation=True, padding=True)

    data = {key: torch.tensor(val[0])
            for key, val in encodings.items()}

    input_ids = data['input_ids'].unsqueeze(dim=0).to(device)
    attention_mask = data['attention_mask'].unsqueeze(dim=0).to(device)

    return input_ids, attention_mask


def predict(input_ids, attention_mask):
    outputs = model(
        input_ids, attention_mask=attention_mask).logits.squeeze()

    prob, predicted_label = torch.max(torch.softmax(outputs, dim=0), dim=0)

    prob = prob.squeeze().item()
    predicted_label_id = predicted_label.squeeze().item()
    predicted_label_name = labels[predicted_label]

    return predicted_label_id, predicted_label_name, prob