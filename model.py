from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


PRE_TRAINED_MODEL_NAME = 'ProsusAI/finbert'
MODEL_PATH = './model'

tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=MODEL_PATH)
model.eval()

labels = ['down', 'same', 'up']
labels_tr = [2, 0, 1]

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
    predicted_label_id = labels_tr[predicted_label.squeeze().item()]
    predicted_label_name = labels[predicted_label_id]

    return predicted_label_id, predicted_label_name, prob