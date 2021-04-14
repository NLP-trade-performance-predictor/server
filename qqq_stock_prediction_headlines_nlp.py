
import datetime as dt
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import transformers
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

model = BertForSequenceClassification.from_pretrained(
    PRE_TRAINED_MODEL_NAME, num_labels=3)

PATH = '/content/drive/MyDrive/NLP Final Project'

stocks = pd.read_csv(f'{PATH}/stocks.csv', header=None)
articles = pd.read_csv(f'{PATH}/articles.csv', header=None)

stocks.head(10)

articles.head(10)

articles[1] = articles[1].map(lambda article: dt.datetime.strftime(
    dt.datetime.strptime(article, '%a %d %b %Y %H:%M:%S GMT'), '%Y-%m-%d'))

joined_df = articles.set_index(1).join(
    stocks.set_index(0), lsuffix='article', rsuffix='stock')
joined_df = joined_df.dropna()
joined_df = joined_df.rename(
    {0: 'headline', '2article': 'publisher', 1: 'diff', '2stock': 'ratio', 3: 'label'}, axis=1)
joined_df['headline'] = joined_df['headline'].map(
    lambda headline: headline[:headline.rindex(' - ')])
joined_df['label'] = joined_df['label'].astype(int)
joined_df.head(10)

"""# BERT Model"""

df_train, df_test = train_test_split(
    joined_df,
    test_size=0.2,
)
df_train, df_val = train_test_split(
    df_train,
    test_size=0.15,
)

train_encodings = tokenizer(
    df_train['headline'].tolist(), truncation=True, padding=True)
val_encodings = tokenizer(
    df_val['headline'].tolist(), truncation=True, padding=True)
test_encodings = tokenizer(
    df_test['headline'].tolist(), truncation=True, padding=True)


train_dataset = HeadlinesStocksDataset(
    train_encodings, df_train['label'].tolist())
val_dataset = HeadlinesStocksDataset(val_encodings, df_val['label'].tolist())
test_dataset = HeadlinesStocksDataset(
    test_encodings, df_test['label'].tolist())

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

trainer = Trainer(
    # the instantiated 🤗 Transformers model to be trained
    model=model,
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()

trainer.evaluate(test_dataset)

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

correct_predictions = 0

with torch.no_grad():
    for data in test_loader:
        labels = data['labels'].to(device)
        correct_predictions += torch.sum(predicted_labels == labels)

print(correct_predictions.item() / len(test_dataset))
