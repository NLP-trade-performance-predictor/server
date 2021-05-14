from pymongo import MongoClient
from sklearn import metrics
from os import getenv

labels = ['down', 'same', 'up']

client = MongoClient(getenv('MONGO_CONN'))
db=client.spi


def save(obj):
    r=db.results.insert_one(obj)


def get_metrics():
    results = db.results.find({'gold_class_id': {'$ne': -1}})

    golds = []
    preds = []
    for result in results:
        golds.append(result['gold_class_id'])
        preds.append(result['predicted_class_id'])

    accuracy = metrics.accuracy_score(golds, preds)
    precision = metrics.precision_score(golds, preds, average='macro')
    recall = metrics.recall_score(golds, preds, average='macro')
    f1_score = metrics.f1_score(golds, preds, average='macro')

    return accuracy, precision, recall, f1_score