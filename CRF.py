import torch
from datasets import load_dataset
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from collections import defaultdict

PAD = "PAD"
UNK = "UNK"
DIM_EMBEDDING = 100
BATCH_SIZE = 32
EPOCHS = 10

def read_conll_file(dataset, id2label):
    data = []
    for example in dataset:
        words = example['tokens']
        tags = [id2label[tag] for tag in example['ner_tags']]
        data.append((words, tags))
    return data

def build_vocab(data):
    word_vocab = defaultdict(lambda: len(word_vocab))
    tag_vocab = defaultdict(lambda: len(tag_vocab))
    word_vocab[PAD]
    word_vocab[UNK]
    for sentence, tags in data:
        for word in sentence:
            word_vocab[word]
        for tag in tags:
            tag_vocab[tag]
    return dict(word_vocab), dict(tag_vocab)

def extract_features(sentence):
    return [{'word': word} for word in sentence]

def extract_labels(labels):
    return labels

all_data = load_dataset('conll2003')
label_list = all_data['train'].features['ner_tags'].feature.names
id2label = {i: label for i, label in enumerate(label_list)}

train_data = read_conll_file(all_data['train'], id2label)
dev_data = read_conll_file(all_data['validation'], id2label)

word_vocab, tag_vocab = build_vocab(train_data)
tagset_size = len(tag_vocab)

X_train = [extract_features(sentence) for sentence, _ in train_data]
y_train = [extract_labels(tags) for _, tags in train_data]
X_dev = [extract_features(sentence) for sentence, _ in dev_data]
y_dev = [extract_labels(tags) for _, tags in dev_data]


crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

y_pred = crf.predict(X_dev)
precision = metrics.flat_precision_score(y_dev, y_pred, average='weighted', labels=list(tag_vocab.keys()))
recall = metrics.flat_recall_score(y_dev, y_pred, average='weighted', labels=list(tag_vocab.keys()))
f1 = metrics.flat_f1_score(y_dev, y_pred, average='weighted', labels=list(tag_vocab.keys()))

print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")
