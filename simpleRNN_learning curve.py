import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from torch import nn, optim
import torch

PAD = "PAD"
UNK = "UNK"
DIM_EMBEDDING = 100
RNN_HIDDEN = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.01
EPOCHS = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def read_conll_file(dataset, id2label):
    data = []
    for example in dataset:
        words = example['tokens']
        tags = [id2label[tag] for tag in example['ner_tags']]
        data.append((words, tags))
    return data


class NERDataset(Dataset):
    def __init__(self, data, word_vocab, tag_vocab):
        self.data = data
        self.word_vocab = word_vocab
        self.tag_vocab = tag_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words, tags = self.data[idx]
        word_indices = [self.word_vocab.get(word, self.word_vocab[UNK]) for word in words]
        tag_indices = [self.tag_vocab[tag] for tag in tags]
        return torch.tensor(word_indices, dtype=torch.long), torch.tensor(tag_indices, dtype=torch.long)

def build_vocab(data):
    word_vocab = defaultdict(lambda: len(word_vocab))
    tag_vocab = defaultdict(lambda: len(tag_vocab))
    # Initialize vocab with PAD and UNK tokens
    word_vocab[PAD]
    word_vocab[UNK]
    for sentence, tags in data:
        for word in sentence:
            word_vocab[word]
        for tag in tags:
            tag_vocab[tag]
    return dict(word_vocab), dict(tag_vocab)

def collate_fn(batch):
    word_seqs, tag_seqs = zip(*batch)
    word_seqs_padded = pad_sequence(word_seqs, batch_first=True, padding_value=0)
    tag_seqs_padded = pad_sequence(tag_seqs, batch_first=True, padding_value=0)
    return word_seqs_padded, tag_seqs_padded

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, tagset_size):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, DIM_EMBEDDING)
        self.rnn = nn.RNN(DIM_EMBEDDING, RNN_HIDDEN, batch_first=True)
        self.hidden2tag = nn.Linear(RNN_HIDDEN, tagset_size)

    def forward(self, sentences):
        embeds = self.embedding(sentences)
        rnn_out, _ = self.rnn(embeds)
        tag_space = self.hidden2tag(rnn_out)
        tag_scores = torch.log_softmax(tag_space, dim=2)
        return tag_scores

def create_subsets(data, fractions=[0.2, 0.4, 0.6, 0.8, 1.0]):
    subsets = []
    n = len(data)
    for frac in fractions:
        subset_size = int(n * frac)
        indices = np.random.choice(range(n), subset_size, replace=False)
        subset = [data[i] for i in indices]
        subsets.append(subset)
    return subsets

def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for words, tags in data_loader:
            outputs = model(words)
            preds = outputs.argmax(dim=2)
            all_preds.extend(preds.view(-1).tolist())
            all_labels.extend(tags.view(-1).tolist())
    return f1_score(all_labels, all_preds, average='weighted')


all_data_conll = load_dataset('conll2003')
all_data_wnut = load_dataset('wnut_17')

label_list_conll = all_data_conll['train'].features['ner_tags'].feature.names
id2label_conll = {i: label for i, label in enumerate(label_list_conll)}

label_list_wnut = all_data_wnut['train'].features['ner_tags'].feature.names
id2label_wnut = {i: label for i, label in enumerate(label_list_wnut)}

train_data_conll = read_conll_file(all_data_conll['train'], id2label_conll)
dev_data_conll = read_conll_file(all_data_conll['validation'], id2label_conll)

train_data_wnut = read_conll_file(all_data_wnut['train'], id2label_wnut)
dev_data_wnut = read_conll_file(all_data_wnut['validation'], id2label_wnut)


word_vocab_conll, tag_vocab_conll = build_vocab(train_data_conll)
word_vocab_wnut, tag_vocab_wnut = build_vocab(train_data_wnut)


train_loader_conll = DataLoader(NERDataset(train_data_conll, word_vocab_conll, tag_vocab_conll), batch_size=BATCH_SIZE,
                                collate_fn=collate_fn)
dev_loader_conll = DataLoader(NERDataset(dev_data_conll, word_vocab_conll, tag_vocab_conll), batch_size=BATCH_SIZE,
                              collate_fn=collate_fn)

train_loader_wnut = DataLoader(NERDataset(train_data_wnut, word_vocab_wnut, tag_vocab_wnut), batch_size=BATCH_SIZE,
                               collate_fn=collate_fn)
dev_loader_wnut = DataLoader(NERDataset(dev_data_wnut, word_vocab_wnut, tag_vocab_wnut), batch_size=BATCH_SIZE,
                             collate_fn=collate_fn)


fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
subsets_conll = create_subsets(train_data_conll, fractions)
subsets_wnut = create_subsets(train_data_wnut, fractions)

train_f1_scores_conll = []
val_f1_scores_conll = []
train_f1_scores_wnut = []
val_f1_scores_wnut = []


def train_and_evaluate(subsets, word_vocab, tag_vocab, val_loader):
    train_f1_scores = []
    val_f1_scores = []

    for subset in subsets:
        # Re-initialize the model
        model = SimpleRNN(vocab_size=len(word_vocab), tagset_size=len(tag_vocab))
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        train_subset_loader = DataLoader(NERDataset(subset, word_vocab, tag_vocab), batch_size=BATCH_SIZE,
                                         collate_fn=collate_fn, shuffle=True)

        for epoch in range(EPOCHS):
            model.train()
            for words, tags in train_subset_loader:
                optimizer.zero_grad()
                outputs = model(words)
                loss = criterion(outputs.view(-1, len(tag_vocab)), tags.view(-1))
                loss.backward()
                optimizer.step()

        train_f1 = evaluate_model(model, train_subset_loader)
        val_f1 = evaluate_model(model, val_loader)

        train_f1_scores.append(train_f1)
        val_f1_scores.append(val_f1)

    return train_f1_scores, val_f1_scores



train_f1_scores_conll, val_f1_scores_conll = train_and_evaluate(subsets_conll, word_vocab_conll, tag_vocab_conll,
                                                                dev_loader_conll)

train_f1_scores_wnut, val_f1_scores_wnut = train_and_evaluate(subsets_wnut, word_vocab_wnut, tag_vocab_wnut,
                                                              dev_loader_wnut)


plt.figure(figsize=(7, 7))
plt.plot(fractions, val_f1_scores_conll, marker='o')
plt.xlabel('Fraction of Training Data')
plt.ylabel('F1 Score')
plt.title('Learning Curve (CoNLL)')
plt.legend()
plt.show()

plt.figure(figsize=(7, 7))
plt.plot(fractions, val_f1_scores_wnut, marker='o')
plt.xlabel('Fraction of Training Data')
plt.ylabel('F1 Score')
plt.title('Learning Curve (WNUT)')
plt.legend()
plt.show()