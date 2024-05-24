import torch
from datasets import load_dataset
from torch import nn, optim
from sklearn.metrics import f1_score
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import random

PAD = "PAD"
UNK = "UNK"
DIM_EMBEDDING = 100
LSTM_HIDDEN = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.01
EPOCHS = 10

def read_conll_file(dataset):
    data = []
    for example in dataset:
        words = example['tokens']
        tags = example['ner_tags']
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

all_data = load_dataset('wnut_17')
train_data = read_conll_file(all_data['train'])
dev_data = read_conll_file(all_data['test'])

word_vocab, tag_vocab = build_vocab(train_data)
train_dataset = NERDataset(train_data, word_vocab, tag_vocab)
dev_dataset = NERDataset(dev_data, word_vocab, tag_vocab)

def create_subsets(dataset, fractions):
    subsets = []
    total_size = len(dataset)
    indices = list(range(total_size))
    random.shuffle(indices)

    for fraction in fractions:
        subset_size = int(fraction * total_size)
        subset_indices = indices[:subset_size]
        subsets.append(Subset(dataset, subset_indices))

    return subsets

fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
train_subsets = create_subsets(train_dataset, fractions)

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=DIM_EMBEDDING, hidden_dim=LSTM_HIDDEN):
        super(SimpleLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)
        return tag_space

def train_and_evaluate_on_subset(subset):
    train_loader = DataLoader(subset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)

    model = SimpleLSTM(len(word_vocab), len(tag_vocab))
    loss_function = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        for words, tags in train_loader:
            model.zero_grad()
            tag_scores = model(words)
            loss = loss_function(tag_scores.view(-1, len(tag_vocab)), tags.view(-1))
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for words, tags in dev_loader:
            tag_scores = model(words)
            preds = torch.argmax(tag_scores, dim=2)
            all_preds.extend(preds.view(-1).tolist())
            all_labels.extend(tags.view(-1).tolist())

    # Filter out padding tokens
    filtered_preds = [p for p, l in zip(all_preds, all_labels) if l != -100]
    filtered_labels = [l for l in all_labels if l != -100]

    f1 = f1_score(filtered_labels, filtered_preds, average='macro')
    return f1

f1_scores = []
for subset in train_subsets:
    f1 = train_and_evaluate_on_subset(subset)
    f1_scores.append(f1)


plt.plot([fraction * 100 for fraction in fractions], f1_scores)
plt.xlabel('Percentage of Data Used')
plt.ylabel('F1 Score')
plt.title('Learning Curve')
plt.show()
