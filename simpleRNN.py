import torch
from datasets import load_dataset
from torch import nn, optim
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


PAD = "PAD"
UNK = "UNK"
DIM_EMBEDDING = 100
RNN_HIDDEN = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.01
EPOCHS = 10


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


all_data_conll = load_dataset('conll2003')
all_data_wnut = load_dataset('wnut_17')

label_list_conll = all_data_conll['train'].features['ner_tags'].feature.names
id2label_conll = {i: label for i, label in enumerate(label_list_conll)}

label_list_wnut = all_data_wnut['train'].features['ner_tags'].feature.names
id2label_wnut = {i: label for i, label in enumerate(label_list_wnut)}


train_data_conll = read_conll_file(all_data_conll['train'], id2label_conll)
dev_data_conll = read_conll_file(all_data_conll['validation'], id2label_conll)
test_data_conll = read_conll_file(all_data_conll['test'], id2label_conll)

train_data_wnut = read_conll_file(all_data_wnut['train'], id2label_wnut)
dev_data_wnut = read_conll_file(all_data_wnut['validation'], id2label_wnut)
test_data_wnut = read_conll_file(all_data_wnut['test'], id2label_wnut)

word_vocab_conll, tag_vocab_conll = build_vocab(train_data_conll)
word_vocab_wnut, tag_vocab_wnut = build_vocab(train_data_wnut)


train_dataset_conll = NERDataset(train_data_conll, word_vocab_conll, tag_vocab_conll)
train_loader_conll = DataLoader(train_dataset_conll, batch_size=BATCH_SIZE, collate_fn=collate_fn)

dev_dataset_conll = NERDataset(dev_data_conll, word_vocab_conll, tag_vocab_conll)
dev_loader_conll = DataLoader(dev_dataset_conll, batch_size=BATCH_SIZE, collate_fn=collate_fn)

test_data_conll = NERDataset(test_data_conll, word_vocab_conll, tag_vocab_conll)
test_loader_conll = DataLoader(test_data_conll, batch_size=BATCH_SIZE, collate_fn=collate_fn)

train_dataset_wnut = NERDataset(train_data_wnut, word_vocab_wnut, tag_vocab_wnut)
train_loader_wnut = DataLoader(train_dataset_wnut, batch_size=BATCH_SIZE, collate_fn=collate_fn)

dev_dataset_wnut = NERDataset(dev_data_wnut, word_vocab_wnut, tag_vocab_wnut)
dev_loader_wnut = DataLoader(dev_dataset_wnut, batch_size=BATCH_SIZE, collate_fn=collate_fn)

test_data_wnut = NERDataset(test_data_wnut, word_vocab_wnut, tag_vocab_wnut)
test_loader_wnut = DataLoader(test_data_wnut, batch_size=BATCH_SIZE, collate_fn=collate_fn)


def train_model(model, train_loader):
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for sentences, tags in train_loader:
            model.zero_grad()
            tag_scores = model(sentences)  # Model predictions
            loss = loss_function(tag_scores.view(-1, tag_scores.shape[-1]), tags.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted_tags = torch.max(tag_scores, dim=2)
            correct_predictions += (predicted_tags == tags).sum().item()
            total_predictions += tags.numel()

        accuracy = correct_predictions / total_predictions * 100
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")



def evaluate_model(model, data_loader):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for sentences, tags in data_loader:
            tag_scores = model(sentences)
            _, predicted_tags = torch.max(tag_scores, dim=2)
            predictions.extend(predicted_tags.view(-1).cpu().numpy())
            true_labels.extend(tags.view(-1).cpu().numpy())

    precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)

    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")


model_conll = SimpleRNN(vocab_size=len(word_vocab_conll), tagset_size=len(tag_vocab_conll))
train_model(model_conll, train_loader_conll)

print("Evaluation on CoNLL2003 validation set:")
evaluate_model(model_conll, dev_loader_conll)

model_wnut = SimpleRNN(vocab_size=len(word_vocab_wnut), tagset_size=len(tag_vocab_wnut))
train_model(model_wnut, train_loader_wnut)

print("Evaluation on WNUT17 validation set:")
evaluate_model(model_wnut, dev_loader_wnut)
