import datasets
import random
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import torch
import numpy as np
from transformers import BertTokenizerFast
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

wnut17 = datasets.load_dataset("wnut_17")
num_labels = len(wnut17["train"].features["ner_tags"].feature.names)

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


def tokenize_and_align_labels(examples, label_all_tokens=True):
    """
    Function to tokenize and align labels with respect to the tokens. This function is specifically designed for
    Named Entity Recognition (NER) tasks where alignment of the labels is necessary after tokenization.

    Parameters:
    examples (dict): A dictionary containing the tokens and the corresponding NER tags.
                     - "tokens": list of words in a sentence.
                     - "ner_tags": list of corresponding entity tags for each word.

    label_all_tokens (bool): A flag to indicate whether all tokens should have labels.
                             If False, only the first token of a word will have a label,
                             the other tokens (subwords) corresponding to the same word will be assigned -100.

    Returns:
    tokenized_inputs (dict): A dictionary containing the tokenized inputs and the corresponding labels aligned with the tokens.
    """
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        # word_ids() => Return a list mapping the tokens
        # to their actual word in the initial sentence.
        # It Returns a list indicating the word corresponding to each token.
        previous_word_idx = None
        label_ids = []
        # Special tokens like `<s>` and `<\s>` are originally mapped to None
        # We need to set the label to -100 so they are automatically ignored in the loss function.
        for word_idx in word_ids:
            if word_idx is None:
                # set –100 as the label for these special tokens
                label_ids.append(-100)
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            elif word_idx != previous_word_idx:
                # if current word_idx is != prev then its the most regular case
                # and add the corresponding token
                label_ids.append(label[word_idx])
            else:
                # to take care of sub-words which have the same word_idx
                # set -100 as well for them, but only if label_all_tokens == False
                label_ids.append(label[word_idx] if label_all_tokens else -100)
                # mask the subword representations after the first subword

            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = wnut17.map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

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

from transformers import TrainingArguments, Trainer
args = TrainingArguments(
"test-ner",
evaluation_strategy = "epoch",
learning_rate=2e-5,
per_device_train_batch_size=16,
per_device_eval_batch_size=16,
num_train_epochs=3,
weight_decay=0.01,
)

data_collator = DataCollatorForTokenClassification(tokenizer)

metric = datasets.load_metric("seqeval")

label_list = wnut17["train"].features["ner_tags"].feature.names


def compute_metrics(eval_preds):
    """
    Function to compute the evaluation metrics for Named Entity Recognition (NER) tasks.
    The function computes precision, recall, F1 score and accuracy.

    Parameters:
    eval_preds (tuple): A tuple containing the predicted logits and the true labels.

    Returns:
    A dictionary containing the precision, recall, F1 score and accuracy.
    """
    pred_logits, labels = eval_preds

    pred_logits = np.argmax(pred_logits, axis=2)
    # the logits and the probabilities are in the same order,
    # so we don’t need to apply the softmax

    # We remove all the values where the label is -100
    predictions = [
        [label_list[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]

    true_labels = [
        [label_list[l] for (eval_preds, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]
    results = metric.compute(predictions=predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def train_and_evaluate_on_subset(train_subset):
    trainer = Trainer(
        model,
        args,
       train_dataset= train_subset,
       eval_dataset= tokenized_datasets["validation"],
       data_collator= data_collator,
       tokenizer= tokenizer,
       compute_metrics= compute_metrics
    )
    trainer.train()
    eval_result = trainer.evaluate()
    return eval_result["eval_f1"]

fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
train_subsets = create_subsets(tokenized_datasets["train"], fractions)

f1_scores = []
for subset in train_subsets:
    f1 = train_and_evaluate_on_subset(subset)
    f1_scores.append(f1)

def plot_performance(fractions, f1_scores):
    plt.plot([int(f * 100) for f in fractions], f1_scores, marker='o')
    plt.xlabel('Training Data Fraction (%)')
    plt.ylabel('F1 Score')
    plt.title('Model Performance vs Training Sample Size')
    plt.grid(True)
    plt.show()

plot_performance(fractions, f1_scores)
