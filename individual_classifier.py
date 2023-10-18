import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaTokenizer, DebertaForSequenceClassification
from transformers import XLNetTokenizerFast, XLNetForSequenceClassification
from transformers import ElectraTokenizerFast, ElectraForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.optim import AdamW
import time
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import re
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from torch.nn.functional import softmax

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    cleaned_text = re.sub(r"'s", "", text)
    tokens = word_tokenize(cleaned_text)
    return ' '.join(tokens)

class PhishDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

# Load Inputs
train_df = pd.read_csv('/content/train_balanced_50.csv')
val_df = pd.read_csv('/content/test_balanced_50.csv')
test_df = pd.read_csv('/content/val_balanced_50.csv')

# Input Preprocessing
train_texts = [preprocess_text(sentence) for sentence in train_df['Sentence/Code'].tolist()]
train_labels = train_df['label'].tolist()
val_texts = [preprocess_text(sentence) for sentence in val_df['Sentence/Code'].tolist()]
val_labels = val_df['label'].tolist()
test_texts = [preprocess_text(sentence) for sentence in test_df['Sentence/Code'].tolist()]
test_labels = test_df['label'].tolist()

# tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
# tokenizer = XLNetTokenizerFast.from_pretrained('xlnet-base-cased')
# tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-small-discriminator')
# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
# tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)
train_dataset = PhishDataset(train_encodings, train_labels)
test_dataset = PhishDataset(test_encodings, test_labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# model = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-base', num_labels=2)
# model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=2)
# model = ElectraForSequenceClassification.from_pretrained('google/electra-small-discriminator', num_labels=2)
# model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
# model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=2)
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
model = model.to('cuda')

optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()
start_time = time.time()
# Train loop
for epoch in range(10):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        labels = batch['labels'].to('cuda')
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

# model.save_pretrained('exp_model_DeBERTa')
# model.save_pretrained('exp_model_XLNet')
# model.save_pretrained('exp_model_Electra')
# model.save_pretrained('exp_model_DistilBERT')
# model.save_pretrained('exp_model_BERT')
# model.save_pretrained('exp_model_RoBERTa_large')
model.save_pretrained('exp_model_RoBERTa_base')
end_time = time.time()
# Test loop
model.eval()
correct_predictions = 0
total_predictions = 0
all_predictions = []
all_true_labels = []
all_test_texts = []
# idx = 0

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        labels = batch['labels'].to('cuda')
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_true_labels.extend(labels.cpu().numpy())
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += labels.size(0)
        for j in range(len(labels)):
            all_test_texts.append(test_texts[i * len(labels) + j])

# Results
accuracy = correct_predictions / total_predictions if total_predictions != 0 else 0
f1 = f1_score(all_true_labels, all_predictions)
conf_matrix = confusion_matrix(all_true_labels, all_predictions)
print(f"Testing Accuracy: {accuracy * 100:.2f}%")
print(f"Testing F1 Score: {f1:.2f}")
print("Time required to fine-tune: ", end_time - start_time)
print(conf_matrix)
predictions_df = pd.DataFrame({
    'True Labels': all_true_labels,
    'Predictions': all_predictions
})

predictions_df.to_csv('test_predictions.csv', index=False)
report = classification_report(all_true_labels, all_predictions, target_names=['not phishing', 'phishing'])
print(report)
