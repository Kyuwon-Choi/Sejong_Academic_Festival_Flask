import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import numpy as np
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import ssl
import os

# SSL 인증 무시
ssl._create_default_https_context = ssl._create_unverified_context

# NLTK 데이터 다운로드
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 텍스트 전처리 함수
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    stemmed_tokens = [stemmer.stem(word) for word in lemmatized_tokens]
    return ' '.join(stemmed_tokens)

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': {
                'category': torch.tensor(label[0], dtype=torch.long),
                'function': torch.tensor(label[1], dtype=torch.long),
                'usage_base': torch.tensor(label[2], dtype=torch.long)
            }
        }

class MultiTaskBertModel(torch.nn.Module):
    def __init__(self, num_category_labels, num_function_labels, num_usage_base_labels):
        super(MultiTaskBertModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.category_classifier = torch.nn.Linear(self.bert.config.hidden_size, num_category_labels)
        self.function_classifier = torch.nn.Linear(self.bert.config.hidden_size, num_function_labels)
        self.usage_base_classifier = torch.nn.Linear(self.bert.config.hidden_size, num_usage_base_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # BERT's pooled output

        category_logits = self.category_classifier(pooled_output)
        function_logits = self.function_classifier(pooled_output)
        usage_base_logits = self.usage_base_classifier(pooled_output)

        return category_logits, function_logits, usage_base_logits
    
    def save_pretrained(self, save_directory):
        # BERT 모델 저장
        self.bert.save_pretrained(save_directory)
        # 분류기 저장
        torch.save(self.category_classifier.state_dict(), f'{save_directory}/category_classifier.bin')
        torch.save(self.function_classifier.state_dict(), f'{save_directory}/function_classifier.bin')
        torch.save(self.usage_base_classifier.state_dict(), f'{save_directory}/usage_base_classifier.bin')
    
    @classmethod
    def from_pretrained(cls, load_directory, num_category_labels, num_function_labels, num_usage_base_labels):
        model = cls(num_category_labels, num_function_labels, num_usage_base_labels)
        # BERT 모델 로드
        model.bert = BertModel.from_pretrained(load_directory)
        # 분류기 로드
        model.category_classifier.load_state_dict(torch.load(f'{load_directory}/category_classifier.bin'))
        model.function_classifier.load_state_dict(torch.load(f'{load_directory}/function_classifier.bin'))
        model.usage_base_classifier.load_state_dict(torch.load(f'{load_directory}/usage_base_classifier.bin'))
        return model

# device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 로드
df = pd.read_csv('/Users/gyuwonchoi/Downloads/final_refined_producthunt_data_with_ai_final.csv')
df.columns = ['name', 'description', 'category', 'function', 'usage_base']

# 데이터 전처리 및 라벨링
df = df.dropna(subset=['description', 'category', 'function', 'usage_base'])
df['description'] = df['description'].apply(preprocess_text)

# 라벨 인코딩
category_labels = {label: idx for idx, label in enumerate(df['category'].unique())}
function_labels = {label: idx for idx, label in enumerate(df['function'].unique())}
usage_base_labels = {label: idx for idx, label in enumerate(df['usage_base'].unique())}

df['category'] = df['category'].map(category_labels)
df['function'] = df['function'].map(function_labels)
df['usage_base'] = df['usage_base'].map(usage_base_labels)

# 데이터 분할
texts = df['description'].tolist()
labels = list(zip(df['category'].tolist(), df['function'].tolist(), df['usage_base'].tolist()))

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1)

# 토크나이저 초기화
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 파라미터 설정
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 2e-5  # 학습률을 낮춤

# 데이터셋 및 데이터로더 생성
train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = TextDataset(val_texts, val_labels, tokenizer, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# 모델 초기화
model = MultiTaskBertModel(
    num_category_labels=len(category_labels),
    num_function_labels=len(function_labels),
    num_usage_base_labels=len(usage_base_labels)
).to(device)

# 옵티마이저 및 스케줄러
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# 손실 함수
category_loss_fn = torch.nn.CrossEntropyLoss().to(device)
function_loss_fn = torch.nn.CrossEntropyLoss().to(device)
usage_base_loss_fn = torch.nn.CrossEntropyLoss().to(device)

# 학습 함수
def train_epoch(model, data_loader, optimizer, device, scheduler, n_examples):
    model.train()
    total_category_loss = 0
    total_function_loss = 0
    total_usage_base_loss = 0
    correct_category_predictions = 0
    correct_function_predictions = 0
    correct_usage_base_predictions = 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        category_labels = batch['labels']['category'].to(device)
        function_labels = batch['labels']['function'].to(device)
        usage_base_labels = batch['labels']['usage_base'].to(device)

        category_logits, function_logits, usage_base_logits = model(input_ids=input_ids, attention_mask=attention_mask)
        
        category_loss = category_loss_fn(category_logits, category_labels)
        function_loss = function_loss_fn(function_logits, function_labels)
        usage_base_loss = usage_base_loss_fn(usage_base_logits, usage_base_labels)

        category_weight = 1.0
        function_weight = 1.6
        usage_base_weight = 0.4

        total_loss = category_weight * category_loss + function_weight * function_loss + usage_base_weight * usage_base_loss

        _, category_preds = torch.max(category_logits, dim=1)
        _, function_preds = torch.max(function_logits, dim=1)
        _, usage_base_preds = torch.max(usage_base_logits, dim=1)

        correct_category_predictions += torch.sum(category_preds == category_labels)
        correct_function_predictions += torch.sum(function_preds == function_labels)
        correct_usage_base_predictions += torch.sum(usage_base_preds == usage_base_labels)
        
        total_category_loss += category_loss.item()
        total_function_loss += function_loss.item()
        total_usage_base_loss += usage_base_loss.item()

        total_loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        logger.info(f"Batch processed. Loss: {total_loss.item()}")

    return (
        correct_category_predictions.double() / n_examples,
        correct_function_predictions.double() / n_examples,
        correct_usage_base_predictions.double() / n_examples,
        total_category_loss / len(data_loader),
        total_function_loss / len(data_loader),
        total_usage_base_loss / len(data_loader)
    )

# 평가 함수
def eval_model(model, data_loader, device, n_examples):
    model.eval()
    total_category_loss = 0
    total_function_loss = 0
    total_usage_base_loss = 0
    correct_category_predictions = 0
    correct_function_predictions = 0
    correct_usage_base_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            category_labels = batch['labels']['category'].to(device)
            function_labels = batch['labels']['function'].to(device)
            usage_base_labels = batch['labels']['usage_base'].to(device)

            category_logits, function_logits, usage_base_logits = model(input_ids=input_ids, attention_mask=attention_mask)
            
            category_loss = category_loss_fn(category_logits, category_labels)
            function_loss = function_loss_fn(function_logits, function_labels)
            usage_base_loss = usage_base_loss_fn(usage_base_logits, usage_base_labels)

            category_weight = 1.0
            function_weight = 1.6
            usage_base_weight = 0.4

            total_loss = category_weight * category_loss + function_weight * function_loss + usage_base_weight * usage_base_loss

            _, category_preds = torch.max(category_logits, dim=1)
            _, function_preds = torch.max(function_logits, dim=1)
            _, usage_base_preds = torch.max(usage_base_logits, dim=1)

            correct_category_predictions += torch.sum(category_preds == category_labels)
            correct_function_predictions += torch.sum(function_preds == function_labels)
            correct_usage_base_predictions += torch.sum(usage_base_preds == usage_base_labels)

            total_category_loss += category_loss.item()
            total_function_loss += function_loss.item()
            total_usage_base_loss += usage_base_loss.item()

    return (
        correct_category_predictions.double() / n_examples,
        correct_function_predictions.double() / n_examples,
        correct_usage_base_predictions.double() / n_examples,
        total_category_loss / len(data_loader),
        total_function_loss / len(data_loader),
        total_usage_base_loss / len(data_loader)
    )

# 학습 루프
for epoch in range(EPOCHS):
    train_category_acc, train_function_acc, train_usage_base_acc, train_category_loss, train_function_loss, train_usage_base_loss = train_epoch(
        model, train_loader, optimizer, device, scheduler, len(train_texts)
    )
    val_category_acc, val_function_acc, val_usage_base_acc, val_category_loss, val_function_loss, val_usage_base_loss = eval_model(
        model, val_loader, device, len(val_texts)
    )

    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print(f'Train loss - Category: {train_category_loss}, Function: {train_function_loss}, Usage Base: {train_usage_base_loss}')
    print(f'Train accuracy - Category: {train_category_acc}, Function: {train_function_acc}, Usage Base: {train_usage_base_acc}')
    print(f'Val   loss - Category: {val_category_loss}, Function: {val_function_loss}, Usage Base: {val_usage_base_loss}')
    print(f'Val   accuracy - Category: {val_category_acc}, Function: {val_function_acc}, Usage Base: {val_usage_base_acc}')

# 모델 저장 디렉토리 설정
save_directory = os.path.join(os.path.dirname(__file__), 'finetuned_bert_model')

# 모델 저장
model.save_pretrained(save_directory)
tokenizer.save_pretrained(os.path.join(save_directory, 'tokenizer'))

       
