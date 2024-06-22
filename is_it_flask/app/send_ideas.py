# import requests
# import pandas as pd
# import numpy as np
# from transformers import BertTokenizer, BertModel, AdamW
# import torch
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer, PorterStemmer
# from nltk.tokenize import word_tokenize
# import ssl
# import base64


# # SSL 인증 무시
# ssl._create_default_https_context = ssl._create_unverified_context

# # NLTK 데이터 다운로드
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # device 설정
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Spring Boot 서버 URL
# SPRING_BOOT_URL = "http://localhost:8080/api/ideas/add"

# class MultiTaskBertModel(torch.nn.Module):
#     def __init__(self, num_category_labels, num_function_labels, num_usage_base_labels):
#         super(MultiTaskBertModel, self).__init__()
#         self.bert = BertModel.from_pretrained('bert-base-uncased')
#         self.category_classifier = torch.nn.Linear(self.bert.config.hidden_size, num_category_labels)
#         self.function_classifier = torch.nn.Linear(self.bert.config.hidden_size, num_function_labels)
#         self.usage_base_classifier = torch.nn.Linear(self.bert.config.hidden_size, num_usage_base_labels)
    
#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         pooled_output = outputs[1]  # BERT's pooled output

#         category_logits = self.category_classifier(pooled_output)
#         function_logits = self.function_classifier(pooled_output)
#         usage_base_logits = self.usage_base_classifier(pooled_output)

#         return category_logits, function_logits, usage_base_logits
    
#     def save_pretrained(self, save_directory):
#         # BERT 모델 저장
#         self.bert.save_pretrained(save_directory)
#         # 분류기 저장
#         torch.save(self.category_classifier.state_dict(), f'{save_directory}/category_classifier.bin')
#         torch.save(self.function_classifier.state_dict(), f'{save_directory}/function_classifier.bin')
#         torch.save(self.usage_base_classifier.state_dict(), f'{save_directory}/usage_base_classifier.bin')
    
#     @classmethod
#     def from_pretrained(cls, load_directory, num_category_labels, num_function_labels, num_usage_base_labels):
#         model = cls(num_category_labels, num_function_labels, num_usage_base_labels)
#         # BERT 모델 로드
#         model.bert = BertModel.from_pretrained(load_directory)
#         # 분류기 로드
#         model.category_classifier.load_state_dict(torch.load(f'{load_directory}/category_classifier.bin'))
#         model.function_classifier.load_state_dict(torch.load(f'{load_directory}/function_classifier.bin'))
#         model.usage_base_classifier.load_state_dict(torch.load(f'{load_directory}/usage_base_classifier.bin'))
#         return model

# # 학습된 BERT 모델 및 토크나이저 로드
# model = MultiTaskBertModel.from_pretrained('finetuned_bert_model', num_category_labels=11, num_function_labels=11, num_usage_base_labels=4).to(device)
# tokenizer = BertTokenizer.from_pretrained('finetuned_bert_tokenizer')

# # 텍스트 전처리 함수
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()
# stemmer = PorterStemmer()

# def preprocess_text(text):
#     if not isinstance(text, str):
#         return ""
#     tokens = word_tokenize(text.lower())
#     tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
#     lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
#     stemmed_tokens = [stemmer.stem(word) for word in lemmatized_tokens]
#     return ' '.join(stemmed_tokens)

# # 임베딩 계산 함수
# def embed_text(text):
#     preprocessed_text = preprocess_text(text)
#     inputs = tokenizer(preprocessed_text, return_tensors='pt', truncation=True, padding=True).to(device)
#     with torch.no_grad():
#         category_logits, function_logits, usage_base_logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    
#     # 각 태스크의 logits를 임베딩으로 사용
#     category_embedding = category_logits.cpu().numpy().flatten().astype(np.float32)
#     function_embedding = function_logits.cpu().numpy().flatten().astype(np.float32)
#     usage_base_embedding = usage_base_logits.cpu().numpy().flatten().astype(np.float32)
    
#     return category_embedding, function_embedding, usage_base_embedding

# # 데이터 로드
# df = pd.read_csv('/Users/gyuwonchoi/Downloads/labeled_producthunt_data_with_ai.csv')
# df.columns = ['name', 'description', 'category', 'function', 'usage_base']

# # 모든 텍스트에 대해 임베딩 계산 및 전송
# for _, row in df.iterrows():
#     title = row['name']
#     text = row['description']
    
#     category_embedding, function_embedding, usage_base_embedding = embed_text(text)
    
#     # 임베딩 값이 유효한지 확인
#     if not (np.all(np.isfinite(category_embedding)) and np.all(np.isfinite(function_embedding)) and np.all(np.isfinite(usage_base_embedding))):
#         print(f"Invalid embedding for idea: {title}")
#         continue
    
#     # 임베딩을 byte array로 변환 후 Base64로 인코딩
#     category_embedding_bytes = base64.b64encode(category_embedding.tobytes()).decode('utf-8')
#     function_embedding_bytes = base64.b64encode(function_embedding.tobytes()).decode('utf-8')
#     usage_base_embedding_bytes = base64.b64encode(usage_base_embedding.tobytes()).decode('utf-8')
    
#     idea = {
#         "title": title,
#         "detailedSummary": text,
#         "categoryEmbedding": category_embedding_bytes,  # Base64로 인코딩된 임베딩 값 사용
#         "functionEmbedding": function_embedding_bytes,  # Base64로 인코딩된 임베딩 값 사용
#         "usageBaseEmbedding": usage_base_embedding_bytes  # Base64로 인코딩된 임베딩 값 사용
#     }
    
#     try:
#         response = requests.post(SPRING_BOOT_URL, json=idea) 
#         if response.status_code == 200:
#             print(f"Successfully added idea: {title}")
#         else:
#             print(f"Failed to add idea: {title}, status code: {response.status_code}")
#     except requests.exceptions.RequestException as e:
#         print(f"Request failed for idea: {title}, error: {e}")

import requests
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import ssl
import base64

# SSL 인증 무시
ssl._create_default_https_context = ssl._create_unverified_context

# NLTK 데이터 다운로드
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Spring Boot 서버 URL
SPRING_BOOT_URL = "http://localhost:8080/api/ideas/add"

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
        
        # 분류기 레이어 재초기화
        model.category_classifier = torch.nn.Linear(model.bert.config.hidden_size, num_category_labels)
        model.function_classifier = torch.nn.Linear(model.bert.config.hidden_size, num_function_labels)
        model.usage_base_classifier = torch.nn.Linear(model.bert.config.hidden_size, num_usage_base_labels)
        
        return model

# 학습된 BERT 모델 및 토크나이저 로드
model = MultiTaskBertModel.from_pretrained('finetuned_bert_model', num_category_labels=22, num_function_labels=22, num_usage_base_labels=4).to(device)
tokenizer = BertTokenizer.from_pretrained('finetuned_bert_tokenizer')

# 텍스트 전처리 함수
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    stemmed_tokens = [stemmer.stem(word) for word in lemmatized_tokens]
    return ' '.join(stemmed_tokens)

# 임베딩 계산 함수
def embed_text(text):
    preprocessed_text = preprocess_text(text)
    inputs = tokenizer(preprocessed_text, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        category_logits, function_logits, usage_base_logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    
    # 각 태스크의 logits를 임베딩으로 사용
    category_embedding = category_logits.cpu().detach().numpy().flatten().astype(np.float32)
    function_embedding = function_logits.cpu().detach().numpy().flatten().astype(np.float32)
    usage_base_embedding = usage_base_logits.cpu().detach().numpy().flatten().astype(np.float32)

    # 상세 설명의 임베딩 계산
    description_embedding = model.bert(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])[1].cpu().detach().numpy().flatten().astype(np.float32)
    
    # 예측된 값
    category = category_logits.argmax(dim=1).item()
    function = function_logits.argmax(dim=1).item()
    usage_base = usage_base_logits.argmax(dim=1).item()
    
    return category, function, usage_base, category_embedding, function_embedding, usage_base_embedding, description_embedding

# 데이터 로드
df = pd.read_csv('/Users/gyuwonchoi/Downloads/final_refined_producthunt_data_with_ai_final.csv')
df.columns = ['name', 'description', 'category', 'function', 'usage_base']

# 모든 텍스트에 대해 임베딩 계산 및 전송
for _, row in df.iterrows():
    title = row['name']
    text = row['description']
    
    category, function, usage_base, category_embedding, function_embedding, usage_base_embedding, description_embedding = embed_text(text)
    
    # 임베딩 값이 유효한지 확인
    if not np.all(np.isfinite(description_embedding)):
        print(f"Invalid embedding for idea: {title}")
        continue
    
    # 임베딩을 byte array로 변환 후 Base64로 인코딩
    category_embedding_bytes = base64.b64encode(category_embedding.tobytes()).decode('utf-8')
    function_embedding_bytes = base64.b64encode(function_embedding.tobytes()).decode('utf-8')
    usage_base_embedding_bytes = base64.b64encode(usage_base_embedding.tobytes()).decode('utf-8')
    description_embedding_bytes = base64.b64encode(description_embedding.tobytes()).decode('utf-8')
    
    idea = {
        "title": title,
        "detailedSummary": text,
        "category": category,
        "function": function,
        "usageBase": usage_base,
        "categoryEmbedding": category_embedding_bytes,  # Base64로 인코딩된 임베딩 값 사용
        "functionEmbedding": function_embedding_bytes,  # Base64로 인코딩된 임베딩 값 사용
        "usageBaseEmbedding": usage_base_embedding_bytes,  # Base64로 인코딩된 임베딩 값 사용
        "descriptionEmbedding": description_embedding_bytes  # Base64로 인코딩된 임베딩 값 사용
    }
    
    try:
        response = requests.post(SPRING_BOOT_URL, json=idea)
        if response.status_code == 200:
            print(f"Successfully added idea: {title}")
        else:
            print(f"Failed to add idea: {title}, status code: {response.status_code}, message: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed for idea: {title}, error: {e}")
