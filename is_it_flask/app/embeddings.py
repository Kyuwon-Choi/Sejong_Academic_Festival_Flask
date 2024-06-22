# from transformers import BertTokenizer
# import torch
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import base64
# from .preprocessing import preprocess_text
# from .deepl import deepl_translate
# import logging
# from sklearn.preprocessing import normalize
# from .multi_task_model import MultiTaskBertModel  # 수정된 모델을 불러옵니다.
# import os

# # 경로 설정
# model_directory = os.path.join(os.path.dirname(__file__), 'finetuned_bert_model')

# # BERT 모델과 토크나이저 로드
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = MultiTaskBertModel.from_pretrained(model_directory, num_category_labels=11, num_function_labels=11, num_usage_base_labels=4).to(device)

# def get_combined_embedding(text):
#     if not isinstance(text, str):
#         return np.zeros((26,))
#     inputs = bert_tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
#     with torch.no_grad():
#         category_logits, function_logits, usage_base_logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    
#     # 각 태스크의 logits를 임베딩으로 사용하고 결합
#     category_embedding = category_logits.cpu().numpy().flatten().astype(np.float32)
#     function_embedding = function_logits.cpu().numpy().flatten().astype(np.float32)
#     usage_base_embedding = usage_base_logits.cpu().numpy().flatten().astype(np.float32)
    
#     combined_embedding = np.concatenate([category_embedding, function_embedding, usage_base_embedding])
#     return combined_embedding

# def embed_input_text(input_text, max_length=512):
#     preprocessed_text = preprocess_text(input_text)
#     logging.debug(f"Preprocessed text: {preprocessed_text}")
#     embedding = get_combined_embedding(preprocessed_text)
#     logging.debug(f"Input embedding shape: {embedding.shape}, values: {embedding}")
#     return embedding

# def find_similar_ideas(input_idea, ideas, top_k=5):
#     logging.debug(f"Finding similar ideas for: {input_idea}")

#     input_embedding = embed_input_text(input_idea)
#     logging.debug(f"Input embedding: {input_embedding}")

#     stored_embeddings = []
#     valid_ideas = []
#     for idea in ideas:
#         try:
#             category_embedding_str = idea.get('categoryEmbedding', "")
#             function_embedding_str = idea.get('functionEmbedding', "")
#             usage_base_embedding_str = idea.get('usageBaseEmbedding', "")

#             category_embedding_bytes = base64.b64decode(category_embedding_str)
#             function_embedding_bytes = base64.b64decode(function_embedding_str)
#             usage_base_embedding_bytes = base64.b64decode(usage_base_embedding_str)

#             category_embedding = np.frombuffer(category_embedding_bytes, dtype=np.float32)
#             function_embedding = np.frombuffer(function_embedding_bytes, dtype=np.float32)
#             usage_base_embedding = np.frombuffer(usage_base_embedding_bytes, dtype=np.float32)

#             if len(category_embedding) == 11 and len(function_embedding) == 11 and len(usage_base_embedding) == 4:
#                 combined_embedding = np.concatenate([category_embedding, function_embedding, usage_base_embedding])
#                 stored_embeddings.append(combined_embedding)
#                 valid_ideas.append(idea)
#             else:
#                 logging.error(f"Invalid embedding length for idea {idea['title']}: category({len(category_embedding)}), function({len(function_embedding)}), usage_base({len(usage_base_embedding)})")
#         except (ValueError, KeyError) as e:
#             logging.error(f"Error decoding embedding for idea {idea['title']}: {e}")

#     if len(stored_embeddings) == 0:
#         logging.error("No valid embeddings found in the ideas")
#         return []

#     stored_embeddings = np.vstack(stored_embeddings)
#     logging.debug(f"Stored embeddings shape: {stored_embeddings.shape}")

#     # Normalize the embeddings
#     input_embedding = normalize(input_embedding.reshape(1, -1))
#     stored_embeddings = normalize(stored_embeddings)

#     similarities = cosine_similarity(input_embedding, stored_embeddings).flatten()
#     logging.debug(f"Similarities: {similarities}")
#     top_k_indices = similarities.argsort()[-top_k:][::-1]

#     similar_ideas_details = []
#     for i in top_k_indices:
#         similarity_score = float(similarities[i])
#         similar_idea = valid_ideas[i]
#         try:
#             detail_summary_ko = deepl_translate(similar_idea['detailedSummary'], target_lang='KO')
#         except Exception as e:
#             logging.error(f"Error translating detail summary: {e}")
#             detail_summary_ko = "Translation error"

#         similar_ideas_details.append({
#             "title": similar_idea['title'],
#             "detailedSummary": detail_summary_ko,
#             "similarity_score": similarity_score * 100
#         })

#     logging.debug(f"Similar ideas details: {similar_ideas_details}")
#     return similar_ideas_details
# embeddings.py
from transformers import BertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import base64
from .preprocessing import preprocess_text
import logging
from sklearn.preprocessing import normalize
from .multi_task_model import MultiTaskBertModel  # 수정된 모델을 불러옵니다.
import os
from .deepl import deepl_translate

# 경로 설정
model_directory = os.path.join(os.path.dirname(__file__), 'finetuned_bert_model')
tokenizer_directory = os.path.join(os.path.dirname(__file__), 'finetuned_bert_tokenizer')

# BERT 모델과 토크나이저 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_tokenizer = BertTokenizer.from_pretrained(tokenizer_directory)
model = MultiTaskBertModel.from_pretrained(model_directory, num_category_labels=22, num_function_labels=22, num_usage_base_labels=4).to(device)

def get_combined_embedding(text):
    """입력 텍스트를 임베딩하여 결합된 임베딩 벡터를 반환합니다."""
    if not isinstance(text, str):
        return np.zeros((48,))
    inputs = bert_tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        category_logits, function_logits, usage_base_logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    
    # 각 태스크의 logits를 임베딩으로 사용하고 결합
    category_embedding = category_logits.cpu().numpy().flatten().astype(np.float32)
    function_embedding = function_logits.cpu().numpy().flatten().astype(np.float32)
    usage_base_embedding = usage_base_logits.cpu().numpy().flatten().astype(np.float32)
    
    combined_embedding = np.concatenate([category_embedding, function_embedding, usage_base_embedding])
    return combined_embedding

def embed_input_text(input_text, max_length=512):
    """입력 텍스트를 전처리하고 임베딩하여 반환합니다."""
    preprocessed_text = preprocess_text(input_text)
    logging.debug(f"Preprocessed text: {preprocessed_text}")
    embedding = get_combined_embedding(preprocessed_text)
    logging.debug(f"Input embedding shape: {embedding.shape}, values: {embedding}")
    return embedding

def filter_similar_data(predicted_category, predicted_function, predicted_usage_base, data):
    """예측된 category, function, usage_base 값을 기준으로 유사한 데이터를 필터링합니다."""
    similar_data = []
    for item in data:
        logging.debug(f"Item category: {item.get('category')}, function: {item.get('function')}, usageBase: {item.get('usageBase')}")
        if (item.get('category') == predicted_category and 
            item.get('function') == predicted_function and 
            item.get('usageBase') == predicted_usage_base):
            similar_data.append(item)
    logging.debug(f"Similar data found: {len(similar_data)} items")
    return similar_data

def find_similar_ideas(input_idea, ideas, top_k=5):
    logging.debug(f"Finding similar ideas for: {input_idea}")

    input_embedding = embed_input_text(input_idea)
    logging.debug(f"Input embedding: {input_embedding}")

    stored_embeddings = []
    valid_ideas = []
    for idea in ideas:
        try:
            category_embedding_str = idea.get('categoryEmbedding', "")
            function_embedding_str = idea.get('functionEmbedding', "")
            usage_base_embedding_str = idea.get('usageBaseEmbedding', "")

            category_embedding_bytes = base64.b64decode(category_embedding_str)
            function_embedding_bytes = base64.b64decode(function_embedding_str)
            usage_base_embedding_bytes = base64.b64decode(usage_base_embedding_str)

            category_embedding = np.frombuffer(category_embedding_bytes, dtype=np.float32)
            function_embedding = np.frombuffer(function_embedding_bytes, dtype=np.float32)
            usage_base_embedding = np.frombuffer(usage_base_embedding_bytes, dtype=np.float32)

            logging.debug(f"Idea title: {idea['title']}")
            logging.debug(f"Category embedding length: {len(category_embedding)}")
            logging.debug(f"Function embedding length: {len(function_embedding)}")
            logging.debug(f"Usage base embedding length: {len(usage_base_embedding)}")

            if len(category_embedding) == 22 and len(function_embedding) == 22 and len(usage_base_embedding) == 4:
                combined_embedding = np.concatenate([category_embedding, function_embedding, usage_base_embedding])
                stored_embeddings.append(combined_embedding)
                valid_ideas.append(idea)
            else:
                logging.error(f"Invalid embedding length for idea {idea['title']}: category({len(category_embedding)}), function({len(function_embedding)}), usage_base({len(usage_base_embedding)})")
        except (ValueError, KeyError) as e:
            logging.error(f"Error decoding embedding for idea {idea['title']}: {e}")

    if len(stored_embeddings) == 0:
        logging.error("No valid embeddings found in the ideas")
        return []

    stored_embeddings = np.vstack(stored_embeddings)
    logging.debug(f"Stored embeddings shape: {stored_embeddings.shape}")

    # Normalize the embeddings
    input_embedding = normalize(input_embedding.reshape(1, -1))
    stored_embeddings = normalize(stored_embeddings)

    # Calculate cosine similarity
    similarities = cosine_similarity(input_embedding, stored_embeddings).flatten()
    logging.debug(f"Similarities: {similarities}")
    top_k_indices = similarities.argsort()[-top_k:][::-1]

    similar_ideas_details = []
    for i in top_k_indices:
        similarity_score = float(similarities[i])
        similar_idea = valid_ideas[i]
        try:
            detail_summary_ko = deepl_translate(similar_idea['detailedSummary'], target_lang='KO')
        except Exception as e:
            logging.error(f"Error translating detail summary: {e}")
            detail_summary_ko = "Translation error"

        similar_ideas_details.append({
            "title": similar_idea['title'],
            "detailedSummary": detail_summary_ko,
            "similarity_score": similarity_score * 100 
        })

    logging.debug(f"Similar ideas details: {similar_ideas_details}")
    return similar_ideas_details