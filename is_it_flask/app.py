from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import requests
import logging
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # CORS 설정 추가

logging.basicConfig(level=logging.DEBUG)

# BERT 모델과 토크나이저 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

# DeepL API 설정
DEEPL_API_KEY = ''  

def deepl_translate(text, target_lang='EN'):
    logging.debug(f"Translating text: {text} to {target_lang}")
    url = "https://api-free.deepl.com/v2/translate"
    data = {'auth_key': DEEPL_API_KEY, 'text': text, 'target_lang': target_lang}
    response = requests.post(url, data=data)
    
    if response.status_code != 200:
        logging.error(f"Failed to translate text: {response.text}")
        return None

    try:
        result = response.json()
        logging.debug(f"Translation result: {result}")
        return result['translations'][0]['text']
    except requests.exceptions.JSONDecodeError as e:
        logging.error(f"JSON decode error: {e}")
        return None

# Spring에서 데이터 가져오기
def fetch_ideas_from_spring():
    logging.debug("Fetching ideas from Spring Boot API")
    response = requests.get('http://localhost:8080/api/ideas')
    logging.debug(f"Response status code: {response.status_code}")
    if response.status_code == 200:
        ideas = response.json()
        logging.debug(f"Fetched ideas: {ideas}")
        return ideas
    else:
        logging.error("Failed to fetch ideas from Spring Boot API")
        return []

# 배치 단위로 입력 텍스트를 BERT 토큰으로 변환
def embed_ideas(ideas, max_length=512):
    texts = [f"{idea['title']} {idea['briefSummary']} {idea['detailSummary']}" for idea in ideas]
    encoded_input = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    return model_output.last_hidden_state.mean(dim=1).cpu().numpy()

def find_similar_ideas(input_idea, ideas, top_k=5):
    logging.debug(f"Finding similar ideas for: {input_idea}")
    embedded_ideas = embed_ideas(ideas)
    input_embedding = embed_ideas([{"title": "", "briefSummary": "", "detailSummary": input_idea}])
    
    similarities = cosine_similarity(input_embedding, embedded_ideas)
    top_k_indices = similarities[0].argsort()[-top_k:][::-1]
    
    similar_ideas_details = []
    for i in top_k_indices:
        similar_idea = ideas[i]
        similarity_score = float(similarities[0][i])
        
        brief_summary_ko = deepl_translate(similar_idea['briefSummary'], target_lang='KO')
        detail_summary_ko = deepl_translate(similar_idea['detailSummary'], target_lang='KO')

        similar_ideas_details.append({
            "title": similar_idea['title'],  # Keep title in English
            "briefSummary": brief_summary_ko,
            "detailSummary": detail_summary_ko,
            "similarity_score": similarity_score * 100  # Convert to percentage
        })
    
    logging.debug(f"Similar ideas details: {similar_ideas_details}")
    return similar_ideas_details

@app.route('/similar-ideas', methods=['POST'])
def similar_ideas():
    data = request.get_json()
    service_name = data.get('serviceName')
    service_detail_ko = data.get('serviceDetail')
    logging.debug(f"Received service name: {service_name}, service detail: {service_detail_ko}")
    
    service_detail_en = deepl_translate(service_detail_ko, target_lang='EN')
    
    if service_detail_en is None:
        return jsonify({"error": "Translation failed"}), 500
    
    logging.debug(f"Translated service detail: {service_detail_en}")

    ideas = fetch_ideas_from_spring()
    if not ideas:
        return jsonify({"error": "Failed to fetch ideas from Spring Boot API"}), 500
    
    similar_ideas = find_similar_ideas(service_detail_en, ideas)
    logging.debug(f"Similar ideas: {similar_ideas}")
    return jsonify(similar_ideas)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
