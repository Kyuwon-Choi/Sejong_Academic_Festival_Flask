from flask import request, jsonify, current_app as app
from .deepl import deepl_translate
from .fetch_ideas import fetch_ideas_from_spring
from .embeddings import find_similar_ideas
import logging

@app.route('/similar-ideas', methods=['POST'])
def similar_ideas():
    try:
        data = request.get_json()
        service_name = data.get('serviceName')
        service_detail_ko = data.get('serviceDetail')
        logging.debug(f"Received service name: {service_name}, service detail: {service_detail_ko}")

        # Translate service detail to English
        service_detail_en = deepl_translate(service_detail_ko, target_lang='EN')
        
        if service_detail_en is None:
            return jsonify({"error": "Translation failed"}), 500
        
        logging.debug(f"Translated service detail: {service_detail_en}")

        # Fetch stored ideas from Spring Boot API
        ideas = fetch_ideas_from_spring()
        if not ideas:
            return jsonify({"error": "Failed to fetch ideas from Spring Boot API"}), 500
        
        # Find similar ideas
        similar_ideas = find_similar_ideas(service_detail_en, ideas)
        logging.debug(f"Similar ideas: {similar_ideas}")
        return jsonify(similar_ideas)
    
    except Exception as e:
        logging.error(f"Internal server error: {e}")
        return jsonify({"error": "Internal server error"}), 500
