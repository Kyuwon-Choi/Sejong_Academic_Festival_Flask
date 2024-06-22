import requests
import logging

def fetch_ideas_from_spring():
    logging.debug("Fetching ideas from Spring Boot API")
    try:
        response = requests.get('http://localhost:8080/api/ideas')
        logging.debug(f"Response status code: {response.status_code}")
        if response.status_code == 200:
            ideas = response.json()
            logging.debug(f"Fetched ideas: {ideas}")
            return ideas
        else:
            logging.error(f"Failed to fetch ideas: {response.text}")
            return []
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching ideas from Spring Boot API: {e}")
        return []
