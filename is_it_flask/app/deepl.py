import requests
import logging

DEEPL_API_KEY = '2a99de55-b6dd-4743-9f3e-615b6c4d8d8f:fx'

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
