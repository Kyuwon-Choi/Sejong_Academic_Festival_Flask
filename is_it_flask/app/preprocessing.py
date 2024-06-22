import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# NLTK 전처리 도구 로드
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def stem_sentence(sentence):
    if not isinstance(sentence, str):
        return ""
    tokenized_sentence = word_tokenize(sentence)
    stemmed_sentence = [stemmer.stem(word) for word in tokenized_sentence]
    return ' '.join(stemmed_sentence)

def lemmatize_sentence(sentence):
    if not isinstance(sentence, str):
        return ""
    tokenized_sentence = word_tokenize(sentence)
    lemmatized_sentence = [lemmatizer.lemmatize(word) for word in tokenized_sentence]
    return ' '.join(lemmatized_sentence)

def remove_stopwords(sentence):
    if not isinstance(sentence, str):
        return ""
    tokenized_sentence = word_tokenize(sentence)
    filtered_sentence = [word for word in tokenized_sentence if word.lower() not in stop_words]
    return ' '.join(filtered_sentence)

def preprocess_text(text):
    no_stopwords = remove_stopwords(text)
    lemmatized = lemmatize_sentence(no_stopwords)
    stemmed = stem_sentence(lemmatized)
    return stemmed
