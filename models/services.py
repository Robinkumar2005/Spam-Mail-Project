import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List
from fastapi import HTTPException
from .config import settings
import joblib
import logging
import string

logger = logging.getLogger(__name__)
model = None
vectorizer = None


def get_model():
    global model
    if model is None:
        try:
            model =  joblib.load(settings.Model_Path)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError("Failed to load model.")
    return model

def get_vectorizer():
    global vectorizer
    if vectorizer is None:
        try:
            vectorizer = joblib.load(settings.vectorizer_path)
            logger.info("Vectorizer loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading vectorizer: {str(e)}")
            raise RuntimeError("Failed to load vectorizer.")
    return vectorizer



def remove_punc(txt):
  return txt.translate(str.maketrans('','',string.punctuation))


def remove_numbers(txt):
  new = ""
  for i in txt:
    if not i.isdigit():
      new += i
  return new

def remove_emojis(txt):
  new = ""
  for i in txt:
    if i.isascii():
      new += i
  return new


stop_words = set(stopwords.words('english'))


def remove_stopwords(txt):
  words = word_tokenize(txt)
  new = [word for word in words if word.lower() not in stop_words]
  return ' '.join(new)

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = remove_emojis(text)
    text = remove_punc(text)
    text = remove_numbers(text)
    text = remove_stopwords(text)
    return text

def predict_spam(text: str):
   if len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")
   text = preprocess_text(text)
   vectorizer = get_vectorizer()
   text_vector = vectorizer.transform([text])
   model = get_model()
   prediction = model.predict(text_vector)
   label = "spam" if prediction[0] == 1 else "ham"
   
   return {
      "prediction": label,
    }

def batch_pred_spam(texts: List[str]):
   text_list = []
   for text in texts:
      if len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")
      text_list.append(text)
   text_list = [preprocess_text(text) for text in text_list]

   vectorizer = get_vectorizer()
   text_vector = vectorizer.transform(text_list)
   model = get_model()
   predictions = model.predict(text_vector)
   labels = ["spam" if pred == 1 else "ham" for pred in predictions]
   return {
    "results": [
        {"prediction": label}
        for label in labels
    ]
}