import os 
from dotenv import load_dotenv

load_dotenv()

class Settings:
    Model_Path = os.getenv("MODEL_PATH", "models/svm_model.pkl")
    vectorizer_path = os.getenv("VECTORIZER_PATH", "models/tfidf_vectorizer.pkl")

settings = Settings()