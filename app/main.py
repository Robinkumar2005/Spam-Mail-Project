from fastapi import FastAPI, HTTPException
from datetime import datetime,UTC
import time
import logging
from models.services import predict_spam, batch_pred_spam, get_model, get_vectorizer
from schemas.schema import PredictionResponse, BatchPredictionResponse, TextRequest, BatchRequest

app = FastAPI(
    title="spam_detector_api",
    description="A simple API for detecting spam emails",
    version="1.0.0"
)

logging.basicConfig(
    filename="logs/spam_detector_api.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request, call_next):
    start_time =  time.time()
    response = await call_next(request)
    end_time = time.time()
    process_time = end_time - start_time
    logger.info(
        f"{request.method} {request.url} - {response.status_code} - {process_time:.2f}s"
    )
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the Spam Detector API...")
    try:
        get_model()
        get_vectorizer()
        logger.info("Model and vectorizer loaded successfully.")
    except Exception as e:
        logger.critical(f"Startup failed: {str(e)}")
        raise RuntimeError("Failed to start application")

@app.get("/")
def home():
    logger.info("Home endpoint accessed.")
    return {"message": "Welcome to the Spam Detector API!"}

@app.get("/model-info")
def model_info():
    logger.info("Model info endpoint accessed.")
    return {
        "model_name": "SVM_MODEL",
        "version": "1.0.0",
        "description": "A NLP model for detecting spam emails."
    }
@app.get("/health")
def health_check():
    logger.info("Health check endpoint accessed.")
    return {
        "status": "healthy",
        "model_loaded": get_model() is not None,
        "vectorizer_loaded": get_vectorizer() is not None,
        "timestamp": datetime.now(UTC).isoformat()
    }


@app.post("/single_predict", response_model= PredictionResponse)
async def predict(data: TextRequest):
    logger.info("Single predict endpoint accessed.")
    
    if not data.text or not data.text.strip():
        logger.warning("Empty text input received.")
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")
    
    logger.info("Prediction Request Received")
    try:
        result = predict_spam(data.text)
    
    except HTTPException:
        raise 

    except Exception as e:
        logger.error(f"Error during prediction")
        raise HTTPException(status_code=500, detail=str(e))
    
    logger.info("Prediction completed successfully.")
    return result

@app.post("/batch_predict", response_model= BatchPredictionResponse)
async def batch_predict(data: BatchRequest):
    logger.info("Batch predict endpoint accessed.")
    
    if not data.texts:
        logger.warning("Empty text list received.")
        raise HTTPException(status_code=400, detail="Text list cannot be empty.")
    
    logger.info(f"Received {len(data.texts)} texts for batch prediction.")
    
    for text in data.texts:
        if not text.strip():
            logger.warning("Empty text input found in batch.")
            raise HTTPException(status_code=400, detail="Text inputs cannot be empty.")
    logger.info(
        f"Batch prediction started for {len(data.texts)} texts."
    )

    try:
        results = batch_pred_spam(data.texts)

    except HTTPException:
        raise

    except Exception as e:
        import traceback
        # logger.error(f"Error during batch prediction")
        return{
            "error":str(e),
            "traceback": traceback.format_exc()
        }
    
    logger.info("Batch prediction completed successfully.")
    return results