from fastapi import FastAPI, HTTPException,Response,Request
from datetime import datetime,UTC
import time
import logging
from app.metrics import PREDICTION_COUNT, REQUEST_COUNT, REQUEST_LATENCY
from models.services import predict_spam, batch_pred_spam, get_model, get_vectorizer
from schemas.schema import PredictionResponse, BatchPredictionResponse, TextRequest, BatchRequest
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import os
from app.prediction_log import log_prediction



app = FastAPI(
    title="spam_detector_api",
    description="A simple API for detecting spam emails",
    version="1.0.0"
)

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/spam_detector_api.log"),
        logging.StreamHandler()
    ]
)


logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request, call_next):
    start_time =  time.time()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Error occurred while processing request")
        raise
    end_time = time.time()
    process_time = end_time - start_time
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path, 
        http_status=response.status_code).inc()
    REQUEST_LATENCY.labels(
        endpoint=request.url.path
    ).observe(process_time)
    logger.info(
        f"{request.method} {request.url} - {response.status_code} - {process_time:.2f}s"
    )
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
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
    
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Spam Detector API shutting down...")

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


@app.get("/metrics")
def metrics():
    logger.info("Metrics endpoint accessed.")
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.post("/single_predict", response_model= PredictionResponse)
async def predict(data: TextRequest):
    logger.info("Single predict endpoint accessed.")
    
    if not data.text or not data.text.strip():
        logger.warning("Empty text input received.")
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")
    
    logger.info("Prediction Request Received")
    try:
        result = predict_spam(data.text)
        PREDICTION_COUNT.labels(label=result['label']).inc()
        log_prediction(text_length=len(data.text),label=result["label"])
    
    except HTTPException:
        raise 
    
    except Exception:
        logger.exception("Single prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error"
        )

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
        for i, prediction in enumerate(results["results"]):
            PREDICTION_COUNT.labels(
            label=prediction["prediction"]
            ).inc()
            log_prediction(text_length=len(data.texts[i]),label=prediction["prediction"])


    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error"
        )
    
    logger.info("Batch prediction completed successfully.")
    return results