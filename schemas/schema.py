from pydantic import BaseModel

class TextRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: list[str]

class PredictionResponse(BaseModel):
    prediction: str

class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]