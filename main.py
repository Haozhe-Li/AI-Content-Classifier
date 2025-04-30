from core.classifier import AIContentClassifier
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

classifier = AIContentClassifier()

class TextRequest(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "AI Content Classifier"}

@app.post("/classify")
async def classify_text(request: TextRequest):
    result = await classifier.classify(request.text)
    return result