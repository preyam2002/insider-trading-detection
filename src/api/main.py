from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import joblib
import numpy as np
import os

from src.models.insider_models import RandomForestModel, XGBoostModel

app = FastAPI(title="Insider Trading Detection API")

model = None
scaler = None
model_type = "random_forest"


class TradeFeatures(BaseModel):
    return_7d: float = 0.0
    return_14d: float = 0.0
    return_30d: float = 0.0
    return_60d: float = 0.0
    volatility_7d: float = 0.0
    volatility_14d: float = 0.0
    volatility_30d: float = 0.0
    volatility_60d: float = 0.0
    sma_ratio_50_200: float = 1.0
    rsi_14d: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    volume_ratio_7d: float = 1.0
    volume_ratio_14d: float = 1.0
    volume_ratio_30d: float = 1.0
    insider_buys_30d: float = 0.0
    insider_sells_30d: float = 0.0
    insider_net_activity: float = 0.0
    market_cap: float = 0.0
    pe_ratio: float = 0.0
    insider_percent: float = 0.0
    institutional_percent: float = 0.0
    beta: float = 1.0
    abnormal_return_7d: float = 0.0
    abnormal_return_14d: float = 0.0
    abnormal_return_30d: float = 0.0
    abnormal_return_60d: float = 0.0


class DetectionRequest(BaseModel):
    features: TradeFeatures


class DetectionResponse(BaseModel):
    is_insider_trade: bool
    probability: float
    confidence: float
    flags: List[str]


@app.on_event("startup")
def load_model():
    global model, scaler, model_type
    
    model_path = os.environ.get("MODEL_PATH", "models/random_forest.pkl")
    
    if os.path.exists(model_path):
        data = joblib.load(model_path)
        model = data['model']
        scaler = data['scaler']
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}")


@app.get("/")
def root():
    return {"message": "Insider Trading Detection API", "status": "running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/detect", response_model=DetectionResponse)
def detect_insider_trade(request: DetectionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    features = request.features
    feature_vector = np.array([[
        features.return_7d, features.return_14d, features.return_30d, features.return_60d,
        features.volatility_7d, features.volatility_14d, features.volatility_30d, features.volatility_60d,
        features.sma_ratio_50_200, features.rsi_14d, features.macd, features.macd_signal,
        features.volume_ratio_7d, features.volume_ratio_14d, features.volume_ratio_30d,
        features.insider_buys_30d, features.insider_sells_30d, features.insider_net_activity,
        features.market_cap, features.pe_ratio, features.insider_percent, features.institutional_percent, features.beta,
        features.abnormal_return_7d, features.abnormal_return_14d, features.abnormal_return_30d, features.abnormal_return_60d
    ]])
    
    X_scaled = scaler.transform(feature_vector)
    proba = model.predict_proba(X_scaled)[0]
    probability = float(proba[1])
    
    is_insider = probability > 0.5
    confidence = abs(probability - 0.5) * 2
    
    flags = []
    if abs(features.abnormal_return_30d) > 0.10:
        flags.append("High abnormal return")
    if features.insider_net_activity > 5:
        flags.append("High insider buying activity")
    if features.rsi_14d > 70:
        flags.append("Overbought (RSI > 70)")
    if features.rsi_14d < 30:
        flags.append("Oversold (RSI < 30)")
    if features.volume_ratio_30d > 3.0:
        flags.append("Unusual volume")
    if features.return_30d > 0.20:
        flags.append("High 30-day return")
    
    return DetectionResponse(
        is_insider_trade=is_insider,
        probability=probability,
        confidence=confidence,
        flags=flags
    )


@app.post("/detect_batch")
def detect_batch(trades: List[DetectionRequest]):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    results = []
    
    for trade in trades:
        features = trade.features
        feature_vector = np.array([[
            features.return_7d, features.return_14d, features.return_30d, features.return_60d,
            features.volatility_7d, features.volatility_14d, features.volatility_30d, features.volatility_60d,
            features.sma_ratio_50_200, features.rsi_14d, features.macd, features.macd_signal,
            features.volume_ratio_7d, features.volume_ratio_14d, features.volume_ratio_30d,
            features.insider_buys_30d, features.insider_sells_30d, features.insider_net_activity,
            features.market_cap, features.pe_ratio, features.insider_percent, features.institutional_percent, features.beta,
            features.abnormal_return_7d, features.abnormal_return_14d, features.abnormal_return_30d, features.abnormal_return_60d
        ]])
        
        X_scaled = scaler.transform(feature_vector)
        proba = model.predict_proba(X_scaled)[0]
        
        results.append({
            "probability": float(proba[1]),
            "is_insider_trade": bool(proba[1] > 0.5)
        })
    
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
