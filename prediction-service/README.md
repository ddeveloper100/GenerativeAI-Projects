# Predictive Model ML Services

This folder contains two Python microservices for the DynamicReportBuilder prediction system:

## Services Overview

| Service | Port | Purpose |
|---------|------|---------|
| `ml_prediction_service.py` | 8002 | ML predictions (Linear Regression, What-If Analysis) using scikit-learn |
| `llm_explanation_layer.py` | 8001 | Human-readable explanations using Ollama LLM |

---

## 1. ML Prediction Service (`ml_prediction_service.py`)

### Features
- **Time-Series Forecasting**: Linear Regression using scikit-learn
- **What-If Analysis**: Scenario-based predictions
- **Auto Column Detection**: Identifies target metric from user questions
- **Confidence Calculation**: Based on R² score and data quantity

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Service info |
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Auto-detect prediction type and run |
| `POST` | `/predict/timeseries` | Direct time-series forecasting |
| `POST` | `/predict/whatif` | Direct what-if analysis |

### Request Format
```json
{
  "question": "What will sales be in the next 3 months?",
  "reportData": [
    {"MONTH": "Jan", "SALES": 1000, "QUANTITY": 50},
    {"MONTH": "Feb", "SALES": 1200, "QUANTITY": 60}
  ],
  "context": {
    "chart": {
      "YAxis": ["SALES"],
      "XAxis": ["MONTH"]
    }
  }
}
```

### Response Format
```json
{
  "predictionType": "TIME_SERIES",
  "result": {
    "forecast": [1400.0, 1600.0, 1800.0],
    "forecastHorizon": 3,
    "target_metric": "SALES",
    "r_squared": 0.95
  },
  "confidence": 0.85,
  "keyDrivers": {
    "predicted_column": "SALES",
    "trend": "upward linear",
    "growth_rate": "20%",
    "model": "Linear Regression (scikit-learn)"
  }
}
```

### Running the Service
```bash
cd C:\Users\danisha.ansari\Product\PredictiveModelLLM

# Install dependencies
pip install -r requirements.txt

# Run the service
python ml_prediction_service.py

# Or with uvicorn (auto-reload for development)
uvicorn ml_prediction_service:app --reload --port 8002
```

---

## 2. LLM Explanation Service (`llm_explanation_layer.py`)

### Features
- Generates human-readable explanations for predictions
- Uses Ollama with Llama2 or Phi models
- Provides data-driven insights

### Requirements
- **Ollama** must be running locally
- Download a model: `ollama pull llama2` or `ollama pull phi`

### Running the Service
```bash
# Make sure Ollama is running first
ollama serve

# Then run the LLM service
python llm_explanation_layer.py

# Or with uvicorn
uvicorn llm_explanation_layer:app --reload --port 8001
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Java Spring Boot Backend                      │
│                        (Port 8000)                               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              PredictionOrchestrator                         │ │
│  │                                                              │ │
│  │   1. Receives prediction request                             │ │
│  │   2. Tries Python ML Service (port 8002)                     │ │
│  │   3. Falls back to Java models if unavailable                │ │
│  │   4. Calls LLM Service for explanation (port 8001)           │ │
│  └────────────────────────────────────────────────────────────┘ │
└───────────────────────────┬──────────────────┬──────────────────┘
                            │                  │
              ┌─────────────▼────────┐  ┌──────▼──────────────────┐
              │  ML Prediction Svc   │  │   LLM Explanation Svc   │
              │    (Port 8002)       │  │      (Port 8001)        │
              │                      │  │                          │
              │  - Linear Regression │  │  - Ollama API            │
              │  - What-If Analysis  │  │  - Natural Language      │
              │  - scikit-learn      │  │  - Llama2/Phi models     │
              └──────────────────────┘  └──────────────────────────┘
```

---

## Configuration

In `application.properties` (Java backend):

```properties
# ML Prediction Service
ml.prediction.api.url=http://localhost:8002
ml.prediction.api.enabled=true
ml.prediction.use-python-service=true
ml.prediction.fallback-to-java=true

# LLM Explanation Service  
llm.explanation.api.url=http://localhost:8001
llm.explanation.enabled=true
```

### Configuration Options

| Property | Default | Description |
|----------|---------|-------------|
| `ml.prediction.use-python-service` | `true` | Use Python ML service for predictions |
| `ml.prediction.fallback-to-java` | `true` | Fall back to Java models if Python fails |
| `ml.prediction.api.enabled` | `true` | Enable/disable Python ML service |

---

## Quick Start (All Services)

### Terminal 1: Java Backend
```bash
cd C:\Users\danisha.ansari\Product\DynamicReportBuilderBackend
mvn spring-boot:run
```

### Terminal 2: ML Prediction Service
```bash
cd C:\Users\danisha.ansari\Product\PredictiveModelLLM
pip install -r requirements.txt
python ml_prediction_service.py
```

### Terminal 3: LLM Explanation Service
```bash
cd C:\Users\danisha.ansari\Product\PredictiveModelLLM
# Make sure Ollama is running: ollama serve
python llm_explanation_layer.py
```

---

## Testing

### Test ML Service
```bash
curl -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What will sales be next month?",
    "reportData": [
      {"MONTH": "Jan", "SALES": 1000},
      {"MONTH": "Feb", "SALES": 1100},
      {"MONTH": "Mar", "SALES": 1200}
    ]
  }'
```

### Test Health
```bash
curl http://localhost:8002/health
curl http://localhost:8001/health
```

