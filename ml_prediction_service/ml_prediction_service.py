"""
ML Prediction Microservice
===========================
A FastAPI-based microservice that provides machine learning predictions.
This service replaces the Java-based prediction logic with proper scikit-learn models.

Supported Prediction Types:
- TIME_SERIES: Linear Regression for forecasting future values
- WHAT_IF: Scenario analysis based on percentage changes
- CLASSIFICATION: (Future) Classification models

This service runs on port 8002 (separate from LLM service on 8001).
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import logging
from datetime import datetime
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ML Prediction Service",
    description="Machine Learning prediction service for time-series forecasting and what-if analysis",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class PredictionContext(BaseModel):
    """Chart/dashboard context"""
    chart: Optional[Dict[str, Any]] = None
    dashboard: Optional[Dict[str, Any]] = None


class PredictionRequest(BaseModel):
    """Request body for predictions - matches Java PredictionRequest DTO"""
    question: Optional[str] = Field(None, description="User's natural language question")
    reportData: List[Dict[str, Any]] = Field(..., description="The data rows to analyze")
    context: Optional[PredictionContext] = Field(None, description="Chart/dashboard metadata")


class PredictionResult(BaseModel):
    """Response matching Java PredictionResult DTO"""
    predictionType: str
    result: Dict[str, Any]
    confidence: float
    keyDrivers: Dict[str, Any]
    warning: Optional[str] = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_column_from_question(question: str, available_columns: List[str], 
                                  numeric_columns: List[str]) -> Optional[str]:
    """
    Extracts target column from user's question.
    Matches keywords in question against column names.
    """
    if not question:
        return None
    
    q = question.lower()
    
    # Score each numeric column
    best_match = None
    best_score = 0
    
    synonyms = {
        "sales": ["revenue", "income", "earnings", "turnover"],
        "quantity": ["count", "amount", "number", "units", "volume"],
        "price": ["cost", "rate", "value", "worth"],
        "profit": ["margin", "gain", "earnings", "net"],
        "total": ["sum", "aggregate", "overall", "combined"]
    }
    
    for column in numeric_columns:
        col_lower = column.lower()
        score = 0
        
        # Direct match
        if col_lower in q:
            score += 10
        
        # Match column parts (e.g., "total_sales" -> "total", "sales")
        for part in re.split(r'[_\s-]+', col_lower):
            if len(part) >= 3 and part in q:
                score += 5
        
        # Synonym matching
        for key, syns in synonyms.items():
            if key in col_lower:
                for syn in syns:
                    if syn in q:
                        score += 3
            for syn in syns:
                if syn in col_lower and key in q:
                    score += 3
        
        if score > best_score:
            best_score = score
            best_match = column
    
    return best_match if best_score >= 2 else None


def extract_forecast_horizon(question: str) -> int:
    """Extract forecast horizon from question. Default is 3."""
    if not question:
        return 3
    
    q = question.lower()
    
    # Pattern: "next X periods/months/quarters/years"
    pattern = r'(?:next|upcoming|following|forecast|predict)\s+(\d+)\s*(?:month|quarter|year|week|day|period|point|value|step)s?'
    match = re.search(pattern, q)
    if match:
        return min(int(match.group(1)), 24)  # Cap at 24
    
    # Special cases
    if "next year" in q or "coming year" in q:
        return 12
    if "next quarter" in q or "coming quarter" in q:
        return 3
    if "next month" in q or "coming month" in q:
        return 1
    
    # Look for any standalone number
    numbers = re.findall(r'\d+', q)
    for num in numbers:
        n = int(num)
        if 1 <= n <= 24:
            return n
    
    return 3


def extract_percentage_change(question: str) -> float:
    """Extract percentage change from question for what-if analysis."""
    if not question:
        return 0.0
    
    q = question.lower()
    
    # Determine if negative
    is_negative = any(word in q for word in 
                      ["decrease", "reduce", "drop", "fall", "decline", "lower", "cut"])
    
    # Pattern 1: "X%"
    match = re.search(r'(\d+(?:\.\d+)?)\s*%', q)
    if match:
        percent = float(match.group(1))
        return -percent / 100.0 if is_negative else percent / 100.0
    
    # Pattern 2: "by X"
    match = re.search(r'(?:increase|decrease|reduce|grow|raise|drop|change)\s+(?:by\s+)?(\d+(?:\.\d+)?)', q)
    if match:
        value = float(match.group(1))
        if value > 1:
            return -value / 100.0 if is_negative else value / 100.0
        return -value if is_negative else value
    
    return 0.0


def detect_prediction_type(question: str) -> str:
    """Detect whether this is TIME_SERIES or WHAT_IF prediction."""
    if not question:
        return "TIME_SERIES"
    
    q = question.lower()
    
    # WHAT_IF keywords (higher priority)
    what_if_keywords = [
        "what if", "what would", "what happens if",
        "increase by", "decrease by", "reduce by", "raise by",
        "grow by", "drop by", "cut by"
    ]
    
    if any(kw in q for kw in what_if_keywords):
        return "WHAT_IF"
    
    if ("if" in q and "%" in q) or ("scenario" in q and "%" in q):
        return "WHAT_IF"
    
    return "TIME_SERIES"


def detect_trend_type(values: List[float]) -> str:
    """Detect trend pattern in data."""
    if len(values) < 3:
        return "insufficient data"
    
    increases = sum(1 for i in range(1, len(values)) if values[i] > values[i-1])
    decreases = sum(1 for i in range(1, len(values)) if values[i] < values[i-1])
    
    total_changes = increases + decreases
    if total_changes == 0:
        return "stable/flat"
    
    change_ratio = increases / (total_changes + 1)
    
    # Check for cyclic pattern
    first, last = values[0], values[-1]
    middle = values[len(values) // 2]
    if abs(first - last) < abs(first - middle) * 0.5:
        return "cyclic"
    
    if change_ratio > 0.7:
        return "upward linear"
    elif change_ratio < 0.3:
        return "downward linear"
    elif change_ratio > 0.55:
        return "slight upward"
    elif change_ratio < 0.45:
        return "slight downward"
    else:
        return "stable/flat"


def extract_numeric_values(report_data: List[Dict], column: str) -> List[float]:
    """Extract numeric values from a specific column."""
    values = []
    for row in report_data:
        val = row.get(column)
        if val is not None:
            try:
                values.append(float(val))
            except (ValueError, TypeError):
                continue
    return values


def get_numeric_columns(report_data: List[Dict]) -> List[str]:
    """Identify columns that contain numeric data."""
    if not report_data:
        return []
    
    first_row = report_data[0]
    numeric_cols = []
    
    for col, val in first_row.items():
        if val is not None:
            try:
                float(val)
                numeric_cols.append(col)
            except (ValueError, TypeError):
                continue
    
    return numeric_cols


def select_target_column(question: str, context: Optional[PredictionContext], 
                         report_data: List[Dict]) -> str:
    """Select the target column for prediction."""
    all_columns = list(report_data[0].keys()) if report_data else []
    numeric_columns = get_numeric_columns(report_data)
    
    logger.info(f"Available columns: {all_columns}")
    logger.info(f"Numeric columns: {numeric_columns}")
    
    # Priority 1: Extract from question
    column = extract_column_from_question(question, all_columns, numeric_columns)
    if column:
        logger.info(f"Selected column from question: {column}")
        return column
    
    # Priority 2: Use YAxis from context
    if context and context.chart:
        y_axis = context.chart.get("YAxis")
        if y_axis and isinstance(y_axis, list) and len(y_axis) > 0:
            column = y_axis[0]
            if column in all_columns:
                logger.info(f"Selected column from YAxis: {column}")
                return column
    
    # Priority 3: First numeric column
    if numeric_columns:
        column = numeric_columns[0]
        logger.info(f"Selected first numeric column: {column}")
        return column
    
    raise ValueError("No suitable numeric column found for prediction")


# ============================================================================
# ML MODELS
# ============================================================================

class TimeSeriesPredictor:
    """
    Linear Regression-based time series forecasting.
    Uses scikit-learn's LinearRegression model.
    """
    
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
    
    def predict(self, values: List[float], forecast_horizon: int) -> Dict[str, Any]:
        """
        Perform time-series prediction using Linear Regression.
        
        Args:
            values: Historical numeric values
            forecast_horizon: Number of future periods to predict
            
        Returns:
            Dictionary containing forecast, confidence, and statistics
        """
        if len(values) < 2:
            raise ValueError("Need at least 2 data points for prediction")
        
        # Prepare data
        X = np.arange(len(values)).reshape(-1, 1)
        y = np.array(values)
        
        # Calculate statistics BEFORE any filtering
        raw_min = float(np.min(y))
        raw_max = float(np.max(y))
        raw_mean = float(np.mean(y))
        raw_median = float(np.median(y))
        
        # Detect and handle outliers using IQR method
        q1, q3 = np.percentile(y, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Filter outliers for model fitting (but keep track of them)
        outlier_mask = (y >= lower_bound) & (y <= upper_bound)
        outlier_count = int(len(y) - np.sum(outlier_mask))  # Convert to native Python int
        has_outliers = outlier_count > 0
        
        # Use filtered data if there are outliers and enough clean data remains
        if has_outliers and np.sum(outlier_mask) >= 2:
            y_clean = y[outlier_mask]
            X_clean = X[outlier_mask]
            logger.info(f"Detected {outlier_count} outliers. Using {len(y_clean)} clean values for model.")
        else:
            y_clean = y
            X_clean = X
        
        # Fit the model on clean data
        self.model.fit(X_clean, y_clean)
        
        # Calculate R² score on clean data
        y_pred_train = self.model.predict(X_clean)
        r2 = r2_score(y_clean, y_pred_train)
        
        # Generate forecast
        future_X = np.arange(len(values), len(values) + forecast_horizon).reshape(-1, 1)
        forecast = self.model.predict(future_X)
        
        # Calculate statistics from clean data
        clean_min = float(np.min(y_clean))
        clean_max = float(np.max(y_clean))
        clean_mean = float(np.mean(y_clean))
        
        # Cap predictions to reasonable bounds (can't go negative if data is positive)
        if clean_min >= 0:
            forecast = np.maximum(forecast, 0)
        
        # Cap predictions to not exceed 2x historical range
        forecast = np.minimum(forecast, clean_max * 2)
        forecast = np.maximum(forecast, clean_min * 0.1 if clean_min > 0 else clean_min)
        
        # Calculate growth rate from ACTUAL forecast values (not raw model)
        # This ensures the reported growth matches what users see
        if len(forecast) > 0 and clean_mean > 0:
            last_forecast = forecast[-1]
            # Growth from historical average to final forecast
            growth_rate = (last_forecast - clean_mean) / clean_mean
            # Cap at realistic bounds: -99% to +500%
            growth_rate = max(-0.99, min(5.0, growth_rate))
        else:
            growth_rate = 0.0
        
        # Determine trend direction for narrative
        if growth_rate > 0.05:
            trend_direction = "increasing"
        elif growth_rate < -0.05:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"
        
        # Calculate confidence (lower if poor fit or outliers)
        confidence = self._calculate_confidence(len(y_clean), r2)
        if has_outliers:
            confidence = max(confidence - 0.1, 0.5)  # Reduce confidence if outliers detected
        if r2 < 0.3:
            confidence = max(confidence - 0.15, 0.45)  # Reduce confidence if poor model fit
        
        # Build warning messages
        warnings = []
        if has_outliers:
            warnings.append(f"Data contains {outlier_count} outlier(s) which were excluded from analysis.")
        if r2 < 0.3:
            warnings.append("The historical pattern shows high variability, making forecasts less reliable.")
        if len(values) < 10:
            warnings.append("Limited historical data available.")
        
        warning_text = " ".join(warnings) if warnings else None
        
        return {
            "forecast": [float(round(f, 2)) for f in forecast.tolist()],
            "forecast_horizon": int(forecast_horizon),
            "confidence": float(round(confidence, 2)),
            "r_squared": float(round(r2, 4)),
            "growth_rate": float(round(growth_rate * 100, 1)),  # Now capped between -99% and +500%
            "trend_direction": trend_direction,
            "historical_avg": float(round(clean_mean, 2)),
            "historical_min": float(round(clean_min, 2)),
            "historical_max": float(round(clean_max, 2)),
            "historical_median": float(round(raw_median, 2)),
            "slope": float(round(float(self.model.coef_[0]), 4)),
            "intercept": float(round(float(self.model.intercept_), 4)),
            "data_points": int(len(values)),
            "clean_data_points": int(len(y_clean)),
            "outliers_detected": int(outlier_count),
            "warning": warning_text
        }
    
    def _calculate_confidence(self, n_points: int, r2: float) -> float:
        """Calculate confidence score based on data quality and model fit."""
        base_confidence = 0.6
        
        # Boost for data quantity
        if n_points >= 30:
            base_confidence += 0.15
        elif n_points >= 15:
            base_confidence += 0.10
        elif n_points >= 10:
            base_confidence += 0.05
        
        # Boost for R² score
        if r2 > 0.8:
            base_confidence += 0.15
        elif r2 > 0.6:
            base_confidence += 0.10
        elif r2 > 0.4:
            base_confidence += 0.05
        
        return min(round(base_confidence, 2), 0.95)


class WhatIfAnalyzer:
    """
    What-If scenario analysis.
    Calculates baseline from historical data and applies percentage changes.
    """
    
    def analyze(self, values: List[float], percentage_change: float) -> Dict[str, Any]:
        """
        Perform what-if analysis.
        
        Args:
            values: Historical numeric values
            percentage_change: The percentage change to apply (e.g., 0.10 for +10%)
            
        Returns:
            Dictionary containing baseline, scenario values, and analysis
        """
        if len(values) < 2:
            raise ValueError("Need at least 2 data points for analysis")
        
        arr = np.array(values)
        
        # Calculate baseline (mean of historical values)
        baseline = float(np.mean(arr))
        
        # Apply scenario change
        scenario_value = baseline * (1 + percentage_change)
        
        # Calculate additional statistics
        std_dev = float(np.std(arr))
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        
        # Confidence based on data quantity and variability
        cv = std_dev / baseline if baseline != 0 else 0  # Coefficient of variation
        confidence = 0.75 - min(cv * 0.1, 0.2)  # Lower confidence if high variability
        
        return {
            "baseline": float(round(baseline, 2)),
            "scenario": float(round(scenario_value, 2)),
            "change_percent": f"{round(percentage_change * 100, 1)}%",
            "absolute_change": float(round(scenario_value - baseline, 2)),
            "confidence": float(round(max(confidence, 0.5), 2)),
            "historical_std": float(round(std_dev, 2)),
            "historical_min": float(round(min_val, 2)),
            "historical_max": float(round(max_val, 2)),
            "data_points": int(len(values))
        }


# Initialize models
time_series_predictor = TimeSeriesPredictor()
what_if_analyzer = WhatIfAnalyzer()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "service": "ML Prediction Service",
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "models": ["LinearRegression", "WhatIfAnalysis"]
    }


@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": True,
        "supported_types": ["TIME_SERIES", "WHAT_IF"],
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/predict", response_model=PredictionResult)
def predict(request: PredictionRequest):
    """
    Main prediction endpoint.
    Automatically detects prediction type and routes to appropriate model.
    """
    try:
        logger.info("="*60)
        logger.info("PREDICTION REQUEST RECEIVED")
        logger.info("="*60)
        logger.info(f"Question: {request.question}")
        logger.info(f"Data rows: {len(request.reportData)}")
        
        if not request.reportData:
            raise HTTPException(status_code=400, detail="reportData is required")
        
        # Log sample data
        if request.reportData:
            logger.info(f"Columns: {list(request.reportData[0].keys())}")
            logger.info(f"Sample row: {request.reportData[0]}")
        
        # Detect prediction type
        pred_type = detect_prediction_type(request.question)
        logger.info(f"Detected prediction type: {pred_type}")
        
        # Select target column
        target_column = select_target_column(
            request.question, 
            request.context, 
            request.reportData
        )
        logger.info(f"Target column: {target_column}")
        
        # Extract values
        values = extract_numeric_values(request.reportData, target_column)
        logger.info(f"Extracted {len(values)} values from '{target_column}'")
        
        if len(values) < 2:
            raise HTTPException(
                status_code=400, 
                detail=f"Insufficient numeric data in column '{target_column}'. Need at least 2 values."
            )
        
        # Route to appropriate model
        if pred_type == "TIME_SERIES":
            return predict_timeseries(values, target_column, request.question)
        else:
            return predict_whatif(values, target_column, request.question)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def predict_timeseries(values: List[float], column: str, question: str) -> PredictionResult:
    """Execute time-series prediction."""
    forecast_horizon = extract_forecast_horizon(question)
    logger.info(f"Forecast horizon: {forecast_horizon}")
    
    # Run prediction
    result = time_series_predictor.predict(values, forecast_horizon)
    
    logger.info(f"Prediction result: forecast={result['forecast']}, growth_rate={result['growth_rate']}%, confidence={result['confidence']}")
    
    # Build response with user-friendly fields
    return PredictionResult(
        predictionType="TIME_SERIES",
        result={
            "forecast": result["forecast"],
            "forecastHorizon": result["forecast_horizon"],
            "target_metric": column,
            "historical_avg": result["historical_avg"],
            "historical_min": result["historical_min"],
            "historical_max": result["historical_max"],
            "trend_direction": result["trend_direction"],  # "increasing", "decreasing", or "stable"
        },
        confidence=result["confidence"],
        keyDrivers={
            "predicted_column": column,
            "trend": result["trend_direction"],  # Use calculated trend direction
            "growth_rate": f"{result['growth_rate']}%",
            "data_points": result["clean_data_points"],  # Use clean data points
            "model": "Linear Regression (scikit-learn)",
            "outliers_removed": result["outliers_detected"]
        },
        warning=result.get("warning")  # Use warning from model
    )


def predict_whatif(values: List[float], column: str, question: str) -> PredictionResult:
    """Execute what-if analysis."""
    percentage_change = extract_percentage_change(question)
    logger.info(f"Percentage change: {percentage_change * 100}%")
    
    # Run analysis
    result = what_if_analyzer.analyze(values, percentage_change)
    
    # Build response
    return PredictionResult(
        predictionType="WHAT_IF",
        result={
            "target_metric": column,
            "baseline": result["baseline"],
            "scenario": result["scenario"],
            "change_percent": result["change_percent"],
            "absolute_change": result["absolute_change"]
        },
        confidence=result["confidence"],
        keyDrivers={
            "analyzed_column": column,
            "assumption": "linear proportional impact",
            "historical_points": result["data_points"],
            "historical_std": result["historical_std"],
            "model": "What-If Analysis (Python)"
        },
        warning="Warning: Very few data points." if len(values) < 5 else None
    )


@app.post("/predict/timeseries")
def predict_timeseries_direct(request: PredictionRequest):
    """Direct endpoint for time-series prediction (bypasses auto-detection)."""
    target_column = select_target_column(request.question, request.context, request.reportData)
    values = extract_numeric_values(request.reportData, target_column)
    return predict_timeseries(values, target_column, request.question)


@app.post("/predict/whatif")
def predict_whatif_direct(request: PredictionRequest):
    """Direct endpoint for what-if analysis (bypasses auto-detection)."""
    target_column = select_target_column(request.question, request.context, request.reportData)
    values = extract_numeric_values(request.reportData, target_column)
    return predict_whatif(values, target_column, request.question)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting ML Prediction Service")
    print("   Port: 8002")
    print("   Models: LinearRegression, WhatIfAnalysis")
    print("   Endpoints:")
    print("     - POST /predict (auto-detect type)")
    print("     - POST /predict/timeseries")
    print("     - POST /predict/whatif")
    print("     - GET /health\n")
    uvicorn.run(app, host="0.0.0.0", port=8002)

