"""
FastAPI service for generating LLM-powered prediction explanations.
Replaces the hardcoded ExplanationService in the Java backend.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import requests
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Prediction Explanation Service",
    description="LLM-powered explanation generation for prediction results",
    version="1.0.0"
)

# Configure CORS to allow Java backend to call this service
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Java backend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Ollama configuration for local Llama 2
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://10.80.3.19:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama2:latest")  # Using phi for faster responses


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class PredictionResult(BaseModel):
    """Matches the Java PredictionResult DTO"""
    predictionType: str = Field(..., description="Type of prediction (TIME_SERIES, WHAT_IF, etc.)")
    result: Dict[str, Any] = Field(..., description="Core prediction output")
    confidence: Optional[float] = Field(None, description="Confidence score (0.0 - 1.0)")
    keyDrivers: Optional[Dict[str, Any]] = Field(None, description="Key factors influencing the prediction")
    warning: Optional[str] = Field(None, description="Warning or note about the prediction")
    explanation: Optional[str] = Field(None, description="Human-readable explanation")


class ExplanationRequest(BaseModel):
    """Request body for explanation generation"""
    predictionResult: PredictionResult
    context: Optional[Dict[str, Any]] = Field(None, description="Chart/dashboard metadata")
    question: Optional[str] = Field(None, description="User's original question")


class ExplanationResponse(BaseModel):
    """Response containing the generated explanation"""
    explanation: str
    generated_at: str
    tokens_used: Optional[int] = None


# ============================================================================
# LLM EXPLANATION GENERATION
# ============================================================================

def build_llm_prompt(
    prediction_result: PredictionResult,
    context: Optional[Dict[str, Any]] = None,
    user_question: Optional[str] = None
) -> str:
    """
    Constructs a detailed prompt for the LLM to generate a natural,
    human-readable explanation of the prediction results.
    """
    
    prompt_parts = [
        "You are a business analyst explaining predictions to non-technical stakeholders.",
        "Write a clear, narrative explanation that anyone can understand.",
        "\n=== RAW DATA (for your reference only - do not expose technical details) ===\n"
    ]
    
    # Add user's original question if available
    if user_question:
        prompt_parts.append(f"User's Question: {user_question}\n")
    
    # Add prediction type
    prompt_parts.append(f"Analysis Type: {prediction_result.predictionType}")
    
    # Add prediction results
    prompt_parts.append(f"\nRaw Results:")
    for key, value in prediction_result.result.items():
        prompt_parts.append(f"  - {key}: {value}")
    
    # Add confidence score
    confidence_pct = 75  # default
    if prediction_result.confidence is not None:
        confidence_pct = round(prediction_result.confidence * 100)
        prompt_parts.append(f"\nModel Confidence: {confidence_pct}%")
    
    # Add key drivers
    if prediction_result.keyDrivers:
        prompt_parts.append(f"\nTechnical Factors:")
        for key, value in prediction_result.keyDrivers.items():
            prompt_parts.append(f"  - {key}: {value}")
    
    # Add warning if present
    if prediction_result.warning:
        prompt_parts.append(f"\nNote: {prediction_result.warning}")
    
    # Add context information
    if context:
        prompt_parts.append(f"\nContext:")
        if isinstance(context, dict):
            for key, value in context.items():
                prompt_parts.append(f"  - {key}: {value}")
    
    # Build narrative instructions
    prompt_parts.append("\n=== YOUR TASK ===")
    
    # Add confidence interpretation text
    if confidence_pct >= 80:
        confidence_text = "relatively high confidence"
    elif confidence_pct >= 60:
        confidence_text = "moderate uncertainty"
    else:
        confidence_text = "limited confidence"
    
    prompt_parts.append(
        f"Write exactly 3 short paragraphs. The prediction has {confidence_text}.\n\n"
        "CRITICAL RULES - YOU MUST FOLLOW THESE:\n"
        "- Output ONLY the explanation text. Nothing else.\n"
        "- DO NOT write 'Here is my explanation' or 'PARAGRAPH 1' or any labels/headers.\n"
        "- DO NOT use words like 'model', 'regression', 'R-squared', 'data points', 'slope'.\n"
        "- DO NOT include exact percentages or technical numbers.\n"
        "- Just write 3 plain paragraphs separated by blank lines.\n"
        "- Start directly with the content like 'Over the past period...'\n"
    )
    
    # Add type-specific example
    if prediction_result.predictionType == "TIME_SERIES":
        prompt_parts.append(
            "\n The Response should be in paragraphs"
            "\nWrite something EXACTLY like this format (but with appropriate content):\n\n"
            "Over the past period, the average transaction amount has been declining. "
            "If this pattern continues, transaction values may drop significantly over the next two years.\n\n"
            "The projected decrease is substantial, which suggests that current trends may not be sustainable long term. "
            "However, this forecast is based purely on historical patterns and assumes no external changes.\n\n"
            "The prediction carries " + confidence_text + ", meaning actual outcomes could vary depending on future market conditions."
        )
    elif prediction_result.predictionType == "WHAT_IF":
        prompt_parts.append(
            "\nWrite something EXACTLY like this format (but with appropriate content):\n\n"
            "Currently, the average value stands at a certain level based on recent performance.\n\n"
            "Under the proposed scenario, values would shift noticeably. "
            "This change could have meaningful implications for planning and resource allocation.\n\n"
            "This projection assumes a direct relationship between the changed factor and outcomes. "
            "Real-world results may differ based on other influencing factors."
        )
    else:
        prompt_parts.append(
            "\nWrite 3 short paragraphs: first about what the data shows, second about what might happen, "
            "third about uncertainty and limitations. Don't use preamble like 'Here is my explanation' or 'PARAGRAPH 1' or any labels/headers, just start with the response."
        )
    
    return "\n".join(prompt_parts)


def generate_explanation_with_llm(
    prediction_result: PredictionResult,
    context: Optional[Dict[str, Any]] = None,
    user_question: Optional[str] = None
) -> tuple[str, int]:
    """
    Calls Ollama Llama 2 API to generate a human-readable explanation.
    
    Returns:
        tuple: (explanation_text, tokens_used)
    """
    
    try:
        # Build the prompt
        prompt = build_llm_prompt(prediction_result, context, user_question)
        
        logger.info(f"Generating explanation for prediction type: {prediction_result.predictionType}")
        logger.debug(f"Prompt: {prompt}")
        
        # Call Ollama API
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.5,  # Balanced temperature for natural but focused narrative
                "options": {
                    "num_predict": 350  # Allow ~350 tokens for 3-paragraph narrative
                }
            },
            timeout=None  # No timeout - wait indefinitely for LLM response
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
        
        result = response.json()
        explanation = result.get("response", "")
        
        # Ollama doesn't provide token count, so we estimate
        tokens_used = len(explanation.split()) + len(prompt.split())
        
        logger.info(f"Generated explanation successfully. Estimated tokens: {tokens_used}")
        logger.debug(f"Explanation: {explanation}")
        
        return explanation, tokens_used
        
    except requests.exceptions.ConnectionError:
        logger.error(f"Failed to connect to Ollama at {OLLAMA_BASE_URL}. Make sure it's running.")
        raise HTTPException(
            status_code=503,
            detail=f"Ollama service unavailable at {OLLAMA_BASE_URL}"
        )
    except requests.exceptions.Timeout:
        logger.error("Ollama request timed out")
        raise HTTPException(
            status_code=504,
            detail="Ollama request timed out. The model may be processing a large request."
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "service": "Prediction Explanation Service",
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health")
def health_check():
    """Detailed health check including Ollama availability"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=None)
        ollama_available = response.status_code == 200
    except:
        ollama_available = False
    
    return {
        "status": "healthy" if ollama_available else "degraded",
        "ollama_available": ollama_available,
        "ollama_url": OLLAMA_BASE_URL,
        "model": OLLAMA_MODEL,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/explain", response_model=ExplanationResponse)
def explain_prediction(request: ExplanationRequest):
    """
    Generate a human-readable explanation for a prediction result.
    
    This endpoint replaces the Java ExplanationService logic with LLM-powered generation.
    """
    
    try:
        logger.info(f"Received explanation request for prediction type: {request.predictionResult.predictionType}")
        
        # Generate explanation using Ollama Llama 2
        explanation, tokens_used = generate_explanation_with_llm(
            prediction_result=request.predictionResult,
            context=request.context,
            user_question=request.question
        )
        
        return ExplanationResponse(
            explanation=explanation,
            generated_at=datetime.utcnow().isoformat(),
            tokens_used=tokens_used
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing explanation request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate explanation: {str(e)}"
        )


@app.post("/explain/batch", response_model=List[ExplanationResponse])
def explain_predictions_batch(requests: List[ExplanationRequest]):
    """
    Batch endpoint for generating multiple explanations at once.
    Useful for dashboard views with multiple predictions.
    """
    
    try:
        logger.info(f"Received batch explanation request with {len(requests)} items")
        
        responses = []
        for request in requests:
            explanation, tokens_used = generate_explanation_with_llm(
                prediction_result=request.predictionResult,
                context=request.context,
                user_question=request.question
            )
            
            responses.append(ExplanationResponse(
                explanation=explanation,
                generated_at=datetime.utcnow().isoformat(),
                tokens_used=tokens_used
            ))
        
        return responses
        
    except Exception as e:
        logger.error(f"Error processing batch explanation request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate batch explanations: {str(e)}"
        )


# ============================================================================
# STARTUP CHECKS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Check Ollama availability and list available models on startup."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=None)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "unknown") for m in models]
            logger.info(f"Ollama is running. Available models: {model_names}")
            
            # Check if configured model is available
            if not any(OLLAMA_MODEL in name or name in OLLAMA_MODEL for name in model_names):
                logger.warning(f"⚠️  Configured model '{OLLAMA_MODEL}' not found!")
                logger.warning(f"   Available models: {model_names}")
                logger.warning(f"   Run: ollama pull {OLLAMA_MODEL.split(':')[0]}")
            else:
                logger.info(f"✓ Using model: {OLLAMA_MODEL}")
        else:
            logger.warning("Ollama responded but may have issues")
    except Exception as e:
        logger.error(f"Cannot connect to Ollama at {OLLAMA_BASE_URL}: {e}")
        logger.error("Make sure Ollama is running: ollama serve")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print(f"\n🚀 Starting LLM Explanation Service")
    print(f"   Ollama URL: {OLLAMA_BASE_URL}")
    print(f"   Model: {OLLAMA_MODEL}")
    print(f"   Port: 8001\n")
    # Port 8001 - Java backend runs on 8000, so LLM service uses 8001
    uvicorn.run(app, host="0.0.0.0", port=8001)