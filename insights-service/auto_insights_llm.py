import os
import re
import json
import logging
from typing import Any, Dict, List, Optional
import asyncio
from fastapi import HTTPException, status
from fastapi.responses import PlainTextResponse

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# --- config / env ---
APP_API_KEY = os.getenv("APP_API_KEY", "local-test-key")
PORT = int(os.getenv("PORT", "3000"))
MODEL_HOST = os.getenv("MODEL_HOST", "http://localhost:11434/api/generate")
MODEL_NAME = os.getenv("MODEL_NAME", "llama2:latest")
# httpx timeout in seconds
MODEL_TIMEOUT = float(os.getenv("MODEL_TIMEOUT", "1200"))  # 20 minutes

# --- logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("local-llm-api")

app = FastAPI(title="Local LLM Insights API")


# --- Pydantic request model (loose) ---
class TimePoint(BaseModel):
    period: Any
    value: float

class Detector(BaseModel):
    name: str
    facts: Dict[str, Any] = {}
    score: float = 0.0
    confidence: str = "LOW"
    explanation: str = ""

class FactsForLlm(BaseModel):
    meta: Dict[str, Any] = {}
    schema: Optional[List[Any]] = None
    series: List[TimePoint] = []
    detectors: List[Detector] = []
    provenance: Dict[str, Any] = {}

    class Config:
        extra = "allow"  # Allow extra fields

class MultiChartRequest(BaseModel):
    reportId: Optional[str] = None
    lastSql: Optional[str] = None
    factsForLlm: List[FactsForLlm] = []

    class Config:
        extra = "allow"

# --- helpers (mirrors JS helpers) ---


def validate_facts_light(facts: Dict[str, Any]) -> Optional[str]:
    if not facts or "detectors" not in facts:
        return "missing detectors array"
    if not isinstance(facts["detectors"], list):
        return "detectors must be an array"
    for d in facts["detectors"]:
        # pydantic will have parsed but still check minimally
        if not isinstance(d, dict) and not hasattr(d, "facts"):
            return f"detector malformed: {str(d)[:80]}"
        facts_map = d.get("facts") if isinstance(d, dict) else getattr(d, "facts", None)
        if facts_map is None:
            return f"detector malformed: missing facts in {str(d)[:80]}"
        if not isinstance(facts_map, dict):
            return f"detector facts must be object for {str(d)[:80]}"
        for k, v in facts_map.items():
            if v is not None and not isinstance(v, (int, float)):
                return f"detector fact {k} must be number or null"
    if "series" not in facts or not isinstance(facts.get("series", []), list):
        return "missing series array"
    return None


PII_KEYS = {"account_number", "pan", "email", "customer_name"}


def redact(obj: Any):
    if obj is None:
        return obj
    if isinstance(obj, list):
        return [redact(x) for x in obj]
    if not isinstance(obj, dict):
        return obj
    out: Dict[str, Any] = {}
    for k, v in obj.items():
        if k.lower() in PII_KEYS:
            out[k] = "[REDACTED]"
        else:
            out[k] = redact(v)
    return out


def build_prompt(safe_facts: Dict[str, Any]) -> str:
    """Build prompt that requests plain text insights instead of JSON — for a single chart"""
    return (
        "SYSTEM:\n"
        "You are a Business Insights Writer. Output ONLY the insights - no preamble, no conversation.\n\n"
        "USER:\n"
        "Data:\n"
        f"{json.dumps(safe_facts)}\n\n"
        "RULES:\n"
        "1. Write exactly ONE paragraph with 3-4 sentences\n"
        "2. Cover the key finding, its trend, and business impact in a flowing paragraph\n"
        "3. NO introductions like 'Here are...', 'Sure!', 'Based on the data...'\n"
        "4. NO conclusions like 'Overall...', 'In summary...', 'Let me know if...'\n"
        "5. Use ONLY business values: amounts, counts, percentages, dates, time periods\n"
        "6. NEVER mention: slope, confidence, score, seasonal_adjusted_slope, detector names, axis labels, JSON fields\n"
        "7. Describe trends plainly: 'increased', 'decreased', 'dropped by X%'\n"
        "8. NO headers, labels, bullet points, or references\n\n"
        "Output format (single paragraph, no extra text):\n"
        "The average transaction value has dropped to $100, marking a 15% decrease from last month. "
        "This downward trend has persisted over the past 6 weeks, suggesting it is not a temporary fluctuation. "
        "Such a sustained decline could indicate changes in customer spending behavior or increased market competition. "
        "Early intervention through targeted promotions may help reverse this trend.\n"
    )


def build_multi_chart_prompt(charts_data: List[Dict[str, Any]]) -> str:
    """Build prompt that requests plain text insights for multiple charts"""
    charts_json = json.dumps(charts_data, indent=2)
    num_charts = len(charts_data)
    return (
        "SYSTEM:\n"
        "You are a Business Insights Writer. Output ONLY the insights - no preamble, no conversation.\n\n"
        "USER:\n"
        f"I have {num_charts} charts on my dashboard. Here is the data for each chart:\n\n"
        f"{charts_json}\n\n"
        "RULES:\n"
        f"1. Write ONE paragraph of 3-4 sentences for EACH of the {num_charts} charts\n"
        "2. Separate each chart's paragraph with a blank line\n"
        f"3. Start each paragraph with 'Chart N:' where N is the chart number (1 to {num_charts}). "
        "If the chart has an xAxis/yAxis in its meta, mention what the chart is about\n"
        "4. Cover the key finding, its trend, and business impact in each paragraph\n"
        "5. NO introductions like 'Here are...', 'Sure!', 'Based on the data...'\n"
        "6. NO conclusions like 'Overall...', 'In summary...', 'Let me know if...'\n"
        "7. Use ONLY business values: amounts, counts, percentages, dates, time periods\n"
        "8. NEVER mention: slope, confidence, score, seasonal_adjusted_slope, detector names, axis labels, JSON fields\n"
        "9. Describe trends plainly: 'increased', 'decreased', 'dropped by X%'\n"
        "10. NO bullet points or numbered lists within each paragraph\n\n"
        f"Output exactly {num_charts} paragraphs, one per chart, separated by blank lines.\n"
    )



async def call_model_and_reconstruct(model_url: str, model_name: str, prompt: str, timeout: float = 60.0):
    """
    Call the model host and reconstruct streaming JSON-chunk responses.
    Returns: model_text (str) - the reconstructed textual output the model produced.
    Raises HTTPException on network/model error.
    """
    payload = {"model": model_name, "prompt": prompt, "max_tokens": 512, "temperature": 0.0}
    # Configure timeouts: quick connect, longer read (adjust as needed)
    timeout_cfg = httpx.Timeout(connect=5.0, read=timeout, write=10.0, pool=15.0)

    async with httpx.AsyncClient(timeout=timeout_cfg) as client:
        try:
            resp = await client.post(model_url, json=payload)
        except httpx.ConnectError as e:
            # connection-level failure (refused, no route)
            logger.exception("Model host connect error: %s %s", e.__class__.__name__, repr(e))
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                                detail=f"model_request_failed: ConnectError: {repr(e)}")
        except httpx.ReadTimeout as e:
            logger.exception("Model host read timeout: %s %s", e.__class__.__name__, repr(e))
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                                detail=f"model_request_failed: ReadTimeout: {repr(e)}")
        except httpx.RequestError as e:
            # generic httpx request error (DNS, TLS, etc.)
            logger.exception("Model host request error: %s %s", e.__class__.__name__, repr(e))
            # include exception class and args (repr) so the detail isn't empty
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                                detail=f"model_request_failed: {e.__class__.__name__}: {repr(e)}")


    # If provider returned an error status, return 502 with details
    if resp.status_code >= 400:
        text = (await resp.aread()).decode(errors="replace") if hasattr(resp, "aread") else resp.text
        logger.error("Model host returned HTTP %s: %.1000s", resp.status_code, text[:1000])
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"model_error: {resp.status_code} {text[:2000]}")

    model_text = ""

    # 1) Try the easy path: server returned a single valid JSON body
    try:
        j = resp.json()
        # common shapes: choices.message.content (OpenAI-like)
        if isinstance(j, dict) and "choices" in j and isinstance(j["choices"], list):
            first = j["choices"][0]
            if isinstance(first, dict) and "message" in first and isinstance(first["message"], dict):
                model_text = first["message"].get("content", "") or ""
            else:
                # fallback: join choice.text or message contents
                parts = []
                for c in j["choices"]:
                    if isinstance(c, dict):
                        if "text" in c:
                            parts.append(str(c.get("text","")))
                        elif "message" in c and isinstance(c["message"], dict):
                            parts.append(str(c["message"].get("content","")))
                model_text = "".join(parts)
        elif isinstance(j, dict) and "output" in j and isinstance(j["output"], list):
            model_text = "".join([str(o.get("content","")) if isinstance(o, dict) else str(o) for o in j["output"]])
        elif isinstance(j, dict) and "text" in j:
            model_text = j["text"]
        elif isinstance(j, str):
            model_text = j
        else:
            # if arbitrary JSON, stringify it
            model_text = json.dumps(j)
    except Exception:
        # 2) Fallback: the response is not a single JSON — likely NDJSON streaming (one JSON object per line)
        raw_text = resp.text  # raw newline-delimited JSON lines
        model_text_parts = []

        # Parse each line; extract common token fields ("response","text","output","content")
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # not a JSON line — append raw line (defensive)
                model_text_parts.append(line)
                continue

            # If obj has nested 'response' field tokens (Ollama-like streaming)
            if isinstance(obj, dict):
                # common token field names to try in order:
                for field in ("response", "text", "output", "content"):
                    if field in obj:
                        val = obj.get(field)
                        # "output" might be an array of objects
                        if isinstance(val, list):
                            for v in val:
                                if isinstance(v, dict):
                                    model_text_parts.append(str(v.get("content","")))
                                else:
                                    model_text_parts.append(str(v))
                        elif val is not None:
                            model_text_parts.append(str(val))
                        break
                else:
                    # If no known fields, but there is an 'message' object with 'content'
                    if "message" in obj and isinstance(obj["message"], dict):
                        model_text_parts.append(str(obj["message"].get("content","")))
                    # otherwise ignore or append nothing
            else:
                # not a dict — append string representation
                model_text_parts.append(str(obj))

        model_text = "".join(model_text_parts)

    # final defensive trimming (avoid huge logs)
    model_text_trimmed = model_text[:20000]  # keep only first 20k chars for logging
    logger.info("Reconstructed model text (truncated): %s", model_text_trimmed.replace("\n","\\n"))

    if not model_text or model_text.strip() == "":
        # nothing reconstructed
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail={"error":"model_bad_output","raw":resp.text[:20000]})

    return model_text


# --- Endpoint ---


def clean_model_output(model_text: str) -> str:
    """Clean up model output, removing JSON artifacts if present"""
    cleaned_text = model_text.strip()
    
    # Remove any JSON-like artifacts if the model still outputs them
    if cleaned_text.startswith('[') or cleaned_text.startswith('{'):
        try:
            parsed = json.loads(cleaned_text)
            if isinstance(parsed, list):
                paragraphs = []
                for item in parsed:
                    if isinstance(item, dict):
                        headline = item.get('headline', '')
                        why = item.get('why', '')
                        if headline and why:
                            paragraphs.append(f"{headline} {why}")
                        elif headline:
                            paragraphs.append(headline)
                        elif why:
                            paragraphs.append(why)
                cleaned_text = "\n\n".join(paragraphs)
        except:
            pass  # If parsing fails, keep original text
    
    return cleaned_text


@app.post("/api/llm/insights")
async def insights_endpoint(request: Request, x_api_key: Optional[str] = Header(None)):
    # 1) auth
    key = x_api_key or request.query_params.get("api_key")
    if not key or key != APP_API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="unauthorized")

    # 2) parse JSON body
    body = await request.json()

    # 3) Detect if this is a multi-chart request or single-chart request
    is_multi_chart = isinstance(body, dict) and "factsForLlm" in body and isinstance(body["factsForLlm"], list)

    if is_multi_chart:
        # --- MULTI-CHART PATH ---
        logger.info("📊 Multi-chart request detected with %d charts", len(body["factsForLlm"]))
        
        # Validate via Pydantic
        try:
            multi_req = MultiChartRequest.parse_obj(body)
        except Exception as e:
            logger.warning("Multi-chart payload validation failed: %s", str(e))
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                              detail={"error": "invalid_payload", "reason": str(e)})

        charts_facts = body.get("factsForLlm", [])
        
        if len(charts_facts) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                              detail={"error": "no_charts", "reason": "factsForLlm array is empty"})

        # If only one chart, use the single-chart path for backward compatibility
        if len(charts_facts) == 1:
            safe_facts = redact(charts_facts[0])
            prompt = build_prompt(safe_facts)
        else:
            # Multiple charts: sanitize each chart's facts and build multi-chart prompt
            safe_charts = [redact(chart) for chart in charts_facts]
            prompt = build_multi_chart_prompt(safe_charts)

    else:
        # --- SINGLE-CHART PATH (backward compatible) ---
        logger.info("📊 Single-chart request detected")
        try:
            FactsForLlm.parse_obj(body)
        except Exception as e:
            logger.warning("Payload validation failed: %s", str(e))
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, 
                              detail={"error": "invalid_payload", "reason": str(e)})

        safe_facts = redact(body)
        prompt = build_prompt(safe_facts)

    # 4) call model
    try:
        model_text = await call_model_and_reconstruct(
            MODEL_HOST, MODEL_NAME, prompt, timeout=float(MODEL_TIMEOUT)
        )
    except HTTPException as e:
        logger.error("Model call failed: %s", e.detail if hasattr(e, "detail") else str(e))
        raise e
    
    # 5) Clean up the model output
    cleaned_text = clean_model_output(model_text)
    
    logger.info("=" * 80)
    logger.info("📤 RETURNING PLAIN TEXT INSIGHTS")
    logger.info(f"Length: {len(cleaned_text)} characters")
    if is_multi_chart:
        logger.info(f"Charts processed: {len(body.get('factsForLlm', []))}")
    logger.info("=" * 80)
    
    # Return as plain text with correct content type
    return PlainTextResponse(content=cleaned_text, status_code=200)


# run with `uvicorn auto_insights_llm:app --reload --port 3000`
