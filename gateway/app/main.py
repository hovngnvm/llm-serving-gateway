"""
FastAPI Security & Optimization Gateway.
Enterprise AI Platform Entrypoint & Interactive Web Playground.
"""

import time
import uuid
from pathlib import Path
from typing import Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Security, Depends, Request, BackgroundTasks
from fastapi.security.api_key import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import httpx
from prometheus_fastapi_instrumentator import Instrumentator

from gateway.app.config import settings
from gateway.app.utils.logger import get_logger
from gateway.app.core.presidio_engine import presidio_engine
from gateway.app.core.semantic_cache import semantic_cache
from gateway.app.core.output_validator import output_validator, FinancialTransactionSchema
from gateway.app.core.guardrails_engine import guardrails_engine
from gateway.app.core.intent_router import intent_router
from gateway.app.db.neon_audit_logger import neon_audit_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AI Serving & Security Gateway initialized successfully.")
    try:
        await neon_audit_logger.init_db()
        logger.info("Neon Serverless PostgreSQL Cloud connection verified.")
    except Exception as e:
        logger.warning(f"Neon DB initialization notice: {e}")
    yield


app = FastAPI(
    title="AI Serving & Security Gateway",
    description="High-Performance LLM Serving, Semantic Caching & Privacy Guardrails.",
    version="1.0.0",
    lifespan=lifespan,
)

# Prometheus Telemetry Instrumentation (Mandatory Core Service)
Instrumentator().instrument(app).expose(app, endpoint=settings.prometheus_metrics_path)

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False, response_model=None)
@app.get("/playground", include_in_schema=False, response_model=None)
async def serve_playground():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "AI Serving & Security Gateway Live. Open /docs for Swagger UI."}


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    if not api_key or api_key != settings.gateway_api_key:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid or missing X-API-Key in request header.",
        )
    return api_key


class MessageItem(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str | None = Field(default="auto")
    messages: list[MessageItem]
    temperature: float | None = 0.2
    max_tokens: int | None = 512


@app.get("/health")
async def health_check() -> dict[str, Any]:
    return {
        "status": "healthy",
        "gateway_version": "1.0.0",
        "model": settings.vllm_model_name,
        "semantic_cache": {
            "hits": semantic_cache.hit_count,
            "misses": semantic_cache.miss_count,
        },
        "schema_validations": output_validator.validation_count,
        "json_auto_repairs": output_validator.auto_repair_count,
    }


class RegisterAdapterRequest(BaseModel):
    adapter_name: str
    description: str
    keywords: list[str]
    regex_patterns: list[str] | None = None
    strategy: str | None = "Strategy 2 (Dynamic Multi-LoRA)"
    priority: int | None = 10


@app.get("/v1/models")
async def list_available_models() -> dict[str, Any]:
    """Returns dynamic model inventory and active serving strategies."""
    return {
        "object": "list",
        "data": intent_router.get_registered_models()
    }


@app.post("/v1/models/register", dependencies=[Depends(verify_api_key)])
async def register_new_adapter(req: RegisterAdapterRequest) -> dict[str, Any]:
    """Dynamically registers a newly exported LoRA adapter at runtime (Zero-Touch)."""
    intent_router.register_adapter(
        target_model=req.adapter_name,
        description=req.description,
        keywords=req.keywords,
        regex_patterns=req.regex_patterns,
        strategy=req.strategy or "Strategy 2 (Dynamic Multi-LoRA)",
        priority=req.priority or 10,
    )
    return {
        "status": "success",
        "message": f"Adapter '{req.adapter_name}' successfully registered into Intent Router.",
        "registered_adapters_count": len(intent_router.rules),
    }


@app.post("/v1/models/refresh", dependencies=[Depends(verify_api_key)])
async def refresh_adapters_from_artifacts() -> dict[str, Any]:
    """Auto-scans artifacts/runs/ for new manifest.json files and loads new adapters."""
    count = intent_router.auto_discover_from_artifacts()
    return {
        "status": "success",
        "message": f"Auto-discovery scanned artifacts/runs/. Discovered and loaded {count} new adapter manifests.",
        "total_active_adapters": len(intent_router.rules),
    }


@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(
    request_data: ChatCompletionRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    start_time = time.time()
    request_id = f"req-{uuid.uuid4().hex[:12]}"
    client_ip = http_request.client.host if http_request.client else "127.0.0.1"

    raw_prompt = ""
    raw_image_b64 = ""
    for msg in request_data.messages:
        if isinstance(msg.content, str):
            raw_prompt += msg.content + "\n"
        elif isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        raw_prompt += part.get("text", "") + "\n"
                    elif part.get("type") == "image_url":
                        img_url = part.get("image_url", {}).get("url", "")
                        if "base64" in img_url:
                            raw_image_b64 = img_url

    raw_prompt = raw_prompt.strip()

    is_safe, violation_msg = guardrails_engine.validate_input(raw_prompt)
    if not is_safe:
        raise HTTPException(status_code=400, detail=violation_msg)

    route_decision = intent_router.resolve_model(
        prompt=raw_prompt,
        requested_model=request_data.model,
    )
    resolved_model = route_decision["target_model"]
    logger.info(f"Route Decision: {route_decision['reason']}")

    cached_payload = await semantic_cache.get(raw_prompt)
    if cached_payload and not raw_image_b64:
        execution_time_ms = (time.time() - start_time) * 1000
        masked_prompt_cache, _, pii_count_cache = presidio_engine.mask_text(raw_prompt)
        background_tasks.add_task(
            neon_audit_logger.log_request,
            request_id=request_id,
            client_ip=client_ip,
            masked_prompt=masked_prompt_cache,
            model_id=resolved_model,
            response_formats=cached_payload.get("formats", {}),
            pii_count=pii_count_cache,
            cached_hit=True,
            execution_time_ms=execution_time_ms,
            status_code=200,
        )
        cached_payload["meta"]["cached_hit"] = True
        cached_payload["meta"]["execution_time_ms"] = round(execution_time_ms, 2)
        cached_payload["meta"]["model_id"] = resolved_model
        return cached_payload

    masked_prompt, pii_mapping, pii_count = presidio_engine.mask_text(raw_prompt)
    redacted_image_preview = ""
    if raw_image_b64:
        redacted_image_preview, ocr_masked_text, ocr_mapping, ocr_pii_count = presidio_engine.process_multimodal_ocr(raw_image_b64)
        pii_mapping.update(ocr_mapping)
        pii_count += ocr_pii_count

        if ocr_masked_text:
            if masked_prompt:
                masked_prompt = f"{masked_prompt}\n\n--- OCR EXTRACTED TEXT (SANITIZED) ---\n{ocr_masked_text}"
            else:
                masked_prompt = f"Extract information from the following invoice:\n\n--- OCR EXTRACTED TEXT (SANITIZED) ---\n{ocr_masked_text}"

    raw_llm_response = ""
    max_tokens_clamped = min(request_data.max_tokens or 512, 512)
    try:
        timeout_cfg = httpx.Timeout(connect=3.0, read=60.0, write=10.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout_cfg) as client:
            vllm_payload = {
                "model": resolved_model or settings.vllm_model_name,
                "messages": [{"role": "user", "content": masked_prompt}],
                "temperature": request_data.temperature,
                "max_tokens": max_tokens_clamped,
            }
            resp = await client.post(
                f"{settings.vllm_base_url}/chat/completions",
                json=vllm_payload,
            )
            if resp.status_code == 200:
                vllm_data = resp.json()
                raw_llm_response = vllm_data["choices"][0]["message"]["content"]
            else:
                logger.error(f"vLLM returned HTTP {resp.status_code}: {resp.text}")
                raise Exception(f"vLLM status {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.error(f"vLLM inference serving error: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"AI Inference Engine is currently initializing or unreachable ({type(e).__name__}). Please ensure vLLM is loaded.",
        )

    target_schema = FinancialTransactionSchema if (resolved_model == "financial_adapter" or "financial" in str(resolved_model).lower()) else None
    parsed_json, is_valid, validation_errors, was_repaired = output_validator.parse_and_validate(
        raw_llm_response,
        schema_class=target_schema,
    )

    formats: dict[str, Any] = {}

    if parsed_json and isinstance(parsed_json, dict) and not parsed_json.get("error"):
        formats["structured_data"] = parsed_json

    markdown_indicators = ["```", "#", "|", "**", "- ", "\n1.", "\n* ", "__"]
    if not formats.get("structured_data") and any(ind in raw_llm_response for ind in markdown_indicators):
        formats["markdown_report"] = presidio_engine.unmask_text(raw_llm_response, pii_mapping)

    if parsed_json and isinstance(parsed_json, dict) and parsed_json.get("message"):
        formats["text_summary"] = parsed_json.get("message")
    else:
        formats["text_summary"] = raw_llm_response

    if redacted_image_preview:
        formats["redacted_image_base64"] = redacted_image_preview

    unmasked_summary = presidio_engine.unmask_text(formats["text_summary"], pii_mapping)
    formats["text_summary"] = unmasked_summary

    execution_time_ms = (time.time() - start_time) * 1000

    response_payload = {
        "request_id": request_id,
        "status": "success",
        "meta": {
            "execution_time_ms": round(execution_time_ms, 2),
            "pii_redacted_count": pii_count,
            "cached_hit": False,
            "json_auto_repaired": was_repaired,
            "schema_validated": is_valid,
            "model_id": resolved_model,
        },
        "formats": formats,
    }

    background_tasks.add_task(
        neon_audit_logger.log_request,
        request_id=request_id,
        client_ip=client_ip,
        masked_prompt=masked_prompt,
        model_id=resolved_model,
        response_formats=formats,
        pii_count=pii_count,
        cached_hit=False,
        execution_time_ms=execution_time_ms,
        status_code=200,
    )

    if not raw_image_b64:
        background_tasks.add_task(semantic_cache.set, prompt=raw_prompt, payload=response_payload)

    return response_payload


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("gateway.app.main:app", host=settings.gateway_host, port=settings.gateway_port, reload=True)
