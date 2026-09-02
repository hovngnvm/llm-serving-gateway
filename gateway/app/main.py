"""
FastAPI Security & Optimization Gateway.
Enterprise AI Platform Entrypoint & Interactive Web Playground.
"""

import time
import uuid
from typing import Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Security, Depends, Request, BackgroundTasks
from fastapi.security.api_key import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import httpx
from prometheus_fastapi_instrumentator import Instrumentator

from gateway.app.config import STATIC_DIR, settings, ensure_directories
from gateway.app.utils.logger import get_logger
from gateway.app.core.presidio_engine import presidio_engine
from gateway.app.core.semantic_cache import semantic_cache
from gateway.app.core.output_validator import output_validator, FinancialTransactionSchema
from gateway.app.core.guardrails_engine import guardrails_engine
from gateway.app.core.intent_router import intent_router
from gateway.app.db.neon_audit_logger import neon_audit_logger

logger = get_logger(__name__)

MAX_OUTPUT_TOKENS_CEILING = 512
DEFAULT_FALLBACK_IP = "127.0.0.1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_directories()
    app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=3.0, read=60.0, write=10.0, pool=10.0),
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
    )
    logger.info("AI Serving & Security Gateway initialized with connection pooling.")
    try:
        await neon_audit_logger.init_db()
    except Exception as e:
        logger.warning(f"Neon DB initialization notice: {e}")
    yield
    if hasattr(app.state, "http_client") and app.state.http_client:
        await app.state.http_client.aclose()


app = FastAPI(
    title="AI Serving & Security Gateway",
    description="High-Performance LLM Serving, Semantic Caching & Privacy Guardrails.",
    version="1.0.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app, endpoint=settings.prometheus_metrics_path)

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


@app.get("/v1/models")
async def list_available_models() -> dict[str, Any]:
    """Returns dynamic model inventory and active serving strategies."""
    return {
        "object": "list",
        "data": intent_router.get_registered_models()
    }


@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(
    request_data: ChatCompletionRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    start_time = time.time()
    request_id = f"req-{uuid.uuid4().hex[:12]}"
    client_ip = http_request.client.host if http_request.client else DEFAULT_FALLBACK_IP

    raw_prompt = ""
    raw_image_b64 = ""
    masked_messages = []
    pii_mapping: dict[str, str] = {}
    total_pii_count = 0

    for msg in request_data.messages:
        role = msg.role
        content = msg.content
        if isinstance(content, str):
            masked_text, m_map, cnt = presidio_engine.mask_text(content)
            pii_mapping.update(m_map)
            total_pii_count += cnt
            if role == "user":
                raw_prompt += content + "\n"
            masked_messages.append({"role": role, "content": masked_text})
        elif isinstance(content, list):
            new_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_val = part.get("text", "")
                    masked_text, m_map, cnt = presidio_engine.mask_text(text_val)
                    pii_mapping.update(m_map)
                    total_pii_count += cnt
                    raw_prompt += text_val + "\n"
                    new_parts.append({"type": "text", "text": masked_text})
                elif isinstance(part, dict) and part.get("type") == "image_url":
                    new_parts.append(part)
                    img_url = part.get("image_url", {}).get("url", "")
                    if "base64" in img_url:
                        raw_image_b64 = img_url
                else:
                    new_parts.append(part)
            masked_messages.append({"role": role, "content": new_parts})
        else:
            masked_messages.append({"role": role, "content": str(content)})

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
        background_tasks.add_task(
            neon_audit_logger.log_request,
            request_id=request_id,
            client_ip=client_ip,
            masked_prompt=raw_prompt,
            model_id=resolved_model,
            response_formats=cached_payload.get("formats", {}),
            pii_count=total_pii_count,
            cached_hit=True,
            execution_time_ms=execution_time_ms,
            status_code=200,
        )
        cached_payload["meta"]["cached_hit"] = True
        cached_payload["meta"]["execution_time_ms"] = round(execution_time_ms, 2)
        cached_payload["meta"]["model_id"] = resolved_model
        return cached_payload

    redacted_image_preview = ""
    if raw_image_b64:
        redacted_image_preview, ocr_masked_text, ocr_mapping, ocr_pii_count = presidio_engine.process_multimodal_ocr(raw_image_b64)
        pii_mapping.update(ocr_mapping)
        total_pii_count += ocr_pii_count

        if ocr_masked_text:
            ocr_block = f"\n\n--- OCR EXTRACTED TEXT (SANITIZED) ---\n{ocr_masked_text}"
            if masked_messages and masked_messages[-1]["role"] == "user":
                last_content = masked_messages[-1]["content"]
                if isinstance(last_content, str):
                    masked_messages[-1]["content"] += ocr_block
                elif isinstance(last_content, list):
                    last_content.append({"type": "text", "text": ocr_block})
            else:
                masked_messages.append({"role": "user", "content": f"Extract invoice:{ocr_block}"})

    raw_llm_response = ""
    max_tokens_clamped = min(request_data.max_tokens or MAX_OUTPUT_TOKENS_CEILING, MAX_OUTPUT_TOKENS_CEILING)
    client = getattr(http_request.app.state, "http_client", None) or httpx.AsyncClient(
        timeout=httpx.Timeout(connect=3.0, read=60.0, write=10.0, pool=10.0)
    )

    try:
        vllm_payload = {
            "model": resolved_model or settings.vllm_model_name,
            "messages": masked_messages,
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

    is_output_safe, output_violation = guardrails_engine.validate_output(raw_llm_response)
    if not is_output_safe:
        raise HTTPException(status_code=502, detail=output_violation)

    target_schema = FinancialTransactionSchema if (resolved_model == "financial_adapter" or "financial" in str(resolved_model).lower()) else None
    parsed_json, is_valid, validation_errors, was_repaired = output_validator.parse_and_validate(
        raw_llm_response,
        schema_class=target_schema,
    )

    formats: dict[str, Any] = {}
    if parsed_json and isinstance(parsed_json, dict) and not parsed_json.get("error"):
        formats["structured_data"] = parsed_json

    if redacted_image_preview:
        formats["redacted_image_base64"] = redacted_image_preview

    unmasked_text = presidio_engine.unmask_text(raw_llm_response, pii_mapping)
    formats["text_summary"] = unmasked_text

    execution_time_ms = (time.time() - start_time) * 1000

    response_payload = {
        "request_id": request_id,
        "status": "success",
        "meta": {
            "execution_time_ms": round(execution_time_ms, 2),
            "pii_redacted_count": total_pii_count,
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
        masked_prompt=raw_prompt,
        model_id=resolved_model,
        response_formats=formats,
        pii_count=total_pii_count,
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
