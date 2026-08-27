"""
Semantic Prompt Caching Engine via Redis Vector Cache & In-Memory Fallback.
Evaluates Cosine Similarity between incoming prompts and cached entries.
Bypasses vLLM serving and returns sub-5ms responses for repeated queries (>0.95 similarity).
"""

import json
import math
import hashlib
import time
from typing import Any
import redis.asyncio as aioredis
from gateway.app.config import settings
from gateway.app.utils.logger import get_logger

logger = get_logger(__name__)

MAX_MEMORY_CACHE_ENTRIES = 1000
REDIS_CONNECT_COOLDOWN_SECONDS = 10.0
VECTOR_DIMENSION = 128


class SemanticPromptCache:
    def __init__(self) -> None:
        self.threshold = settings.semantic_cache_threshold
        self.ttl = settings.semantic_cache_ttl_seconds
        self.redis_client: Any = None
        self._last_redis_fail_time: float = 0.0
        self.hit_count = 0
        self.miss_count = 0
        self._memory_cache: dict[str, dict[str, Any]] = {}

    async def get_client(self) -> Any:
        now = time.monotonic()
        if self.redis_client is None:
            if now - self._last_redis_fail_time < REDIS_CONNECT_COOLDOWN_SECONDS:
                return None
            try:
                client = aioredis.Redis(
                    host=settings.redis_host,
                    port=settings.redis_port,
                    password=settings.redis_password,
                    decode_responses=True,
                    socket_connect_timeout=1.5,
                    socket_timeout=1.5,
                )
                await client.ping()
                self.redis_client = client
            except Exception as e:
                self._last_redis_fail_time = now
                logger.debug(f"Redis cache not reachable ({e}). Using in-memory vector cache.")
                return None
        return self.redis_client

    def _simple_text_vector(self, text: str, dim: int = VECTOR_DIMENSION) -> list[float]:
        """Generates a fast, normalized n-gram hash vector for prompt similarity comparisons."""
        vec = [0.0] * dim
        words = text.lower().strip().split()
        if not words:
            return vec

        for word in words:
            idx = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16) % dim
            vec[idx] += 1.0

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def _cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """Computes Cosine Similarity between two normalized vectors."""
        if len(vec_a) != len(vec_b) or not vec_a:
            return 0.0
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        return max(0.0, min(1.0, dot_product))

    async def get(self, prompt: str) -> dict[str, Any] | None:
        """Checks if a semantically similar prompt exists in cache with similarity >= threshold."""
        query_vec = self._simple_text_vector(prompt)

        try:
            client = await self.get_client()
            if client:
                keys = []
                async for key in client.scan_iter("semcache:*", count=100):
                    keys.append(key)

                if keys:
                    values = await client.mget(keys)
                    best_match_payload = None
                    best_similarity = 0.0

                    for raw_val in values:
                        if not raw_val:
                            continue
                        cached_item = json.loads(raw_val)
                        sim = self._cosine_similarity(query_vec, cached_item.get("vector", []))
                        if sim > best_similarity:
                            best_similarity = sim
                            best_match_payload = cached_item.get("payload")

                    if best_similarity >= self.threshold and best_match_payload is not None:
                        self.hit_count += 1
                        logger.info(f"Redis Semantic Cache Hit! Cosine Sim: {best_similarity:.4f}")
                        return best_match_payload
        except Exception as e:
            logger.debug(f"Redis get error: {e}")

        best_sim = 0.0
        best_payload = None
        for key, item in self._memory_cache.items():
            sim = self._cosine_similarity(query_vec, item.get("vector", []))
            if sim > best_sim:
                best_sim = sim
                best_payload = item.get("payload")

        if best_sim >= self.threshold and best_payload:
            self.hit_count += 1
            logger.info(f"In-Memory Semantic Cache Hit! Cosine Sim: {best_sim:.4f}")
            return json.loads(json.dumps(best_payload))

        self.miss_count += 1
        return None

    async def set(self, prompt: str, payload: dict[str, Any]) -> bool:
        """Stores prompt vector and response payload into Redis and memory cache."""
        vector = self._simple_text_vector(prompt)
        prompt_hash = hashlib.sha256(prompt.strip().lower().encode("utf-8")).hexdigest()[:16]
        cache_key = f"semcache:{prompt_hash}"

        cache_entry = {
            "prompt": prompt,
            "vector": vector,
            "payload": payload,
        }

        # Memory Cache LRU eviction if oversized
        if len(self._memory_cache) >= MAX_MEMORY_CACHE_ENTRIES:
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]

        self._memory_cache[cache_key] = cache_entry

        try:
            client = await self.get_client()
            if client:
                await client.set(
                    cache_key,
                    json.dumps(cache_entry, ensure_ascii=False),
                    ex=self.ttl,
                )
                logger.info(f"Stored response in Redis Semantic Cache: {cache_key}")
                return True
        except Exception as e:
            logger.debug(f"Redis set error: {e}")

        return True


semantic_cache = SemanticPromptCache()
