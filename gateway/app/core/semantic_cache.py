"""
Semantic Prompt Caching Engine via Redis Vector Cache & In-Memory Fallback.
Evaluates Cosine Similarity between incoming prompts and cached entries.
Bypasses vLLM serving and returns sub-5ms responses for repeated queries (>0.95 similarity).
"""

import json
import math
import hashlib
from typing import Any
from gateway.app.config import settings
from gateway.app.utils.logger import get_logger

logger = get_logger("SemanticPromptCache")


class SemanticPromptCache:
    def __init__(self) -> None:
        self.threshold = settings.semantic_cache_threshold
        self.ttl = settings.semantic_cache_ttl_seconds
        self.redis_client: Any = None
        self.hit_count = 0
        self.miss_count = 0
        self._memory_cache: dict[str, dict[str, Any]] = {}

    async def get_client(self) -> Any:
        if self.redis_client is None:
            try:
                import redis.asyncio as aioredis
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
                logger.debug(f"Redis cache not reachable ({e}). Using in-memory vector cache.")
                return None
        return self.redis_client

    def _simple_text_vector(self, text: str, dim: int = 128) -> list[float]:
        """Generates a fast, normalized n-gram hash vector for prompt similarity comparisons."""
        vec = [0.0] * dim
        words = text.lower().strip().split()
        if not words:
            return vec

        for word in words:
            idx = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % dim
            vec[idx] += 1.0

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def _cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """Computes Cosine Similarity between two normalized vectors."""
        if len(vec_a) != len(vec_b):
            return 0.0
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        return max(0.0, min(1.0, dot_product))

    async def get(self, prompt: str) -> dict[str, Any] | None:
        """Checks if a semantically similar prompt exists in cache with similarity >= threshold."""
        query_vec = self._simple_text_vector(prompt)

        try:
            client = await self.get_client()
            if client:
                keys = await client.keys("semcache:*")
                best_match_key = None
                best_similarity = 0.0

                for key in keys:
                    cached_raw = await client.get(key)
                    if not cached_raw:
                        continue
                    cached_item = json.loads(cached_raw)
                    cached_vec = cached_item.get("vector", [])

                    sim = self._cosine_similarity(query_vec, cached_vec)
                    if sim > best_similarity:
                        best_similarity = sim
                        best_match_key = key

                if best_similarity >= self.threshold and best_match_key:
                    cached_raw = await client.get(best_match_key)
                    if cached_raw:
                        cached_item = json.loads(cached_raw)
                        self.hit_count += 1
                        logger.info(f"Redis Semantic Cache Hit! Cosine Sim: {best_similarity:.4f}")
                        return cached_item.get("payload")
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
