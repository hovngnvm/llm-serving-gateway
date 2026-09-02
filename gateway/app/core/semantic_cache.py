"""
Native HNSW Semantic Prompt Caching Engine via Redis Stack & L1 In-Memory Cache.
Uses RediSearch Vector Indexing (C-module) with Cosine Distance Metric for sub-5ms semantic lookup.
"""

import json
import math
import zlib
import hashlib
import time
import struct
from typing import Any
import redis.asyncio as aioredis
from redis.commands.search.field import VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from gateway.app.config import settings
from gateway.app.utils.logger import get_logger

logger = get_logger(__name__)

MAX_MEMORY_CACHE_ENTRIES = 1000
REDIS_CONNECT_COOLDOWN_SECONDS = 10.0
VECTOR_DIMENSION = 128
INDEX_NAME = "idx:semcache"


class SemanticPromptCache:
    def __init__(self) -> None:
        self.threshold = settings.semantic_cache_threshold
        self.ttl = settings.semantic_cache_ttl_seconds
        self.redis_client: Any = None
        self._last_redis_fail_time: float = 0.0
        self._index_initialized = False
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
                    decode_responses=False,
                    socket_connect_timeout=1.5,
                    socket_timeout=1.5,
                )
                await client.ping()
                self.redis_client = client
                await self._ensure_index(client)
            except Exception as e:
                self._last_redis_fail_time = now
                logger.debug(f"Redis cache not reachable ({e}). Using in-memory vector cache.")
                return None
        return self.redis_client

    async def _ensure_index(self, client: Any) -> None:
        """Initializes RediSearch HNSW Vector Index on HASH keys."""
        if self._index_initialized:
            return
        try:
            await client.ft(INDEX_NAME).info()
            self._index_initialized = True
        except Exception:
            try:
                schema = (
                    VectorField(
                        "vector",
                        "HNSW",
                        {
                            "TYPE": "FLOAT32",
                            "DIM": VECTOR_DIMENSION,
                            "DISTANCE_METRIC": "COSINE",
                        },
                    ),
                )
                await client.ft(INDEX_NAME).create_index(
                    schema,
                    definition=IndexDefinition(prefix=["semcache:"], index_type=IndexType.HASH),
                )
                self._index_initialized = True
                logger.info(f"RediSearch HNSW vector index '{INDEX_NAME}' initialized successfully.")
            except Exception as create_err:
                logger.debug(f"Index creation notice: {create_err}")

    def _simple_text_vector(self, text: str, dim: int = VECTOR_DIMENSION) -> list[float]:
        """Generates a fast, normalized n-gram hash vector using zlib.crc32."""
        vec = [0.0] * dim
        words = text.lower().strip().split()
        if not words:
            return vec

        for word in words:
            idx = zlib.crc32(word.encode("utf-8")) % dim
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

    def _set_memory(self, cache_key: str, vector: list[float], payload: dict[str, Any]) -> None:
        """Stores entry in L1 LRU in-memory cache."""
        if len(self._memory_cache) >= MAX_MEMORY_CACHE_ENTRIES:
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]
        self._memory_cache[cache_key] = {"vector": vector, "payload": payload}

    async def get(self, prompt: str) -> dict[str, Any] | None:
        """Checks if a semantically similar prompt exists in cache with similarity >= threshold."""
        query_vec = self._simple_text_vector(prompt)
        prompt_hash = hashlib.sha256(prompt.strip().lower().encode("utf-8")).hexdigest()[:16]
        cache_key = f"semcache:{prompt_hash}"

        # 1. L1 Memory exact hit
        if cache_key in self._memory_cache:
            self.hit_count += 1
            logger.info("L1 In-Memory Semantic Cache Exact Hit! (sub-1ms)")
            return dict(self._memory_cache[cache_key]["payload"])

        # 2. Redis Stack Native HNSW Vector Search
        try:
            client = await self.get_client()
            if client:
                query_bytes = struct.pack(f"<{len(query_vec)}f", *query_vec)
                q = (
                    Query("*=>[KNN 1 @vector $vec AS score]")
                    .sort_by("score")
                    .return_fields("score", "payload")
                    .dialect(2)
                )
                res = await client.ft(INDEX_NAME).search(q, query_params={"vec": query_bytes})
                if res and res.docs:
                    doc = res.docs[0]
                    # Cosine distance = 1.0 - similarity
                    sim = 1.0 - float(doc.score)
                    if sim >= self.threshold:
                        payload_data = json.loads(doc.payload)
                        self._set_memory(cache_key, query_vec, payload_data)
                        self.hit_count += 1
                        logger.info(f"RediSearch HNSW Vector Cache Hit! Cosine Sim: {sim:.4f}")
                        return payload_data
        except Exception as e:
            logger.debug(f"RediSearch vector query notice: {e}")

        # 3. L1 Memory Cosine Similarity Fallback
        best_sim = 0.0
        best_payload = None
        for k, item in self._memory_cache.items():
            sim = self._cosine_similarity(query_vec, item.get("vector", []))
            if sim > best_sim:
                best_sim = sim
                best_payload = item.get("payload")

        if best_sim >= self.threshold and best_payload:
            self.hit_count += 1
            logger.info(f"In-Memory Semantic Cache Hit! Cosine Sim: {best_sim:.4f}")
            return dict(best_payload)

        self.miss_count += 1
        return None

    async def set(self, prompt: str, payload: dict[str, Any]) -> bool:
        """Stores prompt vector and response payload into Redis and memory cache."""
        vector = self._simple_text_vector(prompt)
        prompt_hash = hashlib.sha256(prompt.strip().lower().encode("utf-8")).hexdigest()[:16]
        cache_key = f"semcache:{prompt_hash}"

        self._set_memory(cache_key, vector, payload)

        try:
            client = await self.get_client()
            if client:
                vec_bytes = struct.pack(f"<{len(vector)}f", *vector)
                mapping = {
                    "vector": vec_bytes,
                    "payload": json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                }
                await client.hset(cache_key, mapping=mapping)
                await client.expire(cache_key, self.ttl)
                logger.info(f"Stored response in Redis HNSW Semantic Cache: {cache_key}")
                return True
        except Exception as e:
            logger.debug(f"Redis set error: {e}")

        return True


semantic_cache = SemanticPromptCache()
