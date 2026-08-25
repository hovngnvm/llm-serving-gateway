"""
Neon Serverless PostgreSQL Cloud Audit Logger (SSL/TLS).
Asynchronously logs sanitized, PII-masked compliance records for PCI-DSS & SBV compliance.
"""

import json
from typing import Any
from gateway.app.config import settings
from gateway.app.utils.logger import get_logger

logger = get_logger(__name__)


class NeonAuditLogger:
    def __init__(self) -> None:
        self.db_url = settings.neon_database_url
        self.pool: Any = None
        self._table_initialized = False

    async def get_pool(self) -> Any:
        if self.pool is None:
            try:
                import asyncpg
                self.pool = await asyncpg.create_pool(
                    dsn=self.db_url,
                    min_size=1,
                    max_size=5,
                    timeout=10.0,
                    ssl="require",
                )
                logger.info("Connected to Neon Serverless PostgreSQL Cloud Pool.")
            except ImportError:
                logger.debug("asyncpg driver not installed in current environment. Using simulated audit logger.")
                return None
            except Exception as e:
                logger.warning(f"Unable to connect to Neon PostgreSQL: {e}")
                return None
        return self.pool

    async def init_db(self) -> None:
        """Initializes the audit_logs table if not exists."""
        if self._table_initialized:
            return

        pool = await self.get_pool()
        if not pool:
            return

        query = """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id SERIAL PRIMARY KEY,
            request_id VARCHAR(64) NOT NULL,
            client_ip VARCHAR(64),
            masked_prompt TEXT,
            model_id VARCHAR(128),
            response_formats JSONB,
            pii_count INT DEFAULT 0,
            cached_hit BOOLEAN DEFAULT FALSE,
            execution_time_ms FLOAT,
            status_code INT DEFAULT 200,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_audit_request_id ON audit_logs(request_id);
        CREATE INDEX IF NOT EXISTS idx_audit_created_at ON audit_logs(created_at);
        """
        try:
            async with pool.acquire() as conn:
                await conn.execute(query)
                self._table_initialized = True
                logger.info("Neon PostgreSQL audit_logs table verified successfully.")
        except Exception as e:
            logger.warning(f"Failed to initialize audit_logs table: {e}")

    init_table = init_db

    async def log_request(
        self,
        request_id: str,
        client_ip: str,
        masked_prompt: str,
        model_id: str,
        response_formats: dict[str, Any],
        pii_count: int,
        cached_hit: bool,
        execution_time_ms: float,
        status_code: int = 200,
    ) -> None:
        """Asynchronously inserts an audit trail record without blocking the response."""
        try:
            pool = await self.get_pool()
            if not pool:
                logger.debug(f"[Audit Simulated] Logged request {request_id} (PII count: {pii_count}, Model: {model_id})")
                return

            query = """
            INSERT INTO audit_logs (
                request_id, client_ip, masked_prompt, model_id, response_formats,
                pii_count, cached_hit, execution_time_ms, status_code
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9);
            """
            async with pool.acquire() as conn:
                await conn.execute(
                    query,
                    request_id,
                    client_ip,
                    masked_prompt,
                    model_id,
                    json.dumps(response_formats),
                    pii_count,
                    cached_hit,
                    execution_time_ms,
                    status_code,
                )
            logger.info(f"Audit Log committed to Neon Cloud for request {request_id}")
        except Exception as e:
            logger.warning(f"Failed to record audit log on Neon Cloud: {e}")


neon_audit_logger = NeonAuditLogger()
