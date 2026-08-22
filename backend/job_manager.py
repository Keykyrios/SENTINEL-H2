"""
Thread-pool based job queue. No Redis, no Celery.
Uses concurrent.futures.ThreadPoolExecutor to run simulation jobs in background threads.
Results stored in an in-memory dict keyed by job_id.
"""

import uuid
import time
import traceback
import logging
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any, Callable, Dict, Optional

from config import MAX_WORKERS

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobRecord:
    __slots__ = ("job_id", "status", "result", "error", "submitted_at", "completed_at", "pillar")

    def __init__(self, job_id: str, pillar: str):
        self.job_id = job_id
        self.pillar = pillar
        self.status = JobStatus.PENDING
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.submitted_at = time.time()
        self.completed_at: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "pillar": self.pillar,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "elapsed": round(time.time() - self.submitted_at, 2),
        }


class JobManager:
    """Manages background simulation jobs using a thread pool."""

    def __init__(self):
        self._pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self._jobs: Dict[str, JobRecord] = {}
        logger.info("JobManager initialized with %d workers.", MAX_WORKERS)

    def submit(self, task_fn: Callable, params: dict, pillar: str) -> str:
        """Submit a simulation task. Returns job_id immediately."""
        job_id = str(uuid.uuid4())
        record = JobRecord(job_id, pillar)
        self._jobs[job_id] = record

        def _run():
            record.status = JobStatus.RUNNING
            try:
                record.result = task_fn(params)
                record.status = JobStatus.COMPLETED
            except Exception:
                record.error = traceback.format_exc()
                record.status = JobStatus.FAILED
                logger.error("Job %s failed:\n%s", job_id, record.error)
            finally:
                record.completed_at = time.time()

        self._pool.submit(_run)
        return job_id

    def get(self, job_id: str) -> Optional[JobRecord]:
        return self._jobs.get(job_id)

    def get_status(self, job_id: str) -> Optional[dict]:
        rec = self._jobs.get(job_id)
        if rec is None:
            return None
        return rec.to_dict()


# Singleton instance
manager = JobManager()
