import logging
import os

from langchain_core.runnables import RunnableConfig

PROMPT_VERSION = "v1"
PROJECT_TAG = "unidad-7"

logger = logging.getLogger(__name__)


def tracing_enabled() -> bool:
    tracing = os.getenv("LANGSMITH_TRACING", "")
    return tracing.lower() == "true" and bool(os.getenv("LANGSMITH_API_KEY"))


def log_tracing_status() -> None:
    if tracing_enabled():
        logger.info("LangSmith tracing ON, project=%s", os.getenv("LANGSMITH_PROJECT", "default"))
    else:
        logger.warning("LangSmith tracing OFF: set LANGSMITH_TRACING=true and LANGSMITH_API_KEY in .env")


def build_run_config(job_id: str) -> RunnableConfig:
    return {
        "configurable": {"thread_id": job_id},
        "run_name": f"travel-job-{job_id[:8]}",
        "tags": [PROJECT_TAG, f"prompt:{PROMPT_VERSION}"],
        "metadata": {"job_id": job_id, "prompt_version": PROMPT_VERSION},
    }
