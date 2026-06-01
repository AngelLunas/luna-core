"""luna-core public API surface.

Host applications import everything they need from this top-level module.
Nothing here triggers side effects (no FastAPI app, no uvicorn, no Celery
worker boot) — instantiating these collaborators is the host's job.
"""
from luna_core.engine.agent import AgentRunner
from luna_core.engine.emitter import EventEmitter
from luna_core.engine.nodes import NodeExecutor
from luna_core.engine.runner import FlowRunner
from luna_core.engine.websocket import WebSocketManager
from luna_core.llm.base import (
    AbortSignalError,
    BaseLLMProvider,
    LLMRateLimitError,
    ToolDefinition,
)
from luna_core.llm.providers.generic import GenericProvider
from luna_core.llm.router import LLMRouter
from luna_core.mcp.builder import MCPServerBuilder
from luna_core.mcp.client import MCPClient
from luna_core.services.dedup import DedupService
from luna_core.services.embedding import EmbeddingService
from luna_core.services.rag import RAGService
from luna_core.storage.base import BaseStorageBackend
from luna_core.storage.service import build_storage_backend

__version__ = "0.3.0"

__all__ = [
    "AbortSignalError",
    "AgentRunner",
    "BaseLLMProvider",
    "BaseStorageBackend",
    "DedupService",
    "EmbeddingService",
    "EventEmitter",
    "FlowRunner",
    "GenericProvider",
    "LLMRateLimitError",
    "LLMRouter",
    "MCPClient",
    "MCPServerBuilder",
    "NodeExecutor",
    "RAGService",
    "ToolDefinition",
    "WebSocketManager",
    "__version__",
    "build_storage_backend",
]
