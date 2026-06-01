from luna_core.engine.agent import (
    AgentRunner,
    AgentRunnerError,
    OutputSchemaValidationError,
)
from luna_core.engine.emitter import EventEmitter, run_event_channel
from luna_core.engine.nodes import (
    HumanCheckpointInterrupt,
    NodeExecutionError,
    NodeExecutor,
)
from luna_core.engine.runner import FlowRunner, FlowState
from luna_core.engine.websocket import WebSocketManager
from luna_core.mcp.system_tools import install_builtins as _install_system_tool_builtins

# Register built-in system tools (stash_records, yield_iteration) on the
# process-wide default registry as a side effect of importing the engine.
# Hosts that need isolation construct their own SystemToolRegistry and
# pass it to AgentRunner; tests do the same. We swallow the duplicate
# registration error so re-imports during test runs don't blow up.
try:
    _install_system_tool_builtins()
except ValueError:
    pass

__all__ = [
    "AgentRunner",
    "AgentRunnerError",
    "EventEmitter",
    "FlowRunner",
    "FlowState",
    "HumanCheckpointInterrupt",
    "NodeExecutionError",
    "NodeExecutor",
    "OutputSchemaValidationError",
    "WebSocketManager",
    "run_event_channel",
]
