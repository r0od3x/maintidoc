"""Phase 6 — serveur MCP de maintenance et actions sensibles.

    from maintidoc.mcp_server import MaintenanceService, ApprovalSigner
"""

from maintidoc.mcp_server.approvals import ApprovalError, ApprovalSigner
from maintidoc.mcp_server.models import (
    Equipment,
    Intervention,
    Ticket,
    TicketStatus,
    ToolCallRecord,
)
from maintidoc.mcp_server.repository import MaintenanceRepository
from maintidoc.mcp_server.seed import seed
from maintidoc.mcp_server.server import (
    TOOL_DEFINITIONS,
    MaintenanceToolRouter,
    build_mcp_server,
)
from maintidoc.mcp_server.service import (
    AccessRefused,
    CreatedTicket,
    MaintenanceService,
    ToolError,
)

__all__ = [
    "TOOL_DEFINITIONS",
    "AccessRefused",
    "ApprovalError",
    "ApprovalSigner",
    "CreatedTicket",
    "Equipment",
    "Intervention",
    "MaintenanceRepository",
    "MaintenanceService",
    "MaintenanceToolRouter",
    "Ticket",
    "TicketStatus",
    "ToolCallRecord",
    "ToolError",
    "build_mcp_server",
    "seed",
]
