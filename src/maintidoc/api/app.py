"""FastAPI application.

Every route except `/health` and `/auth/token` requires a verified token, and
the `Principal` it yields is the only source of permissions in the request.
A body field claiming a user or a group is never read — the request models use
`extra="forbid"`, so attempting it is rejected outright.

The approval step is a real pause. `POST /ask` with `prepare_ticket` returns
**202** together with the exact ticket content; nothing is created until
`POST /approve` carries a decision. The status code is part of the contract,
so a client cannot mistake "waiting for a human" for "done".

Dependency aliases are module level on purpose: this module uses
`from __future__ import annotations`, so FastAPI resolves annotations through
module globals. Defined inside the factory they would be invisible, and every
dependency would silently degrade into a query parameter.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from maintidoc.agent.state import initial_state
from maintidoc.api.deps import Runtime
from maintidoc.api.security import AuthError, TokenResponse
from maintidoc.contracts import Principal
from maintidoc.mcp_server.service import ToolError

bearer = HTTPBearer(auto_error=False)


# --- request and response bodies -----------------------------------------


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str
    password: str


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)
    equipment_id: str | None = None
    visual_query: str | None = None
    prepare_ticket: bool = False
    thread_id: str | None = None


class ApproveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str
    approved: bool
    ticket: dict[str, Any] | None = None


class CitationOut(BaseModel):
    ref: str
    document_id: str
    page: int
    element_id: str
    structurally_verified: bool
    grounding: str


class AskResponse(BaseModel):
    status: str
    answer: str
    citations: list[CitationOut] = []
    uncertainties: list[str] = []
    clarification_question: str | None = None
    sourced: bool = False
    evidence_refs: list[str] = []
    warnings: list[str] = []
    trace: list[str] = []
    budgets: dict[str, Any] = {}
    thread_id: str
    pending_approval: dict[str, Any] | None = None
    ticket_id: str | None = None
    outcome: str = ""


# --- dependencies ---------------------------------------------------------


def get_runtime(request: Request) -> Runtime:
    return request.app.state.runtime


def current_principal(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> Principal:
    if credentials is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Authentification requise",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return get_runtime(request).tokens.principal(credentials.credentials)
    except AuthError as exc:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


RuntimeDep = Annotated[Runtime, Depends(get_runtime)]
PrincipalDep = Annotated[Principal, Depends(current_principal)]


# --- application ----------------------------------------------------------


def create_app(runtime: Runtime | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Models are built once, here, and shared by every request. Loading
        # them per request fails outright on a machine that cannot hold two
        # copies — which is how the constraint was discovered in phase 4.
        app.state.runtime = runtime or Runtime.full()
        try:
            yield
        finally:
            app.state.runtime.close()

    app = FastAPI(
        title="MaintiDoc",
        version="0.1.0",
        description="Assistant documentaire multimodal et agentique (maintenance SAN).",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(runtime: RuntimeDep) -> dict[str, Any]:
        return {
            "status": "ok",
            "model": runtime.model_name,
            "corpus_version": runtime.settings.corpus_version,
            "degraded": runtime.degraded,
        }

    @app.post("/auth/token", response_model=TokenResponse)
    def login(body: LoginRequest, runtime: RuntimeDep) -> TokenResponse:
        try:
            user = runtime.directory.authenticate(body.user_id, body.password)
        except AuthError as exc:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc
        return runtime.tokens.issue(user)

    @app.get("/me")
    def me(principal: PrincipalDep) -> dict[str, Any]:
        return {"user_id": principal.user_id, "groups": sorted(principal.groups)}

    @app.post("/ask", response_model=AskResponse)
    def ask(
        body: AskRequest,
        principal: PrincipalDep,
        runtime: RuntimeDep,
        response: Response,
    ) -> AskResponse:
        thread_id = body.thread_id or f"{principal.user_id}-{id(body):x}"
        state = initial_state(
            body.question,
            principal,
            runtime.new_budget(),
            equipment_id=body.equipment_id,
            visual_query=body.visual_query,
            wants_ticket=body.prepare_ticket,
        )
        result = runtime.graph.run(state, thread_id=thread_id)
        pending = runtime.graph.pending_approval(thread_id=thread_id)
        if pending is not None:
            response.status_code = status.HTTP_202_ACCEPTED
        return _to_response(result, thread_id, pending)

    @app.post("/approve", response_model=AskResponse)
    def approve(
        body: ApproveRequest, principal: PrincipalDep, runtime: RuntimeDep
    ) -> AskResponse:
        pending = runtime.graph.pending_approval(thread_id=body.thread_id)
        if pending is None:
            raise HTTPException(
                status.HTTP_409_CONFLICT, "Aucune approbation en attente pour ce fil"
            )
        decision: dict[str, Any] = {"approved": body.approved}
        if body.ticket:
            decision["ticket"] = body.ticket
        result = runtime.graph.resume(decision, thread_id=body.thread_id)
        return _to_response(result, body.thread_id, None)

    @app.get("/equipment/{equipment_id}/tickets")
    def tickets(
        equipment_id: str, principal: PrincipalDep, runtime: RuntimeDep
    ) -> dict[str, Any]:
        try:
            found = runtime.service.get_open_tickets(
                user_id=principal.user_id, equipment_id=equipment_id
            )
        except ToolError as exc:
            raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
        return {"tickets": [t.model_dump(mode="json") for t in found]}

    @app.get("/equipment/{equipment_id}/history")
    def history(
        equipment_id: str, principal: PrincipalDep, runtime: RuntimeDep
    ) -> dict[str, Any]:
        try:
            found = runtime.service.get_equipment_history(
                user_id=principal.user_id, equipment_id=equipment_id
            )
        except ToolError as exc:
            raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
        return {"interventions": [i.model_dump(mode="json") for i in found]}

    return app


def _to_response(
    result: dict[str, Any], thread_id: str, pending: dict[str, Any] | None
) -> AskResponse:
    answer = result.get("answer")
    bundle = result.get("bundle")
    outcome = result.get("outcome") or {}
    budget = result.get("budget")

    citations: list[CitationOut] = []
    if answer is not None:
        allowed = bundle.allowed_refs if bundle else frozenset()
        for citation in answer.citations:
            citations.append(
                CitationOut(
                    ref=citation.ref,
                    document_id=citation.document_id,
                    page=citation.page,
                    element_id=citation.element_id,
                    structurally_verified=citation.ref in allowed,
                    grounding="verifie" if result.get("grounded") else "non etabli",
                )
            )

    return AskResponse(
        status=answer.status.value if answer else outcome.get("status", "failed"),
        answer=answer.answer if answer else outcome.get("detail", ""),
        citations=citations,
        uncertainties=list(answer.uncertainties) if answer else [],
        clarification_question=answer.clarification_question if answer else None,
        sourced=bool(result.get("grounded")),
        evidence_refs=sorted(bundle.allowed_refs) if bundle else [],
        warnings=list(result.get("warnings", [])),
        trace=list(result.get("trace", [])),
        budgets=budget.snapshot() if budget is not None else {},
        thread_id=thread_id,
        pending_approval=pending,
        ticket_id=result.get("ticket_id"),
        outcome=outcome.get("status", ""),
    )
