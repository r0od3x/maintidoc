"""LangGraph orchestration.

    check_access -> retrieve -> [maybe_history] -> compose -> verify
                 -> answer | clarify | draft_ticket
                 -> interrupt(approbation humaine) -> create_ticket

LangGraph manages the state, the transitions and the interrupt; the budgets
are enforced in the nodes, because a graph edge cannot refuse to be taken.

The human approval is a real interrupt backed by a checkpointer, not a
callback. The run stops, the state is persisted, and nothing resumes until a
decision arrives — which is what makes "no creation without approval"
structural rather than a matter of the agent behaving well.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from maintidoc.agent.budget import BudgetExceeded
from maintidoc.agent.state import AgentState
from maintidoc.contracts import AnswerStatus, TicketDraft
from maintidoc.generation.answering import Answerer
from maintidoc.generation.client import GenerationError
from maintidoc.mcp_server.approvals import ApprovalError
from maintidoc.mcp_server.service import MaintenanceService, ToolError
from maintidoc.retrieval.retriever import AccessDenied, Retriever


def _note(state: AgentState, message: str) -> None:
    state.setdefault("trace", []).append(message)


def _warn(state: AgentState, message: str) -> None:
    state.setdefault("warnings", []).append(message)


def _stop(state: AgentState, status: str, detail: str) -> AgentState:
    state["outcome"] = {"status": status, "detail": detail}
    _note(state, f"arret: {status}")
    return state


class MaintiDocGraph:
    """Builds and runs the agent graph."""

    def __init__(
        self,
        retriever: Retriever,
        answerer: Answerer,
        *,
        service: MaintenanceService | None = None,
        top_k: int = 6,
    ) -> None:
        self.retriever = retriever
        self.answerer = answerer
        self.service = service
        self.top_k = top_k
        self.checkpointer = MemorySaver()
        self.graph = self._build()

    # --- nodes ------------------------------------------------------------

    def check_access(self, state: AgentState) -> AgentState:
        principal = state["principal"]
        _note(state, "check_access")
        if not principal.can_read_anything:
            return _stop(
                state, "access_denied",
                f"{principal.user_id} n'appartient a aucun groupe autorise",
            )
        return state

    def retrieve(self, state: AgentState) -> AgentState:
        budget = state["budget"]
        try:
            budget.spend_retrieval()
            budget.check_time()
        except BudgetExceeded as exc:
            return _stop(state, "budget_exceeded", str(exc))

        _note(state, f"retrieve (tour {budget.retrieval_rounds})")
        try:
            result = self.retriever.retrieve(
                state["question"],
                state["principal"],
                visual_query=state.get("visual_query"),
                top_k=self.top_k,
            )
        except AccessDenied as exc:
            return _stop(state, "access_denied", str(exc))

        state["bundle"] = result.bundle
        if not result.bundle.items:
            _warn(state, "Aucune preuve recuperee")
        return state

    def consult_history(self, state: AgentState) -> AgentState:
        """Optional MCP lookup, bounded by the MCP budget."""
        equipment_id = state.get("equipment_id")
        if self.service is None or not equipment_id:
            return state

        budget = state["budget"]
        try:
            budget.spend_mcp()
        except BudgetExceeded as exc:
            _warn(state, str(exc))
            return state

        _note(state, "consult_history (MCP)")
        try:
            interventions = self.service.get_equipment_history(
                user_id=state["principal"].user_id, equipment_id=equipment_id
            )
            state["history"] = [i.model_dump(mode="json") for i in interventions]
        except ToolError as exc:
            # A refused tool call is information, not a crash.
            _warn(state, f"Historique indisponible : {exc}")
        return state

    def compose(self, state: AgentState) -> AgentState:
        bundle = state.get("bundle")
        if bundle is None:
            return _stop(state, "failed", "Aucun dossier de preuves")

        budget = state["budget"]
        try:
            budget.spend_model()
            budget.check_time()
        except BudgetExceeded as exc:
            return _stop(state, "budget_exceeded", str(exc))

        _note(state, "compose")
        try:
            result = self.answerer.answer(bundle)
        except GenerationError as exc:
            return _stop(state, "failed", f"Generation impossible : {exc}")

        state["answer"] = result.answer
        state["grounded"] = result.is_sourced
        for warning in result.warnings:
            _warn(state, warning)
        return state

    def verify(self, state: AgentState) -> AgentState:
        """Decide what the run produced. The checks already ran in `compose`."""
        answer = state.get("answer")
        if answer is None:
            return _stop(state, "failed", "Aucune reponse produite")
        _note(state, f"verify: {answer.status.value}")
        return state

    def draft_ticket(self, state: AgentState) -> AgentState:
        answer = state["answer"]
        equipment_id = state.get("equipment_id")
        if self.service is None or not equipment_id:
            return _stop(state, "answered", "Reponse fournie, aucun ticket demande")

        assert answer is not None
        state["draft"] = TicketDraft(
            equipment_id=equipment_id,
            title=f"Verification documentaire : {state['question'][:80]}",
            description=(
                f"{answer.answer}\n\n"
                f"Sources citees : {', '.join(answer.cited_refs) or 'aucune'}.\n"
                "Brouillon prepare par MaintiDoc, a valider par un technicien habilite."
            ),
        )
        _note(state, "draft_ticket")
        return state

    def await_approval(self, state: AgentState) -> AgentState:
        """Stop and wait for a human. Nothing is created before this returns.

        The clock is paused: a technician reading the draft is not spending
        the processing budget.
        """
        draft = state.get("draft")
        if draft is None:
            return _stop(state, "answered", "Aucun brouillon a approuver")

        budget = state["budget"]
        budget.pause()
        _note(state, "await_approval (interruption humaine)")
        decision = interrupt(
            {
                "action": "create_ticket",
                "ticket": draft.model_dump(mode="json"),
                "content_hash": draft.content_hash(),
                "message": "Approuver la creation de ce ticket ?",
            }
        )
        budget.resume()

        approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
        if not approved:
            return _stop(state, "rejected", "Creation refusee par l'operateur")

        # An edited draft is a different draft: it must be re-approved, so the
        # edit is recorded and the approval re-issued against the new content.
        if isinstance(decision, dict) and decision.get("ticket"):
            state["draft"] = TicketDraft.model_validate(decision["ticket"])
            _note(state, "brouillon modifie par l'operateur")
        return state

    def create_ticket(self, state: AgentState) -> AgentState:
        draft = state.get("draft")
        if self.service is None or draft is None:
            return _stop(state, "failed", "Service de maintenance indisponible")

        budget = state["budget"]
        try:
            budget.spend_mcp()
        except BudgetExceeded as exc:
            return _stop(state, "budget_exceeded", str(exc))

        user_id = state["principal"].user_id
        try:
            approval = self.service.prepare_approval(user_id=user_id, draft=draft)
            created = self.service.create_ticket(
                user_id=user_id, draft=draft, approval=approval
            )
        except (ApprovalError, ToolError) as exc:
            return _stop(state, "failed", f"Creation refusee : {exc}")

        state["approval"] = approval
        state["ticket_id"] = created.ticket.ticket_id
        _note(state, "create_ticket")
        return _stop(
            state,
            "ticket_created" if created.created else "ticket_existing",
            created.ticket.ticket_id,
        )

    def finish(self, state: AgentState) -> AgentState:
        if state.get("outcome") is None:
            answer = state.get("answer")
            status = answer.status.value if answer else "failed"
            state["outcome"] = {"status": status, "detail": ""}
        return state

    # --- edges ------------------------------------------------------------

    @staticmethod
    def _after_access(state: AgentState) -> str:
        return "stop" if state.get("outcome") else "retrieve"

    @staticmethod
    def _after_retrieve(state: AgentState) -> str:
        if state.get("outcome"):
            return "stop"
        return "history" if state.get("equipment_id") else "compose"

    @staticmethod
    def _after_compose(state: AgentState) -> str:
        return "stop" if state.get("outcome") else "verify"

    @staticmethod
    def _after_verify(state: AgentState) -> str:
        """Route on the answer's own status.

        A ticket is only drafted from an answer that held up: an unsupported
        or unclear answer must not become a maintenance request.
        """
        answer = state.get("answer")
        if answer is None:
            return "stop"
        if answer.status is not AnswerStatus.ANSWERED:
            return "stop"
        if state.get("wants_ticket") and state.get("equipment_id"):
            return "draft"
        return "stop"

    @staticmethod
    def _after_approval(state: AgentState) -> str:
        return "stop" if state.get("outcome") else "create"

    def _build(self):
        graph = StateGraph(AgentState)
        graph.add_node("check_access", self.check_access)
        graph.add_node("retrieve", self.retrieve)
        graph.add_node("consult_history", self.consult_history)
        graph.add_node("compose", self.compose)
        graph.add_node("verify", self.verify)
        graph.add_node("draft_ticket", self.draft_ticket)
        graph.add_node("await_approval", self.await_approval)
        graph.add_node("create_ticket", self.create_ticket)
        graph.add_node("finish", self.finish)

        graph.set_entry_point("check_access")
        graph.add_conditional_edges(
            "check_access", self._after_access,
            {"retrieve": "retrieve", "stop": "finish"},
        )
        graph.add_conditional_edges(
            "retrieve", self._after_retrieve,
            {"history": "consult_history", "compose": "compose", "stop": "finish"},
        )
        graph.add_edge("consult_history", "compose")
        graph.add_conditional_edges(
            "compose", self._after_compose,
            {"verify": "verify", "stop": "finish"},
        )
        graph.add_conditional_edges(
            "verify", self._after_verify,
            {"draft": "draft_ticket", "stop": "finish"},
        )
        graph.add_edge("draft_ticket", "await_approval")
        graph.add_conditional_edges(
            "await_approval", self._after_approval,
            {"create": "create_ticket", "stop": "finish"},
        )
        graph.add_edge("create_ticket", "finish")
        graph.add_edge("finish", END)
        return graph.compile(checkpointer=self.checkpointer)

    # --- running ----------------------------------------------------------

    def run(self, state: AgentState, *, thread_id: str = "default") -> dict[str, Any]:
        config = {"configurable": {"thread_id": thread_id}}
        return self.graph.invoke(state, config=config)

    def resume(self, decision: dict[str, Any], *, thread_id: str = "default") -> dict[str, Any]:
        from langgraph.types import Command

        config = {"configurable": {"thread_id": thread_id}}
        return self.graph.invoke(Command(resume=decision), config=config)

    def pending_approval(self, *, thread_id: str = "default") -> dict[str, Any] | None:
        """The payload the human is being asked to decide on, if any."""
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = self.graph.get_state(config)
        for task in snapshot.tasks:
            for request in getattr(task, "interrupts", ()) or ():
                return request.value
        return None
