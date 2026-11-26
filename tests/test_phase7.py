"""Agent orchestration: routing, budgets, and the human interrupt.

Driven with in-memory retrieval and a scripted model, so the graph's control
flow is what is under test rather than retrieval quality.
"""

from __future__ import annotations

import json

import pytest

from maintidoc.agent import BudgetExceeded, BudgetTracker, MaintiDocGraph, initial_state
from maintidoc.contracts import (
    AnswerStatus,
    Evidence,
    EvidenceBundle,
    Modality,
    Principal,
    RetrievalBranch,
    TicketDraft,
)
from maintidoc.generation import Answerer, LexicalGroundingChecker, StubVlmClient
from maintidoc.mcp_server import (
    ApprovalSigner,
    MaintenanceRepository,
    MaintenanceService,
    seed,
)
from maintidoc.mcp_server.seed import equipment_id_for
from maintidoc.retrieval.retriever import RetrievalResult, RetrievalTrace

SAN = equipment_id_for("SAN-LAB 2000")
THERMAL = (
    "Sur le SAN-LAB 2000, le voyant orange fixe signale un avertissement "
    "thermique sur le controleur A. Verifier la ventilation du local."
)
TECH = Principal(user_id="tech-san", groups=frozenset({"maintenance", "maintenance_san"}))


def _bundle(question: str = "Que signifie le voyant orange fixe ?") -> EvidenceBundle:
    return EvidenceBundle(
        question=question,
        items=(
            Evidence(
                document_id="SAN-2000-MANU",
                page=2,
                element_id="p02_text_01",
                modality=Modality.TEXT,
                excerpt=THERMAL,
                branch=RetrievalBranch.TEXT,
                rank=1,
            ),
        ),
    )


class FakeRetriever:
    """Returns a fixed bundle and counts how often it was asked."""

    def __init__(self, bundle: EvidenceBundle | None = None) -> None:
        self.bundle = bundle if bundle is not None else _bundle()
        self.calls = 0

    def retrieve(self, question, principal, *, visual_query=None, top_k=8):
        self.calls += 1
        return RetrievalResult(bundle=self.bundle, candidates=[], trace=RetrievalTrace())


def _answer_json(answer: str, citations: list[dict], status: str = "answered") -> str:
    return json.dumps(
        {
            "status": status,
            "answer": answer,
            "citations": citations,
            "uncertainties": [],
            "clarification_question": None,
            "proposed_action": None,
        },
        ensure_ascii=False,
    )


CITE = [{"document_id": "SAN-2000-MANU", "page": 2, "element_id": "p02_text_01"}]


def _answerer(*responses: str) -> Answerer:
    return Answerer(
        StubVlmClient(responses=list(responses)), checker=LexicalGroundingChecker()
    )


@pytest.fixture
def service():
    repository = MaintenanceRepository()
    seed(repository)
    yield MaintenanceService(repository, ApprovalSigner("secret-de-test"))
    repository.close()


def _graph(answerer: Answerer, *, service=None, retriever=None) -> MaintiDocGraph:
    return MaintiDocGraph(retriever or FakeRetriever(), answerer, service=service)


def _state(**overrides):
    base = initial_state("Que signifie le voyant orange fixe ?", TECH, BudgetTracker())
    base.update(overrides)
    return base


# --- the acceptance criterion ---------------------------------------------


def test_an_induced_retrieval_loop_stops_on_budget():
    """The phase 7 acceptance criterion.

    A tracker already at its retrieval limit must make the node refuse rather
    than loop. The limit is enforced by a counter, not by asking the model to
    behave.
    """
    budget = BudgetTracker(max_retrieval_rounds=2)
    budget.retrieval_rounds = 2
    graph = _graph(_answerer(_answer_json(THERMAL, CITE)))

    result = graph.run(_state(budget=budget), thread_id="loop")

    assert result["outcome"]["status"] == "budget_exceeded"
    assert "recuperation" in result["outcome"]["detail"]


def test_the_model_call_budget_stops_composition():
    budget = BudgetTracker(max_model_calls=1)
    budget.model_calls = 1
    graph = _graph(_answerer(_answer_json(THERMAL, CITE)))
    result = graph.run(_state(budget=budget), thread_id="model-budget")
    assert result["outcome"]["status"] == "budget_exceeded"


def test_the_mcp_budget_is_enforced(service):
    budget = BudgetTracker(max_mcp_calls=0)
    graph = _graph(_answerer(_answer_json(THERMAL, CITE)), service=service)
    result = graph.run(
        _state(budget=budget, equipment_id=SAN), thread_id="mcp-budget"
    )
    # History is skipped but the run still answers.
    assert any("Budget depasse" in w for w in result["warnings"])
    assert result["answer"] is not None


def test_budget_counters_are_independent():
    budget = BudgetTracker(max_mcp_calls=1, max_model_calls=1)
    budget.spend_mcp()
    budget.spend_model()
    with pytest.raises(BudgetExceeded, match="MCP"):
        budget.spend_mcp()
    with pytest.raises(BudgetExceeded, match="modele"):
        budget.spend_model()


def test_time_spent_waiting_for_a_human_is_not_charged():
    """Otherwise the limit punishes the technician for reading carefully."""
    budget = BudgetTracker(max_seconds=0.5)
    budget.pause()
    budget._paused_at -= 5.0  # simulate five seconds of human deliberation
    budget.resume()
    budget.check_time()  # must not raise


# --- routing --------------------------------------------------------------


def test_a_principal_without_groups_never_reaches_retrieval():
    retriever = FakeRetriever()
    graph = _graph(_answerer(), retriever=retriever)
    result = graph.run(
        _state(principal=Principal(user_id="inconnu")), thread_id="no-group"
    )
    assert result["outcome"]["status"] == "access_denied"
    assert retriever.calls == 0


def test_a_grounded_answer_without_a_ticket_request_simply_answers():
    graph = _graph(_answerer(_answer_json(THERMAL, CITE)))
    result = graph.run(_state(), thread_id="plain")
    assert result["answer"].status is AnswerStatus.ANSWERED
    assert result["ticket_id"] is None
    assert "verify: answered" in result["trace"]


def test_an_unsupported_answer_never_reaches_a_ticket(service):
    """A claim its sources do not bear out must not become a work request."""
    graph = _graph(
        _answerer(_answer_json("Affirmation totalement hors sujet.", CITE)),
        service=service,
    )
    result = graph.run(
        _state(equipment_id=SAN, wants_ticket=True), thread_id="unsupported"
    )
    assert result["answer"].status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert result["draft"] is None
    assert result["ticket_id"] is None


def test_a_clarification_does_not_reach_a_ticket(service):
    clarification = json.dumps(
        {
            "status": "needs_clarification",
            "answer": "Le modele doit etre confirme.",
            "citations": [],
            "uncertainties": [],
            "clarification_question": "Pouvez-vous photographier la plaque ?",
            "proposed_action": None,
        }
    )
    graph = _graph(_answerer(clarification), service=service)
    result = graph.run(
        _state(equipment_id=SAN, wants_ticket=True), thread_id="clarify"
    )
    assert result["answer"].status is AnswerStatus.NEEDS_CLARIFICATION
    assert result["ticket_id"] is None


def test_history_is_consulted_when_equipment_is_known(service):
    graph = _graph(_answerer(_answer_json(THERMAL, CITE)), service=service)
    result = graph.run(_state(equipment_id=SAN), thread_id="history")
    assert result["history"]
    assert "consult_history (MCP)" in result["trace"]


def test_a_refused_history_lookup_does_not_abort_the_run(service):
    """tech-san may not read UPS equipment; the answer still happens."""
    graph = _graph(_answerer(_answer_json(THERMAL, CITE)), service=service)
    result = graph.run(
        _state(equipment_id=equipment_id_for("UPS-LAB 500")), thread_id="refused"
    )
    assert any("Historique indisponible" in w for w in result["warnings"])
    assert result["answer"] is not None


def test_an_empty_bundle_is_reported_not_hidden():
    graph = _graph(
        _answerer(), retriever=FakeRetriever(EvidenceBundle(question="rien"))
    )
    result = graph.run(_state(), thread_id="empty")
    assert any("Aucune preuve" in w for w in result["warnings"])
    assert result["answer"].status is AnswerStatus.INSUFFICIENT_EVIDENCE


# --- human in the loop ----------------------------------------------------


def test_the_run_stops_and_waits_before_creating_anything(service):
    """Nothing is created while the graph is interrupted."""
    graph = _graph(_answerer(_answer_json(THERMAL, CITE)), service=service)
    graph.run(_state(equipment_id=SAN, wants_ticket=True), thread_id="approval")

    pending = graph.pending_approval(thread_id="approval")
    assert pending is not None
    assert pending["action"] == "create_ticket"
    assert pending["ticket"]["equipment_id"] == SAN
    assert service.get_open_tickets(user_id="tech-san", equipment_id=SAN) == [
        t for t in service.get_open_tickets(user_id="tech-san", equipment_id=SAN)
        if t.ticket_id == "TCK-SEED00001"
    ]


def test_approval_creates_the_ticket(service):
    graph = _graph(_answerer(_answer_json(THERMAL, CITE)), service=service)
    graph.run(_state(equipment_id=SAN, wants_ticket=True), thread_id="approve")
    result = graph.resume({"approved": True}, thread_id="approve")

    assert result["outcome"]["status"] == "ticket_created"
    assert result["ticket_id"].startswith("TCK-")
    created = [
        t for t in service.get_open_tickets(user_id="tech-san", equipment_id=SAN)
        if t.ticket_id == result["ticket_id"]
    ]
    assert len(created) == 1


def test_refusal_creates_nothing(service):
    graph = _graph(_answerer(_answer_json(THERMAL, CITE)), service=service)
    graph.run(_state(equipment_id=SAN, wants_ticket=True), thread_id="reject")
    result = graph.resume({"approved": False}, thread_id="reject")

    assert result["outcome"]["status"] == "rejected"
    assert result["ticket_id"] is None
    assert len(service.get_open_tickets(user_id="tech-san", equipment_id=SAN)) == 1


def test_an_edited_draft_is_what_gets_created(service):
    """The operator's wording wins, and is the content actually approved."""
    graph = _graph(_answerer(_answer_json(THERMAL, CITE)), service=service)
    graph.run(_state(equipment_id=SAN, wants_ticket=True), thread_id="edit")

    edited = TicketDraft(
        equipment_id=SAN,
        title="Controle thermique a planifier",
        description="Formulation revue par le responsable avant creation du ticket.",
    )
    result = graph.resume(
        {"approved": True, "ticket": edited.model_dump(mode="json")}, thread_id="edit"
    )

    assert result["outcome"]["status"] == "ticket_created"
    created = next(
        t for t in service.get_open_tickets(user_id="tech-san", equipment_id=SAN)
        if t.ticket_id == result["ticket_id"]
    )
    assert created.title == "Controle thermique a planifier"
    assert "brouillon modifie par l'operateur" in result["trace"]


def test_the_trace_records_the_path_taken(service):
    graph = _graph(_answerer(_answer_json(THERMAL, CITE)), service=service)
    graph.run(_state(equipment_id=SAN, wants_ticket=True), thread_id="trace")
    result = graph.resume({"approved": True}, thread_id="trace")

    trace = result["trace"]
    for step in ("check_access", "retrieve", "compose", "draft_ticket", "create_ticket"):
        assert step in trace or any(step in entry for entry in trace)
