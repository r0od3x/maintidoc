"""API: authentication, RBAC, and the demonstration scenario end to end.

The runtime is the lean one — no large model, lexical retrieval, scripted
generator — so these run offline in milliseconds. What is under test is the
HTTP contract and the access rules, not answer quality.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from maintidoc.api import Runtime, create_app
from maintidoc.contracts import Element, Modality
from maintidoc.generation import LexicalGroundingChecker, StubVlmClient
from maintidoc.mcp_server.seed import equipment_id_for

SAN = equipment_id_for("SAN-LAB 2000")
UPS = equipment_id_for("UPS-LAB 500")
SHA = "d" * 64
THERMAL = (
    "Sur le SAN-LAB 2000, le voyant orange fixe signale un avertissement "
    "thermique sur le controleur A. Verifier la ventilation du local technique."
)


def _element(element_id: str, text: str, *, acl=("maintenance_san",), doc="SAN-2000-MANU"):
    return Element(
        document_id=doc,
        document_version="1.0",
        document_sha256=SHA,
        page=2,
        element_id=element_id,
        modality=Modality.TEXT,
        equipment_family="san" if "SAN" in doc else "onduleur",
        acl_groups=acl,
        text=text,
    )


ELEMENTS = [
    _element("p02_text_01", THERMAL),
    _element(
        "p02_text_01",
        "Sur le UPS-LAB 500, le voyant orange fixe signale un fonctionnement sur batterie.",
        acl=("maintenance_energie",),
        doc="UPS-500-MANU",
    ),
]


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


@pytest.fixture
def client():
    """A client whose generator answers correctly, several times over."""
    vlm = StubVlmClient(responses=[_answer_json(THERMAL, CITE) for _ in range(12)])
    runtime = Runtime.lean(
        vlm=vlm, checker=LexicalGroundingChecker(), elements=ELEMENTS
    )
    with TestClient(create_app(runtime)) as test_client:
        yield test_client


def _token(client: TestClient, user_id: str, password: str) -> str:
    response = client.post("/auth/token", json={"user_id": user_id, "password": password})
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# --- authentication -------------------------------------------------------


def test_health_is_public_and_reports_degradation(client: TestClient):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "modeles non charges" in " ".join(body["degraded"])


def test_every_other_route_requires_a_token(client: TestClient):
    for method, path in (
        ("get", "/me"),
        ("post", "/ask"),
        ("get", f"/equipment/{SAN}/tickets"),
        ("get", f"/equipment/{SAN}/history"),
    ):
        kwargs = {"json": {"question": "q"}} if method == "post" else {}
        response = getattr(client, method)(path, **kwargs)
        assert response.status_code == 401, path


def test_bad_credentials_are_refused(client: TestClient):
    assert client.post(
        "/auth/token", json={"user_id": "tech-san", "password": "faux"}
    ).status_code == 401


def test_an_unknown_user_is_indistinguishable_from_a_wrong_password(client: TestClient):
    unknown = client.post("/auth/token", json={"user_id": "fantome", "password": "x"})
    wrong = client.post("/auth/token", json={"user_id": "tech-san", "password": "x"})
    assert unknown.status_code == wrong.status_code == 401
    assert unknown.json()["detail"] == wrong.json()["detail"]


def test_a_forged_token_is_refused(client: TestClient):
    assert client.get("/me", headers=_auth("pas.un.jeton")).status_code == 401


def test_a_token_signed_with_another_secret_is_refused(client: TestClient):
    from maintidoc.api.security import DEMO_USERS, TokenService

    foreign = TokenService("un-autre-secret").issue(DEMO_USERS[0]).access_token
    assert client.get("/me", headers=_auth(foreign)).status_code == 401


def test_the_token_carries_the_groups(client: TestClient):
    body = client.get("/me", headers=_auth(_token(client, "tech-san", "san"))).json()
    assert body["user_id"] == "tech-san"
    assert "maintenance_san" in body["groups"]


# --- RBAC -----------------------------------------------------------------


def test_a_technician_only_sees_their_own_documents(client: TestClient):
    token = _token(client, "tech-san", "san")
    body = client.post(
        "/ask", json={"question": "Que signifie le voyant orange fixe ?"},
        headers=_auth(token),
    ).json()
    assert body["evidence_refs"]
    assert all("UPS" not in ref for ref in body["evidence_refs"])


def test_another_group_sees_the_other_half(client: TestClient):
    token = _token(client, "tech-nrj", "nrj")
    body = client.post(
        "/ask", json={"question": "Que signifie le voyant orange fixe ?"},
        headers=_auth(token),
    ).json()
    assert body["evidence_refs"]
    assert all("SAN" not in ref for ref in body["evidence_refs"])


def test_a_user_without_groups_is_denied(client: TestClient):
    token = _token(client, "visiteur", "visiteur")
    body = client.post(
        "/ask", json={"question": "Que signifie le voyant orange ?"}, headers=_auth(token)
    ).json()
    assert body["outcome"] == "access_denied"
    assert body["evidence_refs"] == []


def test_tools_refuse_equipment_outside_the_group(client: TestClient):
    token = _token(client, "tech-san", "san")
    assert client.get(f"/equipment/{UPS}/tickets", headers=_auth(token)).status_code == 403
    assert client.get(f"/equipment/{SAN}/tickets", headers=_auth(token)).status_code == 200


def test_the_body_cannot_claim_an_identity(client: TestClient):
    """Permissions come from the token, never from the request body."""
    token = _token(client, "tech-san", "san")
    response = client.post(
        "/ask",
        json={"question": "Voyant orange ?", "user_id": "chef-maint", "groups": ["tout"]},
        headers=_auth(token),
    )
    assert response.status_code == 422  # extra="forbid" rejects the attempt


# --- the demonstration scenario ------------------------------------------


def test_the_full_scenario_runs_through_the_api(client: TestClient):
    """The phase 8 acceptance criterion.

    Question, sourced answer, history through MCP, ticket prepared, approved
    and created — all over HTTP, no notebook cell involved.
    """
    token = _token(client, "tech-san", "san")

    # 1. A sourced answer.
    asked = client.post(
        "/ask",
        json={
            "question": "Que signifie le voyant orange fixe sur le SAN-LAB 2000 ?",
            "equipment_id": SAN,
            "prepare_ticket": True,
            "thread_id": "demo",
        },
        headers=_auth(token),
    )
    assert asked.status_code == 202, "l'attente d'approbation doit etre explicite"
    body = asked.json()
    assert body["status"] == "answered"
    assert body["citations"]

    # 2. The run is paused on the exact content to be created.
    pending = body["pending_approval"]
    assert pending is not None
    assert pending["ticket"]["equipment_id"] == SAN

    # 3. Nothing exists yet.
    before = client.get(f"/equipment/{SAN}/tickets", headers=_auth(token)).json()["tickets"]

    # 4. The human approves.
    approved = client.post(
        "/approve", json={"thread_id": "demo", "approved": True}, headers=_auth(token)
    )
    assert approved.status_code == 200
    created = approved.json()
    assert created["ticket_id"]
    assert created["outcome"] == "ticket_created"

    # 5. The ticket now exists, exactly once.
    after = client.get(f"/equipment/{SAN}/tickets", headers=_auth(token)).json()["tickets"]
    assert len(after) == len(before) + 1
    assert sum(t["ticket_id"] == created["ticket_id"] for t in after) == 1


def test_refusing_the_approval_creates_nothing(client: TestClient):
    token = _token(client, "tech-san", "san")
    client.post(
        "/ask",
        json={
            "question": "Voyant orange fixe ?",
            "equipment_id": SAN,
            "prepare_ticket": True,
            "thread_id": "refus",
        },
        headers=_auth(token),
    )
    before = client.get(f"/equipment/{SAN}/tickets", headers=_auth(token)).json()["tickets"]
    body = client.post(
        "/approve", json={"thread_id": "refus", "approved": False}, headers=_auth(token)
    ).json()
    after = client.get(f"/equipment/{SAN}/tickets", headers=_auth(token)).json()["tickets"]

    assert body["outcome"] == "rejected"
    assert body["ticket_id"] is None
    assert len(after) == len(before)


def test_approving_an_unknown_thread_is_a_conflict(client: TestClient):
    token = _token(client, "tech-san", "san")
    response = client.post(
        "/approve", json={"thread_id": "inexistant", "approved": True}, headers=_auth(token)
    )
    assert response.status_code == 409


def test_a_question_without_a_ticket_request_returns_200(client: TestClient):
    token = _token(client, "tech-san", "san")
    response = client.post(
        "/ask", json={"question": "Voyant orange fixe ?"}, headers=_auth(token)
    )
    assert response.status_code == 200
    assert response.json()["pending_approval"] is None


def test_the_response_reports_budgets_and_trace(client: TestClient):
    token = _token(client, "tech-san", "san")
    body = client.post(
        "/ask", json={"question": "Voyant orange fixe ?"}, headers=_auth(token)
    ).json()
    assert body["trace"]
    assert "model_calls" in body["budgets"]
    assert body["thread_id"]
