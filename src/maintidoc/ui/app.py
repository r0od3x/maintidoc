"""Streamlit front end, speaking to the API over HTTP.

It holds no model, no index and no database: every permission decision is
taken server-side. A front end that could reach the retriever directly would
be a way around the access control, so it deliberately cannot.

Replaces `legacy/app_streamlit_prototype.py`, which rebuilt an index on every
start, had no authentication, and asked the technician to retype their
question in English for the image search.

    uv run streamlit run src/maintidoc/ui/app.py
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API = os.environ.get("MAINTIDOC_API_URL", "http://127.0.0.1:8000")
TIMEOUT = 180

st.set_page_config(page_title="MaintiDoc", page_icon="🛠️", layout="wide")


def api(method: str, path: str, **kwargs):
    token = st.session_state.get("token")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    headers.update(kwargs.pop("headers", {}))
    return requests.request(method, f"{API}{path}", headers=headers, timeout=TIMEOUT, **kwargs)


def login_panel() -> None:
    st.sidebar.header("Connexion")
    user_id = st.sidebar.text_input("Identifiant", value="tech-san")
    password = st.sidebar.text_input("Mot de passe", type="password", value="san")
    if st.sidebar.button("Se connecter", type="primary"):
        try:
            response = api("post", "/auth/token", json={"user_id": user_id, "password": password})
        except requests.RequestException as exc:
            st.sidebar.error(f"API injoignable : {exc}")
            return
        if response.status_code != 200:
            st.sidebar.error("Identifiants invalides")
            return
        payload = response.json()
        st.session_state["token"] = payload["access_token"]
        st.session_state["user"] = payload["user_id"]
        st.session_state["groups"] = payload["groups"]
        st.rerun()


def health_panel() -> None:
    try:
        payload = api("get", "/health").json()
    except requests.RequestException:
        st.sidebar.warning("API injoignable")
        return
    st.sidebar.caption(f"Modele : {payload.get('model', 'inconnu')}")
    for note in payload.get("degraded", []):
        st.sidebar.warning(note)


def render_answer(payload: dict) -> None:
    status = payload.get("status", "")
    if status == "answered" and payload.get("sourced"):
        st.success("Reponse sourcee")
    elif status == "needs_clarification":
        st.info("Clarification demandee")
    else:
        st.warning(f"Statut : {status} — reponse non presentee comme sourcee")

    st.markdown(f"### Reponse\n\n{payload.get('answer', '')}")

    citations = payload.get("citations", [])
    if citations:
        st.markdown("#### Sources")
        for citation in citations:
            mark = "OK" if citation["structurally_verified"] else "NON VERIFIEE"
            st.write(
                f"- `{citation['ref']}` — page {citation['page']} "
                f"[structure : {mark} | soutien : {citation['grounding']}]"
            )
    else:
        st.caption("Aucune source structuree fournie.")

    for uncertainty in payload.get("uncertainties", []):
        st.warning(uncertainty)
    for warning in payload.get("warnings", []):
        st.caption(f"Avertissement : {warning}")

    with st.expander("Preuves recuperees et parcours", expanded=False):
        st.write("Elements transmis :", payload.get("evidence_refs", []))
        st.write("Parcours :", " -> ".join(payload.get("trace", [])))
        st.write("Budgets :", payload.get("budgets", {}))


def approval_panel(payload: dict) -> None:
    """Show the exact content to be created, and nothing less.

    The technician approves these words; the server binds the approval to this
    content, so an edit here means a new approval.
    """
    pending = payload.get("pending_approval")
    if not pending:
        return

    st.divider()
    st.subheader("Validation humaine requise")
    st.caption(
        "Le ticket sera cree exactement avec ce contenu. Toute modification "
        "invalide l'approbation precedente."
    )
    ticket = pending["ticket"]
    title = st.text_input("Titre", value=ticket["title"])
    description = st.text_area("Description", value=ticket["description"], height=180)
    st.caption(f"Equipement : {ticket['equipment_id']}")

    approve, refuse = st.columns(2)
    decision = None
    if approve.button("Approuver et creer", type="primary"):
        decision = True
    if refuse.button("Refuser"):
        decision = False
    if decision is None:
        return

    body = {"thread_id": payload["thread_id"], "approved": decision}
    edited = title != ticket["title"] or description != ticket["description"]
    if decision and edited:
        body["ticket"] = {
            "equipment_id": ticket["equipment_id"],
            "title": title,
            "description": description,
        }
    result = api("post", "/approve", json=body)
    if result.status_code != 200:
        st.error(f"Echec : {result.text}")
        return
    st.session_state["last"] = result.json()
    st.rerun()


def main() -> None:
    st.title("MaintiDoc · assistant documentaire de maintenance")
    st.caption(
        "Aide documentaire. Ne pilote aucun equipement, ne garantit aucun "
        "diagnostic, ne cree aucun ticket sans validation humaine."
    )

    health_panel()
    if not st.session_state.get("token"):
        login_panel()
        st.info("Connectez-vous pour poser une question.")
        return

    st.sidebar.success(f"{st.session_state['user']}")
    st.sidebar.caption("Groupes : " + ", ".join(st.session_state.get("groups", [])))
    if st.sidebar.button("Se deconnecter"):
        st.session_state.clear()
        st.rerun()

    question = st.text_area(
        "Question du technicien",
        value="Que signifie le voyant orange fixe sur le SAN-LAB 2000 ?",
        height=90,
    )
    left, right = st.columns(2)
    equipment_id = left.text_input("Equipement (optionnel)", value="SAN-LAB-2000")
    prepare_ticket = right.checkbox("Preparer un ticket")
    st.file_uploader(
        "Photographie (transmise a l'API dans une prochaine version)",
        type=["png", "jpg", "jpeg"],
    )

    if st.button("Interroger", type="primary"):
        body = {
            "question": question,
            "equipment_id": equipment_id or None,
            "prepare_ticket": prepare_ticket,
            "thread_id": f"ui-{st.session_state['user']}",
        }
        try:
            response = api("post", "/ask", json=body)
        except requests.RequestException as exc:
            st.error(f"API injoignable : {exc}")
            return
        if response.status_code not in (200, 202):
            st.error(f"Erreur {response.status_code} : {response.text}")
            return
        st.session_state["last"] = response.json()

    payload = st.session_state.get("last")
    if payload:
        render_answer(payload)
        approval_panel(payload)
        if payload.get("ticket_id"):
            st.success(f"Ticket cree : {payload['ticket_id']}")


main()
