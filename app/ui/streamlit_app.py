"""Streamlit chat UI - a client of the FastAPI backend.

Run with:
    streamlit run app/ui/streamlit_app.py

The API must be running first (default http://127.0.0.1:8000; override with
the RAG_API_BASE_URL environment variable). In Docker the UI is a separate
container that talks to the API service over the compose network.
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import quote

# Make the `app` package importable. Streamlit adds this script's folder to
# sys.path (not the project root), so we add the root explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import httpx
import streamlit as st

from app.config import settings

BASE_URL = settings.API_BASE_URL
TIMEOUT = httpx.Timeout(600.0)


def _headers() -> dict:
    token = st.session_state.get("token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _request(method: str, path: str, **kw) -> httpx.Response:
    resp = httpx.request(
        method, f"{BASE_URL}{path}", headers=_headers(), timeout=TIMEOUT, **kw
    )
    if resp.status_code == 401 and "token" in st.session_state:
        st.session_state.pop("token", None)
        st.session_state.pop("user", None)
        st.rerun()
    return resp


st.set_page_config(page_title="Enterprise RAG Assistant", page_icon=":material/search:")
st.title("Enterprise RAG Knowledge Assistant")
st.caption("Local-first answers with source citations from your internal documents.")

# ---------------------------------------------------------------------------
# Login gate / first-run setup wizard
# ---------------------------------------------------------------------------

if "user" not in st.session_state:
    setup_resp = httpx.get(f"{BASE_URL}/auth/setup", timeout=30)
    setup_needed = setup_resp.status_code == 200 and setup_resp.json().get("needed", False)

    if setup_needed:
        st.write("**First run** - create your enterprise and admin account.")
        st.caption("This is done once; afterwards only sign-in is available.")
        tenant_name = st.text_input("Company / tenant name")
        username = st.text_input("Admin username")
        password = st.text_input("Admin password", type="password")
        confirm = st.text_input("Confirm password", type="password")
        if st.button("Create admin account", type="primary"):
            if not tenant_name or len(username) < 3 or len(password) < 8:
                st.error("Tenant name, a username of at least 3 characters, and a password of at least 8 characters are required.")
            elif password != confirm:
                st.error("Passwords do not match.")
            else:
                resp = httpx.post(
                    f"{BASE_URL}/auth/setup",
                    json={
                        "tenant_name": tenant_name,
                        "username": username,
                        "password": password,
                    },
                    timeout=30,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.token = data["token"]
                    st.session_state.user = data
                    st.rerun()
                else:
                    detail = resp.json().get("detail") if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                    st.error(detail if isinstance(detail, str) else "Setup failed.")
    else:
        st.write("Sign in to continue.")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Sign in", type="primary"):
            if not username or not password:
                st.error("Enter both username and password.")
            else:
                resp = httpx.post(
                    f"{BASE_URL}/auth/login",
                    json={"username": username, "password": password},
                    timeout=30,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.token = data["token"]
                    st.session_state.user = data
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
    st.stop()

user = st.session_state["user"]

# ---------------------------------------------------------------------------
# Sidebar: tenant info + document management
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header(f"Tenant: {user['tenant_name']}")
    st.caption(f"Signed in as **{user['username']}** ({user['role']}) · `{user['tenant_id']}`")

    if user["role"] in ("admin", "superadmin"):
        view = st.radio("View", ["Chat", "Admin"], horizontal=True)
    else:
        view = "Chat"

    st.divider()
    st.subheader("Documents")

    uploaded = st.file_uploader(
        "Upload a document",
        type=[ext.lstrip(".") for ext in settings.SUPPORTED_EXTENSIONS],
    )
    if uploaded is not None and st.button("Upload & index", type="primary"):
        with st.spinner("Uploading, embedding, and indexing..."):
            resp = _request(
                "post",
                "/documents",
                files={
                    "file": (
                        uploaded.name,
                        uploaded.getvalue(),
                        uploaded.type or "application/octet-stream",
                    )
                },
            )
        if resp.status_code == 200:
            info = resp.json()
            st.success(f"Indexed {info['documents']} document(s) / {info['chunks']} chunk(s).")
            st.rerun()
        else:
            st.error(resp.text)

    if st.button("Rebuild index from stored docs"):
        with st.spinner("Rebuilding index..."):
            resp = _request("post", "/ingest")
        if resp.status_code == 200:
            st.success("Index rebuilt.")
            st.rerun()
        else:
            st.error(resp.text)

    resp = _request("get", "/documents")
    if resp.status_code == 200:
        docs = resp.json()
        st.caption(f"Stored documents ({len(docs)})")
        for doc in docs:
            col1, col2 = st.columns([5, 1])
            col1.write(doc["filename"])
            if col2.button("x", key=f"del_{doc['filename']}"):
                _request("delete", f"/documents/{quote(doc['filename'])}")
                st.rerun()

    st.divider()
    st.caption(f"Embedding model: `{settings.EMBEDDING_MODEL}`")
    st.caption(f"LLM backend: `{settings.LLM_BACKEND}` · model `{settings.LLM_MODEL}`")

    if st.button("Sign out"):
        st.session_state.clear()
        st.rerun()

# ---------------------------------------------------------------------------
# Admin panel
#   superadmin  -> whole app: create tenants, manage any tenant, see all logs
#   admin       -> own enterprise only: manage its users + API keys
# ---------------------------------------------------------------------------


def _tenant_management(tenant_id: str, can_disable: bool, t: dict) -> None:
    if can_disable:
        col1, col2 = st.columns([3, 1])
        if col2.button("Disable" if t["is_active"] else "Enable", key=f"tog_{tenant_id}"):
            _request("patch", f"/admin/tenants/{tenant_id}", json={"is_active": not t["is_active"]})
            st.rerun()

    # A freshly created secret (password / API key) survives the rerun so it can
    # be copied. It is stored only in this session, never by the server.
    pending = st.session_state.pop(f"new_secret_{tenant_id}", None)
    if pending:
        label, secret = pending
        st.code(secret, language=None)
        st.warning(f"{label} - shown once. Share it securely now; it will not be shown again.")

    role_choices = ["user", "admin", "superadmin"] if user["role"] == "superadmin" else ["user", "admin"]

    st.write("**Users**")
    users = _request("get", f"/admin/tenants/{tenant_id}/users").json()
    for u in users:
        uc1, uc2, uc3 = st.columns([4, 2, 2])
        uc1.caption(f"{u['username']} ({u['role']})")
        uc2.caption("active" if u["is_active"] else "disabled")
        if uc3.button("Disable" if u["is_active"] else "Enable", key=f"utog_{tenant_id}_{u['username']}"):
            _request("patch", f"/admin/tenants/{tenant_id}/users/{u['username']}", json={"is_active": not u["is_active"]})
            st.rerun()
    with st.form(f"new_user_{tenant_id}"):
        uname = st.text_input("Username", key=f"nu_{tenant_id}")
        upass = st.text_input(
            "Password (leave blank to auto-generate)", type="password", key=f"np_{tenant_id}"
        )
        urole = st.selectbox("Role", role_choices, key=f"nr_{tenant_id}")
        if st.form_submit_button("Create user"):
            if len(uname) < 3:
                st.error("Username must be at least 3 characters.")
            elif upass and len(upass) < 8:
                st.error("Password (if set) must be at least 8 characters.")
            else:
                r = _request(
                    "post",
                    f"/admin/tenants/{tenant_id}/users",
                    json={"username": uname, "password": upass, "role": urole},
                )
                if r.status_code == 200:
                    data = r.json()
                    if data.get("password"):
                        st.session_state[f"new_secret_{tenant_id}"] = (
                            f"Password for '{uname}'",
                            data["password"],
                        )
                    st.success(f"User '{uname}' created.")
                    st.rerun()
                else:
                    st.error(r.text)

    st.write("**API keys**")
    keys = _request("get", f"/admin/tenants/{tenant_id}/api-keys").json()
    for k in keys:
        st.caption(f"{k['label'] or '(no label)'} · `{k['key_hash'][:12]}…` · {'active' if k['is_active'] else 'disabled'}")
    with st.form(f"new_key_{tenant_id}"):
        label = st.text_input("Label", key=f"nk_{tenant_id}")
        if st.form_submit_button("Create API key"):
            r = _request(
                "post",
                f"/admin/tenants/{tenant_id}/api-keys",
                json={"label": label},
            )
            if r.status_code == 200:
                st.session_state[f"new_secret_{tenant_id}"] = (
                    f"API key '{label or '(no label)'}'",
                    r.json()["key"],
                )
                st.success("API key created.")
                st.rerun()
            else:
                st.error(r.text)


def admin_panel() -> None:
    st.subheader("Admin")
    is_super = user["role"] == "superadmin"

    if is_super:
        st.write("**Create tenant**")
        with st.form("new_tenant"):
            tid = st.text_input("Tenant ID (a-z, 0-9, `-`/`_`; up to 64 chars)", key="nt_id")
            tname = st.text_input("Display name", key="nt_name")
            if st.form_submit_button("Create tenant"):
                if not tid:
                    st.error("Tenant ID is required.")
                else:
                    resp = _request("post", "/admin/tenants", json={"tenant_id": tid, "name": tname})
                    if resp.status_code == 200:
                        st.success(f"Tenant '{tid}' created.")
                        st.rerun()
                    else:
                        st.error(resp.text)

        resp = _request("get", "/admin/tenants")
        if resp.status_code != 200:
            st.error(resp.text)
            return
        st.divider()
        st.write("**Tenants**")
        for t in resp.json():
            title = f"{t['name']} · `{t['tenant_id']}` · {t['documents']} docs · {'active' if t['is_active'] else 'disabled'}"
            with st.expander(title):
                _tenant_management(t["tenant_id"], can_disable=True, t=t)
    else:
        # Enterprise admin: only their own enterprise, no cross-tenant powers.
        tid = user["tenant_id"]
        title = f"{user['tenant_name']} · `{tid}` · {'active'}"
        with st.expander(title):
            _tenant_management(tid, can_disable=False, t={})


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

if view == "Admin":
    admin_panel()
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.write(text)

stats = _request("get", "/stats")
if stats.status_code == 200 and stats.json().get("chunks", 0) == 0:
    st.info("No documents indexed for this tenant yet. Upload a document from the sidebar.")
    st.stop()

if prompt := st.chat_input("Ask about your documents..."):
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            resp = _request("post", "/query", json={"question": prompt})

        if resp.status_code == 200:
            result = resp.json()
            st.write(result["answer"])
            st.caption(f"Grounding sources ({len(result['sources'])})")
            for s in result["sources"]:
                page = f" - page {s['page']}" if s.get("page") is not None else ""
                with st.expander(f"{s['source']}{page} (sim {s['similarity']:.2f})"):
                    st.write(s["snippet"])
            st.session_state.history.append(("user", prompt))
            st.session_state.history.append(("assistant", result["answer"]))
        else:
            st.error(resp.text)
