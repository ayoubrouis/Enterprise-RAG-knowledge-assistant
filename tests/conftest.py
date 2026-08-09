"""Global test configuration.

Imported by pytest before any test module, so it must set the signing secret
before ``app.config`` (and thus ``settings``) is first imported. The app
refuses to start without a strong ``RAG_SECRET_KEY``.
"""

from __future__ import annotations

import os

os.environ.setdefault("RAG_SECRET_KEY", "test-secret-0123456789abcdef")
# Keep the shared /query rate limiter from interfering across tests; the
# limiter itself is exercised directly by dedicated tests.
os.environ.setdefault("RAG_QUERY_RATE_LIMIT_MAX", "1000000")
