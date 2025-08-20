from __future__ import annotations

from typing import List
import os


def _import_amadeus_toolkit():
    """Support both new and legacy import paths for AmadeusToolkit."""
    try:
        # Preferred modern path
        from langchain_community.agent_toolkits import AmadeusToolkit  # type: ignore
        return AmadeusToolkit
    except Exception:
        # Legacy path fallback
        from langchain.agents.agent_toolkits import AmadeusToolkit  # type: ignore
        return AmadeusToolkit


def get_amadeus_tools() -> List:
    """
    Create Amadeus tools via LangChain's AmadeusToolkit.

    Requires environment variables:
    - AMADEUS_CLIENT_ID
    - AMADEUS_CLIENT_SECRET
    """
    client_id = os.getenv("AMADEUS_CLIENT_ID")
    client_secret = os.getenv("AMADEUS_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise ValueError(
            "Missing Amadeus credentials. Set AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET."
        )

    AmadeusToolkit = _import_amadeus_toolkit()
    toolkit = AmadeusToolkit()
    return toolkit.get_tools()

