"""Fleet wire contracts and the portable worker boundary (WI-FLEET-001, #242).

This package holds the *definition* WP-77 builds on. It owns no authority:
enrollment, lease and routing stay in the existing pipeline, Cloudflare stays
the sole knowledge authority, and tool execution stays client-owned. See
:mod:`oai2.fleet.contract`.
"""

from __future__ import annotations

from .contract import (
    AUTHORISING_STATES,
    CONTRACT_VERSION,
    SUPPORTED_CONTRACT_VERSIONS,
    BackendVersion,
    CapabilityState,
    ContractError,
    JobContract,
    NodeIdentity,
    Readiness,
    ResourceEnvelope,
    ResultContract,
    WorkerCapability,
    assert_no_secrets,
    check_compatible,
    negotiate_version,
)

__all__ = [
    "AUTHORISING_STATES",
    "BackendVersion",
    "CONTRACT_VERSION",
    "CapabilityState",
    "ContractError",
    "JobContract",
    "NodeIdentity",
    "Readiness",
    "ResourceEnvelope",
    "ResultContract",
    "SUPPORTED_CONTRACT_VERSIONS",
    "WorkerCapability",
    "assert_no_secrets",
    "check_compatible",
    "negotiate_version",
]
