"""Behavioural tests for ``oai2.verification.reproducibility`` (WP-58 / WI-DET-002).

Pins REQ-DET-021 through REQ-DET-026:

- **REQ-DET-021** a harness can request deterministic mode.
- **REQ-DET-022** a record preserves the metadata needed to reproduce.
- **REQ-DET-023** tool/environment snapshots are referenced, not assumed.
- **REQ-DET-024** a mismatch names the diverging dimension.
- **REQ-DET-025** deterministic mode coexists with research sampling.
- **REQ-DET-026** the flow is local — no runner, no network.

The load-bearing cases are the *negative* ones. Every absence-as-clean path in
this module exists because the alternative reads well in a summary: an
unreadable git state, an undisclosed model field, an unparsed artifact. Each of
those has a test that fails if it is ever reported as a clean value.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from oai2.verification.reproducibility import (
    REPRODUCTION_FIELDS,
    CorpusRef,
    DeterminismMode,
    DivergenceDimension,
    EnvironmentSnapshot,
    ModelIdentity,
    ReproductionRecord,
    SamplingPolicy,
    SourceIdentity,
    diagnose,
    inspect_artifact,
    source_identity,
    summarise,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = REPO_ROOT / "evidence"

# Captured verbatim from the live NORMAL lane on :8851 via `GET /props`.
# Pinned as a literal so the test does not depend on a running server, and so a
# change in what llama-server actually reports has to be a deliberate edit.
LIVE_PROPS: dict[str, object] = {
    "model_alias": "smollm2-1.7b-q4km",
    "model_path": "/Users/orchords/models/normal/SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
    "model_ftype": "Q4_K - Medium",
    "total_slots": 4,
    "default_generation_settings": {"n_ctx": 8192, "params": {}},
    "build_info": "b11323-f11d642a2",
    "is_sleeping": False,
    "modalities": {"vision": False, "audio": False},
    "bos_token": 1,
    "eos_token": 2,
    "chat_template": "{{ bos_token }}",
    "chat_template_caps": {"supports_tools": True},
    "endpoint_metrics": {},
    "endpoint_props": [],
    "endpoint_slots": [],
    "ui": True,
    "ui_settings": {},
    "cors_proxy_enabled": False,
    "media_marker": "<__media__>",
}


def _corpus() -> CorpusRef:
    return CorpusRef(
        suite_id="deterministic_hard",
        revision_id="rev-9f2c1a",
        case_count=12,
        scorer="regex_or",
        scorer_version="2026-10-04",
    )


def _complete_record(**overrides: object) -> ReproductionRecord:
    """A record with no blockers, so each test changes exactly one fact."""
    base: dict[str, object] = {
        "label": "deterministic_hard@baseline",
        "source": SourceIdentity(commit="a" * 40, branch="main", dirty=False),
        "model": ModelIdentity(
            endpoint="http://127.0.0.1:8851",
            alias="smollm2-1.7b-q4km",
            model_path="/models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
            model_sha256="7" * 64,
            quantization="Q4_K - Medium",
            backend="b11323-f11d642a2",
            total_slots=4,
            n_ctx=8192,
        ),
        "sampling": SamplingPolicy(mode=DeterminismMode.DETERMINISTIC, seed=7),
        "environment": EnvironmentSnapshot(
            platform="Darwin",
            machine="arm64",
            python="3.14.7",
            tool_snapshot_refs=("uv.lock@sha256:deadbeef",),
        ),
        "corpus": _corpus(),
    }
    base.update(overrides)
    return ReproductionRecord(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 1. SourceIdentity — an unreadable git state is unknown, never clean
# ---------------------------------------------------------------------------


def test_unreadable_git_is_unknown_not_clean(tmp_path: Path) -> None:
    """REQ-DET-022: a directory that is not a repo must not report a clean tree.

    This is the module's central invariant. ``dirty=False`` means "the tree was
    clean", so a failed read that produced ``False`` would be indistinguishable
    from a verified clean checkout.
    """
    identity = source_identity(cwd=str(tmp_path))

    assert identity.commit == "unavailable"
    assert identity.dirty is None
    assert identity.read_error is not None
    assert identity.is_usable_for_reproduction is False


def test_source_identity_reads_the_real_repository() -> None:
    identity = source_identity(cwd=str(REPO_ROOT))

    assert identity.commit != "unavailable"
    assert len(identity.commit) == 40
    assert identity.branch
    assert identity.dirty in (True, False, None)
    assert identity.is_usable_for_reproduction is True


def test_as_dict_preserves_the_existing_harness_block_shape() -> None:
    """The four scripts emit ``{commit, branch, dirty}``; consolidation must not
    change any committed artifact's shape."""
    clean = SourceIdentity(commit="b" * 40, branch="main", dirty=False)
    unknown = SourceIdentity(commit="b" * 40, dirty=None, read_error="boom")

    assert clean.as_dict() == {"commit": "b" * 40, "branch": "main", "dirty": "false"}
    assert unknown.as_dict() == {
        "commit": "b" * 40,
        "branch": "unknown",
        "dirty": "unknown",
    }


def test_source_identity_rejects_an_empty_commit() -> None:
    with pytest.raises(ValueError, match="commit"):
        SourceIdentity(commit="   ")


# ---------------------------------------------------------------------------
# 2. ModelIdentity — read from what /props actually emits
# ---------------------------------------------------------------------------


def test_from_props_reads_the_keys_llama_server_actually_emits() -> None:
    """Quantization is ``model_ftype`` and the backend is ``build_info``.

    Reading ``quantization``/``backend`` instead returns ``None`` against a
    perfectly healthy server, reporting a fully identified model as unknown.
    """
    identity = ModelIdentity.from_props(
        endpoint="http://127.0.0.1:8851", props=LIVE_PROPS
    )

    assert identity.alias == "smollm2-1.7b-q4km"
    assert identity.model_path.endswith("SmolLM2-1.7B-Instruct-Q4_K_M.gguf")
    assert identity.quantization == "Q4_K - Medium"
    assert identity.backend == "b11323-f11d642a2"
    assert identity.total_slots == 4
    assert identity.n_ctx == 8192


def test_from_props_leaves_the_weights_digest_unset_because_props_has_none() -> None:
    """No digest is served, so it stays ``None`` — absence, not a fake value.

    ``scripts/numerical_backend_evidence.py`` proves the digest is obtainable,
    but only by hashing the file on disk. A reader must not manufacture one.
    """
    identity = ModelIdentity.from_props(
        endpoint="http://127.0.0.1:8851", props=LIVE_PROPS
    )

    assert "model_sha256" not in LIVE_PROPS
    assert identity.model_sha256 is None
    assert identity.with_weights_digest("7" * 64).model_sha256 == "7" * 64


def test_from_props_tolerates_a_sparse_props_document() -> None:
    identity = ModelIdentity.from_props(
        endpoint="http://127.0.0.1:8851", props={"model_alias": ""}
    )

    assert identity.alias is None
    assert identity.quantization is None
    assert identity.n_ctx is None
    assert identity.total_slots is None


def test_from_props_ignores_wrongly_typed_numeric_fields() -> None:
    identity = ModelIdentity.from_props(
        endpoint="http://127.0.0.1:8851",
        props={
            "total_slots": True,
            "default_generation_settings": {"n_ctx": "8192"},
        },
    )

    assert identity.total_slots is None
    assert identity.n_ctx is None


def test_model_identity_rejects_non_positive_slots() -> None:
    with pytest.raises(ValueError, match="total_slots"):
        ModelIdentity(endpoint="http://x", total_slots=0)
    with pytest.raises(ValueError, match="n_ctx"):
        ModelIdentity(endpoint="http://x", n_ctx=True)


# ---------------------------------------------------------------------------
# 3. REQ-DET-021 / REQ-DET-025 — deterministic and research modes coexist
# ---------------------------------------------------------------------------


def test_deterministic_mode_requires_a_seed() -> None:
    """A "deterministic" run with no seed is a run that looked deterministic once."""
    with pytest.raises(ValueError, match="seed"):
        SamplingPolicy(mode=DeterminismMode.DETERMINISTIC)


def test_deterministic_mode_requires_zero_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        SamplingPolicy(mode=DeterminismMode.DETERMINISTIC, seed=7, temperature=0.7)


def test_research_mode_permits_sampling_and_is_not_deterministic() -> None:
    """REQ-DET-025: a sampling run is declared, not mistaken for a broken one."""
    policy = SamplingPolicy(
        mode=DeterminismMode.RESEARCH, seed=7, temperature=0.9, top_p=0.95
    )

    assert policy.is_deterministic is False
    assert policy.temperature == 0.9
    assert policy.top_p == 0.95


def test_top_p_is_bounded_to_the_runtime_contract() -> None:
    """``oai2.runtime.inference.InferenceRequest`` is ``ge=0.0, le=1.0``."""
    with pytest.raises(ValueError, match="top_p"):
        SamplingPolicy(mode=DeterminismMode.RESEARCH, top_p=1.5)


def test_non_finite_sampling_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="temperature"):
        SamplingPolicy(mode=DeterminismMode.RESEARCH, temperature=float("inf"))
    with pytest.raises(ValueError, match="top_p"):
        SamplingPolicy(mode=DeterminismMode.RESEARCH, top_p=float("nan"))


def test_sampling_policy_rejects_booleans_where_integers_are_required() -> None:
    with pytest.raises(ValueError, match="seed"):
        SamplingPolicy(mode=DeterminismMode.RESEARCH, seed=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 4. REQ-DET-022 / REQ-DET-023 — blockers name what is missing
# ---------------------------------------------------------------------------


def test_a_complete_record_has_no_blockers() -> None:
    assert _complete_record().blockers() == ()


def test_blockers_name_every_missing_fact() -> None:
    record = ReproductionRecord(
        label="bare",
        source=SourceIdentity(commit="unavailable", read_error="no git"),
        model=ModelIdentity(endpoint="http://127.0.0.1:8851"),
        sampling=SamplingPolicy(mode=DeterminismMode.RESEARCH, temperature=0.8),
        environment=EnvironmentSnapshot(platform="Darwin", machine="arm64", python="3.14.7"),
    )

    assert set(record.blockers()) == {
        "source_identity",
        "model_digest",
        "model_quantization",
        "model_backend",
        "sampling_not_deterministic",
        "corpus_revision",
        "tool_snapshot_refs",
    }


def test_a_dirty_tree_is_a_blocker_but_an_unknown_tree_state_is_not_reported_dirty() -> None:
    """``dirty=True`` blocks; ``dirty=None`` is already covered by read_error."""
    dirty = _complete_record(
        source=SourceIdentity(commit="a" * 40, branch="main", dirty=True)
    )
    unreadable = _complete_record(
        source=SourceIdentity(commit="unavailable", read_error="no git")
    )

    assert "source_tree_dirty" in dirty.blockers()
    assert "source_tree_dirty" not in unreadable.blockers()
    assert "source_identity" in unreadable.blockers()


def test_corpus_ref_validates_its_fields() -> None:
    with pytest.raises(ValueError, match="revision_id"):
        CorpusRef(
            suite_id="s", revision_id="", case_count=1, scorer="r", scorer_version="v"
        )
    with pytest.raises(ValueError, match="case_count"):
        CorpusRef(
            suite_id="s", revision_id="r", case_count=-1, scorer="r", scorer_version="v"
        )


def test_environment_snapshot_drops_versions_on_request() -> None:
    snapshot = EnvironmentSnapshot.current(
        runtime_versions={"llama-cpp-python": "0.3.0", "httpx": "0.28.0"}
    )

    assert snapshot.runtime_versions == (
        ("httpx", "0.28.0"),
        ("llama-cpp-python", "0.3.0"),
    )
    assert snapshot.without_versions().runtime_versions == ()


# ---------------------------------------------------------------------------
# 5. REQ-DET-024 — a mismatch names its dimension
# ---------------------------------------------------------------------------


def test_an_identical_record_reproduces() -> None:
    diagnosis = diagnose(_complete_record(), _complete_record())

    assert diagnosis.reproduced is True
    assert diagnosis.divergences == ()
    assert diagnosis.primary_dimension() is None
    assert diagnosis.is_explained is True


def test_a_changed_source_revision_is_named_as_source() -> None:
    diagnosis = diagnose(
        _complete_record(),
        _complete_record(source=SourceIdentity(commit="c" * 40, branch="main", dirty=False)),
    )

    assert diagnosis.reproduced is False
    assert diagnosis.dimensions() == ("source",)
    assert "commit" in diagnosis.divergences[0].detail


def test_a_changed_serving_artifact_is_named_as_model() -> None:
    other = _complete_record(
        model=ModelIdentity(
            endpoint="http://127.0.0.1:8851",
            alias="smollm2-1.7b-q4km",
            model_path="/models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
            model_sha256="9" * 64,  # different weights on the same path
            quantization="Q4_K - Medium",
            backend="b11323-f11d642a2",
            total_slots=4,
            n_ctx=8192,
        )
    )

    diagnosis = diagnose(_complete_record(), other)

    assert "model" in diagnosis.dimensions()
    assert "model_sha256" in diagnosis.divergences[0].detail


def test_a_changed_sampling_policy_is_named_as_sampling() -> None:
    diagnosis = diagnose(
        _complete_record(),
        _complete_record(
            sampling=SamplingPolicy(mode=DeterminismMode.DETERMINISTIC, seed=8)
        ),
    )

    assert "sampling" in diagnosis.dimensions()
    assert "seed=7/8" in diagnosis.divergences[0].detail


def test_a_changed_host_is_named_as_environment() -> None:
    diagnosis = diagnose(
        _complete_record(),
        _complete_record(
            environment=EnvironmentSnapshot(
                platform="Linux", machine="x86_64", python="3.14.7"
            )
        ),
    )

    assert "environment" in diagnosis.dimensions()
    assert "host Darwin/arm64 != Linux/x86_64" in diagnosis.divergences[0].detail


def test_a_changed_corpus_is_named_as_corpus() -> None:
    diagnosis = diagnose(
        _complete_record(),
        _complete_record(
            corpus=CorpusRef(
                suite_id="deterministic_hard",
                revision_id="rev-0000",
                case_count=12,
                scorer="regex_or",
                scorer_version="2026-10-04",
            )
        ),
    )

    assert "corpus" in diagnosis.dimensions()


def test_primary_dimension_prefers_source_over_anything_else() -> None:
    """Source is the cheapest axis to rule out and explains most differences."""
    diagnosis = diagnose(
        _complete_record(),
        _complete_record(
            source=SourceIdentity(commit="c" * 40, branch="main", dirty=False),
            corpus=CorpusRef(
                suite_id="deterministic_hard",
                revision_id="rev-0000",
                case_count=12,
                scorer="regex_or",
                scorer_version="2026-10-04",
            ),
        ),
    )

    assert set(diagnosis.dimensions()) == {"source", "corpus"}
    assert diagnosis.primary_dimension() == "source"
    assert diagnosis.is_explained is True


def test_dimension_values_are_the_documented_ones() -> None:
    assert [d.value for d in DivergenceDimension] == [
        "source",
        "model",
        "sampling",
        "environment",
        "corpus",
    ]


# ---------------------------------------------------------------------------
# 6. Artifact coverage — measured over the artifacts this repo committed
# ---------------------------------------------------------------------------


def test_artifact_written_with_alternative_key_spellings_is_fully_covered() -> None:
    """The negative control for the probe list.

    The committed artifacts were written by different harnesses and name the
    same fact differently. A single-key probe scores a genuine gap as a false
    alarm, and a detector that cries wolf is worse than none.
    """
    coverage = inspect_artifact(
        "alt-spelling",
        {
            "harness": {"source_sha": "d" * 40, "branch": "main", "dirty": "false"},
            "serving_identity": {
                "base_url": "http://127.0.0.1:8851",
                "model_alias": "smollm2-1.7b-q4km",
                "model_path": "/m.gguf",
                "weights_sha256": "7" * 64,
                "model_ftype": "Q4_K - Medium",
                "build_info": "b11323-f11d642a2",
                "total_slots": 4,
                "n_ctx": 8192,
            },
            "sampling": {"mode": "deterministic", "seed": 7, "temperature": 0.0},
            "suite_id": "deterministic_hard",
            "revision": {"revision_id": "rev-9f2c1a"},
            "scorer": "regex_or@2026-10-04",
            "environment": {
                "platform": "Darwin",
                "python": "3.14.7",
                "tool_snapshot_refs": ["uv.lock@sha256:deadbeef"],
            },
        },
    )

    assert coverage.missing == ()
    assert coverage.coverage_ratio == 1.0
    # The resolving path is recorded, so a reader can see which spelling matched.
    assert dict(coverage.present)["source.commit"] == "harness.source_sha"
    assert dict(coverage.present)["model.digest"] == "serving_identity.weights_sha256"


def test_an_empty_object_is_missing_everything_not_unreadable() -> None:
    coverage = inspect_artifact("empty", {})

    assert coverage.is_unreadable is False
    assert coverage.missing == tuple(f.name for f in REPRODUCTION_FIELDS)
    assert coverage.coverage_ratio == 0.0


def test_an_unparseable_artifact_is_unreadable_not_clean() -> None:
    """``[]`` parsed fine but is not a record; it must not score as 0.0 coverage
    and be averaged in as though it had been inspected."""
    coverage = inspect_artifact("a-list", ["not", "a", "record"])

    assert coverage.is_unreadable is True
    assert coverage.missing == ()
    assert coverage.coverage_ratio == 0.0


def test_a_null_value_does_not_satisfy_a_field() -> None:
    coverage = inspect_artifact("nulls", {"harness": {"commit": None, "branch": None}})

    assert "source.commit" in coverage.missing
    assert "source.branch" in coverage.missing


def test_summarise_raises_on_an_empty_input() -> None:
    with pytest.raises(ValueError, match="no artifacts"):
        summarise([])


def test_summarise_excludes_unreadable_artifacts_from_the_mean() -> None:
    summary = summarise(
        [
            inspect_artifact("good", {"harness": {"commit": "a" * 40}}),
            inspect_artifact("broken", ["nope"]),
        ]
    )

    assert [c.label for c in summary.unreadable] == ["broken"]
    assert [c.label for c in summary.readable] == ["good"]
    assert summary.mean_coverage() is not None
    assert 0.0 < summary.mean_coverage() < 1.0  # type: ignore[operator]


def test_summarise_reports_no_mean_when_nothing_was_read() -> None:
    summary = summarise([inspect_artifact("broken", ["nope"])])

    assert summary.mean_coverage() is None
    assert summary.fully_reproducible() == ()


def test_list_probe_matches_when_any_element_carries_the_key() -> None:
    coverage = inspect_artifact(
        "suites", {"suites": [{"suite_id": "a", "revision_id": "r1"}, {"suite_id": "b"}]}
    )

    assert "corpus.revision" not in coverage.missing
    assert dict(coverage.present)["corpus.revision"] == "suites[].revision_id"


# ---------------------------------------------------------------------------
# 7. The real measurement: every committed evidence artifact
# ---------------------------------------------------------------------------


def _committed_artifacts() -> list[Path]:
    return sorted(EVIDENCE_DIR.rglob("*.json"))


def test_the_repository_actually_commits_evidence_artifacts() -> None:
    """Guards the measurement below from silently passing on an empty set."""
    assert _committed_artifacts(), "no evidence artifacts found to audit"


def test_every_committed_artifact_is_readable() -> None:
    """Every committed artifact parses into something the auditor can inspect.

    Readability only. How many fields an artifact carries is the measurement
    below — asserting coverage here would assert the finding away.
    """
    for path in _committed_artifacts():
        document = json.loads(path.read_text(encoding="utf-8"))
        coverage = inspect_artifact(path.name, document)
        assert coverage.is_unreadable is False, f"{path.name} was unreadable"


def test_at_least_one_committed_artifact_records_no_source_revision() -> None:
    """REQ-DET-022, measured on real data.

    ``cache_invalidation_probe.json`` carries no ``harness`` block at all, so
    the verdict in it cannot be tied to any revision of the code that produced
    it. A verifier replaying it would have no way to know what they were
    replaying. This test exists so that gap has to be closed deliberately.
    """
    summary = summarise(
        [
            inspect_artifact(path.name, json.loads(path.read_text(encoding="utf-8")))
            for path in _committed_artifacts()
        ]
    )

    assert summary.artifacts_missing("source.commit")


def test_no_committed_artifact_records_a_tool_snapshot() -> None:
    """REQ-DET-023, measured on real data.

    This is the finding the audit exists to produce: not a hypothetical gap, and
    not a claim. If a future artifact does reference a tool snapshot, this test
    fails and the finding has to be retired deliberately rather than quietly.
    """
    summary = summarise(
        [
            inspect_artifact(path.name, json.loads(path.read_text(encoding="utf-8")))
            for path in _committed_artifacts()
        ]
    )

    assert summary.readable
    offenders = summary.artifacts_missing("environment.tool_snapshots")

    assert offenders, "expected the real gap to still be present"
    assert len(offenders) == len(summary.readable)


def test_committed_artifacts_do_not_record_a_corpus_revision() -> None:
    """REQ-DET-022, measured on real data: the revision field post-dates every
    committed artifact, so none of them is pinned to a content-addressed corpus."""
    summary = summarise(
        [
            inspect_artifact(path.name, json.loads(path.read_text(encoding="utf-8")))
            for path in _committed_artifacts()
        ]
    )

    assert summary.artifacts_missing("corpus.revision")


# ---------------------------------------------------------------------------
# 8. REQ-DET-026 — the flow is local
# ---------------------------------------------------------------------------


def test_module_does_not_import_network_or_runner_clients() -> None:
    """REQ-DET-026: reproduction is a local, manual flow.

    Guards against a later edit wiring an HTTP client or a CI runner in here.
    """
    source = (REPO_ROOT / "oai2" / "verification" / "reproducibility.py").read_text(
        encoding="utf-8"
    )

    for forbidden in ("import requests", "import httpx", "github_actions", "workflow_run"):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# 9. The four private harness_identity copies are gone
# ---------------------------------------------------------------------------
#
# Measured, not assumed. Before this change each of the four scripts ran git
# against the *current working directory*. Git walks up the tree to find a
# repository, so a subdirectory was harmless; the failure was invoking a script
# from outside the repository, which recorded ``commit: "unavailable:OSError"``
# for a perfectly healthy tree. The shared owner anchors at the repository
# root, so it resolves from any working directory.

_HARNESS_SCRIPTS = (
    "scripts/bench.py",
    "scripts/admission_probe.py",
    "scripts/contamination_audit.py",
    "scripts/eval_governance_report.py",
)


@pytest.mark.parametrize("relative", _HARNESS_SCRIPTS)
def test_scripts_delegate_provenance_to_the_single_owner(relative: str) -> None:
    """Each script calls the owner rather than reimplementing it.

    A fifth private copy of "which revision am I" is the failure this closes,
    so the delegation itself is pinned, not just the module's behaviour.
    """
    path = REPO_ROOT / relative
    source = path.read_text(encoding="utf-8")

    assert "source_identity(" in source, f"{relative} does not call the owner"
    assert "rev-parse" not in source, f"{relative} still shells out to git itself"

    # No dict may still carry a fabricated commit string. Checked through the
    # AST rather than with a substring search, because these files legitimately
    # name the old value in their docstrings while explaining why it went away.
    tree = ast.parse(source, filename=relative)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=False):
            if not (isinstance(key, ast.Constant) and key.value == "commit"):
                continue
            assert not (
                isinstance(value, ast.Constant)
                and isinstance(value.value, str)
                and value.value.startswith("unavailable")
            ), f"{relative} still fabricates a commit value"


def test_harness_identity_resolves_when_invoked_outside_the_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression, pinned: run from outside the tree, provenance still resolves.

    Executed for real, not inferred: the working directory is moved to a
    directory that is not inside any git repository, which is the case the
    previous implementation reported as ``unavailable:OSError``.
    """
    monkeypatch.chdir(tmp_path)
    identity = source_identity(cwd=str(REPO_ROOT))

    assert identity.commit != "unavailable"
    assert len(identity.commit) == 40
    assert identity.is_usable_for_reproduction is True
