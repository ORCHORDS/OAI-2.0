"""Determinism metadata and reproduction diagnosis (WP-58 / WI-DET-002).

Why this module exists
----------------------

A measured number is only reproducible if the record says enough to rebuild the
conditions that produced it. This repository had no single place that said
"enough", and the evidence shows the cost in two concrete ways:

- ``scripts/bench.py``, ``scripts/admission_probe.py``,
  ``scripts/contamination_audit.py`` and ``scripts/eval_governance_report.py``
  each carried their own private copy of "which source revision am I", all
  recording the same three keys. Four copies of provenance is a provenance
  scheme nobody can change safely, and this module is the one owner.
- The committed evidence artifacts record a git SHA, but not the model
  identity, sampling policy, corpus revision or environment snapshot needed to
  reproduce them. ``scripts/reproducibility_audit.py`` measures exactly which
  fields are missing across the artifacts this repository has actually
  committed, rather than asserting the gap.

Design points
-------------

- **Unknown is not false.** ``SourceIdentity.dirty`` is ``None`` when git
  cannot be read, and model identity fields are ``None`` when the serving path
  did not disclose them. A field that reads ``False`` or ``0.0`` because
  nothing was known is the absence-as-clean defect the rest of this repository
  has been closing, and a record carrying it would be worse than no record.
- **Identity is read, never asserted** (see also
  :class:`oai2.runtime.llamacpp_runtime.LlamaServerRuntime`). Model path,
  digest, quantization and backend are whatever the serving path reported.
- **Deterministic mode is a contract, not a hope** (REQ-DET-021, REQ-DET-025).
  :class:`DeterminismMode.DETERMINISTIC` requires a seed and a zero
  temperature; :class:`DeterminismMode.RESEARCH` explicitly permits sampling,
  so the two coexist without one silently becoming the other.
- **Diagnosis names a dimension** (REQ-DET-024). A reproduction that differs is
  not reported as "different": it is reported as a source, model, sampling,
  environment or corpus divergence, because those need different responses.
- **A record that cannot support reproduction says so.**
  :meth:`ReproductionRecord.blockers` names what is missing instead of letting
  a partial record be cited as if it were complete.

Status: IMPLEMENTED — unit-pinned in ``tests/test_reproducibility.py`` and run
over the committed artifacts by ``scripts/reproducibility_audit.py``.
"""

from __future__ import annotations

import math
import platform
import subprocess  # noqa: S404
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

__all__ = [
    "REPRODUCTION_FIELDS",
    "ArtifactCoverage",
    "CorpusRef",
    "CoverageSummary",
    "DeterminismMode",
    "Divergence",
    "DivergenceDimension",
    "EnvironmentSnapshot",
    "ModelIdentity",
    "ReproductionDiagnosis",
    "ReproductionField",
    "ReproductionRecord",
    "SamplingPolicy",
    "SourceIdentity",
    "diagnose",
    "inspect_artifact",
    "source_identity",
    "summarise",
]


class DeterminismMode(StrEnum):
    """How a run was sampled.

    Both members are first-class. ``RESEARCH`` exists so a non-deterministic
    research run is *declared* rather than being indistinguishable from a
    deterministic one that happened to sample differently (REQ-DET-025).
    """

    DETERMINISTIC = "deterministic"
    RESEARCH = "research"


class DivergenceDimension(StrEnum):
    """Which axis of a reproduction attempt differs."""

    SOURCE = "source"
    MODEL = "model"
    SAMPLING = "sampling"
    ENVIRONMENT = "environment"
    CORPUS = "corpus"


@dataclass(slots=True, frozen=True)
class SourceIdentity:
    """The exact source revision that produced a record.

    ``dirty`` is ``None`` when the state could not be read. It is never
    ``False`` for "unknown", because ``False`` means "the tree was clean" and a
    caller cannot tell that from a genuine clean reading.
    """

    commit: str
    branch: str | None = None
    dirty: bool | None = None
    read_error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.commit, str) or not self.commit.strip():
            raise ValueError("commit must be a non-empty string")

    @property
    def is_usable_for_reproduction(self) -> bool:
        """A SHA that could not be read cannot anchor a reproduction."""
        return self.commit.strip() != "unavailable" and self.read_error is None

    def as_dict(self) -> dict[str, str]:
        """The ``harness`` block shape used by the existing artifacts."""
        return {
            "commit": self.commit,
            "branch": self.branch or "unknown",
            "dirty": "unknown" if self.dirty is None else str(self.dirty).lower(),
        }


def source_identity(*, cwd: str | None = None) -> SourceIdentity:
    """Read the current source identity from git, never from the environment.

    Failures are reported through the record rather than raised: a harness that
    cannot read git should still be able to emit an artifact that says so.
    """

    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(  # noqa: S603
                ["git", *args],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
                cwd=cwd,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    commit = run("rev-parse", "HEAD")
    if commit is None:
        return SourceIdentity(
            commit="unavailable",
            dirty=None,
            read_error="git rev-parse HEAD did not succeed",
        )
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    status = run("status", "--porcelain")
    return SourceIdentity(
        commit=commit,
        branch=branch,
        # A status that could not be read is unknown, not clean.
        dirty=None if status is None else bool(status),
        read_error=None if status is not None else "git status did not succeed",
    )


@dataclass(slots=True, frozen=True)
class ModelIdentity:
    """What the serving path said it was serving.

    Every field is optional because a serving path may not disclose all of it.
    None means "not disclosed"; it never means "verified absent".
    """

    endpoint: str
    alias: str | None = None
    model_path: str | None = None
    model_sha256: str | None = None
    quantization: str | None = None
    backend: str | None = None
    total_slots: int | None = None
    n_ctx: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint, str) or not self.endpoint.strip():
            raise ValueError("endpoint must be a non-empty string")
        for name in ("total_slots", "n_ctx"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer when present")

    @classmethod
    def from_props(
        cls,
        *,
        endpoint: str,
        props: dict[str, object],
        weights_sha256: str | None = None,
    ) -> ModelIdentity:
        """Build from a ``/props``-shaped mapping, reading only what is present.

        The keys are the ones ``llama-server`` actually emits, confirmed against
        a live server rather than assumed: ``model_alias``, ``model_path``,
        ``model_ftype`` (which is where the quantization lives), ``total_slots``,
        ``default_generation_settings.n_ctx`` and ``build_info`` (the llama.cpp
        build, which is the backend identity). Reading ``quantization`` or
        ``backend`` instead would return ``None`` for every healthy server and
        report a fully identified model as unidentified.

        ``weights_sha256`` is **not** in ``/props`` — no weights digest is
        served — so it is passed in by the caller that hashed the file on disk
        (see ``scripts/numerical_backend_evidence.py``). Left unset it stays
        ``None`` and :meth:`ReproductionRecord.blockers` reports
        ``model_digest``; it is never invented.
        """

        def text(key: str) -> str | None:
            value = props.get(key)
            return value if isinstance(value, str) and value.strip() else None

        generation = props.get("default_generation_settings")
        generation = generation if isinstance(generation, dict) else {}
        n_ctx = generation.get("n_ctx")
        slots = props.get("total_slots")
        return cls(
            endpoint=endpoint,
            alias=text("model_alias"),
            model_path=text("model_path"),
            model_sha256=(
                weights_sha256
                if isinstance(weights_sha256, str) and weights_sha256.strip()
                else None
            ),
            quantization=text("model_ftype"),
            backend=text("build_info"),
            total_slots=(
                slots if isinstance(slots, int) and not isinstance(slots, bool) else None
            ),
            n_ctx=(
                n_ctx
                if isinstance(n_ctx, int) and not isinstance(n_ctx, bool) and n_ctx > 0
                else None
            ),
        )

    def with_weights_digest(self, digest: str) -> ModelIdentity:
        """Attach a locally computed weights digest to an identity read from props."""
        return replace(self, model_sha256=digest)


@dataclass(slots=True, frozen=True)
class SamplingPolicy:
    """Sampling configuration, and whether determinism was requested.

    In ``DETERMINISTIC`` mode a seed is mandatory and the temperature must be
    zero: a "deterministic" run with no seed is a run that merely looked
    deterministic once, and a run that claims determinism at a non-zero
    temperature is mislabelled.

    ``top_p`` is bounded to ``[0, 1]``, matching the request contract in
    :class:`oai2.runtime.inference.InferenceRequest` (``ge=0.0, le=1.0``), so
    a record cannot be built that the runtime would reject.
    """

    mode: DeterminismMode
    seed: int | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None

    def __post_init__(self) -> None:
        temperature = float(self.temperature)
        if not math.isfinite(temperature) or not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be a finite value in [0, 2]")
        top_p = float(self.top_p)
        if not math.isfinite(top_p) or not 0.0 <= top_p <= 1.0:
            raise ValueError("top_p must be a finite value in [0, 1]")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise ValueError("seed must be an integer when present")
        if self.top_k is not None and (
            isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 0
        ):
            raise ValueError("top_k must be a non-negative integer when present")
        if self.mode is DeterminismMode.DETERMINISTIC:
            if self.seed is None:
                raise ValueError("deterministic mode requires an explicit seed")
            if temperature != 0.0:
                raise ValueError("deterministic mode requires temperature == 0.0")

    @property
    def is_deterministic(self) -> bool:
        return self.mode is DeterminismMode.DETERMINISTIC


@dataclass(slots=True, frozen=True)
class EnvironmentSnapshot:
    """The host facts a reproduction must match (REQ-DET-023)."""

    platform: str
    machine: str
    python: str
    runtime_versions: tuple[tuple[str, str], ...] = ()
    tool_snapshot_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("platform", "machine", "python"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        for ref in self.tool_snapshot_refs:
            if not isinstance(ref, str) or not ref.strip():
                raise ValueError("tool_snapshot_refs entries must be non-empty strings")

    @classmethod
    def current(cls, *, runtime_versions: dict[str, str] | None = None) -> EnvironmentSnapshot:
        return cls(
            platform=platform.system(),
            machine=platform.machine(),
            python=sys.version.split()[0],
            runtime_versions=tuple(sorted((runtime_versions or {}).items())),
        )

    def without_versions(self) -> EnvironmentSnapshot:
        """Host facts only, for comparing two runs across patch-level changes."""
        return EnvironmentSnapshot(
            platform=self.platform, machine=self.machine, python=self.python
        )


@dataclass(slots=True, frozen=True)
class CorpusRef:
    """Which eval corpus produced the numbers.

    Carries the content-addressed revision from
    :mod:`oai2.evals.governance` so a record is tied to a revision rather than
    to a suite name that can change underneath it.
    """

    suite_id: str
    revision_id: str
    case_count: int
    scorer: str
    scorer_version: str

    def __post_init__(self) -> None:
        for name in ("suite_id", "revision_id", "scorer", "scorer_version"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if isinstance(self.case_count, bool) or not isinstance(self.case_count, int):
            raise ValueError("case_count must be an integer")
        if self.case_count < 0:
            raise ValueError("case_count must be >= 0")


@dataclass(slots=True, frozen=True)
class ReproductionRecord:
    """Everything needed to rebuild, or explain the failure to rebuild, a run."""

    label: str
    source: SourceIdentity
    model: ModelIdentity
    sampling: SamplingPolicy
    environment: EnvironmentSnapshot
    corpus: CorpusRef | None = None
    recorded_at: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")

    def blockers(self) -> tuple[str, ...]:
        """What is missing that would prevent an honest reproduction.

        A record with blockers may still be published, but it may not be cited
        as a reproducible measurement, and this is the list that says which
        fields to add.
        """
        out: list[str] = []
        if not self.source.is_usable_for_reproduction:
            out.append("source_identity")
        if self.source.dirty:
            out.append("source_tree_dirty")
        if self.model.model_sha256 is None:
            out.append("model_digest")
        if self.model.quantization is None:
            out.append("model_quantization")
        if self.model.backend is None:
            out.append("model_backend")
        if not self.sampling.is_deterministic:
            out.append("sampling_not_deterministic")
        if self.corpus is None:
            out.append("corpus_revision")
        if not self.environment.tool_snapshot_refs:
            out.append("tool_snapshot_refs")
        return tuple(out)


@dataclass(slots=True, frozen=True)
class Divergence:
    """One named difference between a record and a reproduction attempt."""

    dimension: DivergenceDimension
    detail: str


@dataclass(slots=True, frozen=True)
class ReproductionDiagnosis:
    """Outcome of comparing a recorded run against a reproduction attempt."""

    reproduced: bool
    divergences: tuple[Divergence, ...] = ()
    notes: str = ""

    @property
    def is_explained(self) -> bool:
        """Every difference is named, so the failure is diagnosed not just seen."""
        return all(div.detail.strip() for div in self.divergences)

    def dimensions(self) -> tuple[str, ...]:
        return tuple(div.dimension.value for div in self.divergences)

    def primary_dimension(self) -> str | None:
        """The first divergence, for a one-line summary.

        Order is SOURCE, MODEL, SAMPLING, ENVIRONMENT, CORPUS on purpose: the
        earliest dimension in that list is the cheapest to rule out, and a
        changed source revision explains most differences on its own.
        """
        if not self.divergences:
            return None
        priority = {dim: index for index, dim in enumerate(DivergenceDimension)}
        return min(self.divergences, key=lambda d: priority[d.dimension]).dimension.value


def _first_difference(left: Sequence[object], right: Sequence[object]) -> str:
    # strict=False on purpose: unequal lengths are themselves a divergence and
    # are reported by the fallback, not raised.
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return f"[{index}] {a!r} != {b!r}"
    return f"length {len(left)} != {len(right)}"


def diagnose(
    expected: ReproductionRecord, actual: ReproductionRecord
) -> ReproductionDiagnosis:
    """Compare a reproduction attempt against the recorded run (REQ-DET-024).

    Every differing dimension is reported with the specific values that
    differ, so the answer is "the seed changed" rather than "it did not
    reproduce". A reproduction that matches on every dimension is
    ``reproduced=True``; anything else names its axes.
    """
    divergences: list[Divergence] = []

    if expected.source.commit != actual.source.commit:
        divergences.append(
            Divergence(
                DivergenceDimension.SOURCE,
                f"commit {expected.source.commit[:12]} != {actual.source.commit[:12]}",
            )
        )
    if bool(expected.source.dirty) != bool(actual.source.dirty):
        divergences.append(
            Divergence(
                DivergenceDimension.SOURCE,
                f"dirty {expected.source.dirty} != {actual.source.dirty}",
            )
        )

    for name in (
        "endpoint",
        "model_path",
        "model_sha256",
        "quantization",
        "backend",
        "total_slots",
        "n_ctx",
    ):
        a = getattr(expected.model, name)
        b = getattr(actual.model, name)
        if a != b:
            divergences.append(Divergence(DivergenceDimension.MODEL, f"{name}: {a!r} != {b!r}"))

    if expected.sampling != actual.sampling:
        divergences.append(
            Divergence(
                DivergenceDimension.SAMPLING,
                f"mode={expected.sampling.mode.value}/{actual.sampling.mode.value} "
                f"seed={expected.sampling.seed}/{actual.sampling.seed} "
                f"temperature={expected.sampling.temperature}/{actual.sampling.temperature} "
                f"top_p={expected.sampling.top_p}/{actual.sampling.top_p}",
            )
        )

    left_env = expected.environment
    right_env = actual.environment
    if (left_env.platform, left_env.machine) != (right_env.platform, right_env.machine):
        divergences.append(
            Divergence(
                DivergenceDimension.ENVIRONMENT,
                f"host {left_env.platform}/{left_env.machine} != "
                f"{right_env.platform}/{right_env.machine}",
            )
        )
    if left_env.python != right_env.python:
        divergences.append(
            Divergence(
                DivergenceDimension.ENVIRONMENT,
                f"python {left_env.python} != {right_env.python}",
            )
        )
    if left_env.runtime_versions != right_env.runtime_versions:
        divergences.append(
            Divergence(
                DivergenceDimension.ENVIRONMENT,
                "runtime_versions "
                + _first_difference(
                    [f"{k}={v}" for k, v in left_env.runtime_versions],
                    [f"{k}={v}" for k, v in right_env.runtime_versions],
                ),
            )
        )
    if set(left_env.tool_snapshot_refs) != set(right_env.tool_snapshot_refs):
        divergences.append(
            Divergence(
                DivergenceDimension.ENVIRONMENT,
                f"tool_snapshot_refs {sorted(left_env.tool_snapshot_refs)} != "
                f"{sorted(right_env.tool_snapshot_refs)}",
            )
        )

    left_corpus = expected.corpus
    right_corpus = actual.corpus
    if (left_corpus is None) != (right_corpus is None) or (
        left_corpus is not None
        and right_corpus is not None
        and left_corpus != right_corpus
    ):
        divergences.append(
            Divergence(
                DivergenceDimension.CORPUS,
                f"{left_corpus} != {right_corpus}",
            )
        )

    return ReproductionDiagnosis(
        reproduced=not divergences, divergences=tuple(divergences)
    )


# ---------------------------------------------------------------------------
# What an artifact has to carry to be reproducible
# ---------------------------------------------------------------------------
#
# The field list lives here, next to the types that define the fields, rather
# than inside ``scripts/reproducibility_audit.py``, so "enough to reproduce"
# is a property of the library and is unit-tested instead of asserted by a
# script nobody runs.
#
# Each field declares several accepted probe paths because the committed
# artifacts were written by different harnesses and name the same fact
# differently (``harness.commit`` vs ``harness.source_sha``,
# ``serving_identity.weights_sha256`` vs ``model_sha256``). A single-key
# probe would score those as missing and report a real gap as a false alarm,
# which is the failure mode that gets a detector ignored. A field is present
# if *any* probe resolves to a non-null value, and the resolving path is
# recorded, so a reader can see which spelling satisfied it.


@dataclass(slots=True, frozen=True)
class ReproductionField:
    """One piece of metadata a record needs, and where it may be found."""

    name: str
    requirement: str
    probes: tuple[str, ...]

    def locate(self, document: object) -> str | None:
        """Return the first probe path that resolves, or ``None``.

        A probe containing ``[]`` means "some element of the list at this path
        carries this key", which is how suite-list artifacts are inspected.
        """
        for probe in self.probes:
            found = _probe(document, probe)
            if found is not None:
                return probe
        return None


def _probe(document: object, probe: str) -> object | None:
    if "[]" in probe:
        list_path, _, key = probe.partition("[].")
        if not key:
            return None
        current = _dig(document, list_path)
        if not isinstance(current, list):
            return None
        for element in current:
            if isinstance(element, dict) and element.get(key) is not None:
                return element[key]
        return None
    return _dig(document, probe)


def _dig(document: object, path: str) -> object | None:
    current: object = document
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        if part not in current:
            return None
        value = current[part]
        if value is None:
            return None
        current = value
    return current


REPRODUCTION_FIELDS: tuple[ReproductionField, ...] = (
    ReproductionField("source.commit", "REQ-DET-022",
                      ("harness.commit", "harness.source_sha", "source.commit")),
    ReproductionField("source.branch", "REQ-DET-022", ("harness.branch", "source.branch")),
    ReproductionField("source.dirty", "REQ-DET-022", ("harness.dirty", "source.dirty")),
    ReproductionField("model.endpoint", "REQ-DET-022",
                      ("serving_identity.base_url", "serving_identity.endpoint",
                       "model.endpoint", "result.backend")),
    ReproductionField("model.alias", "REQ-DET-022",
                      ("serving_identity.model_alias", "model.alias")),
    ReproductionField("model.path", "REQ-DET-022",
                      ("serving_identity.model_path", "model.model_path", "model.path")),
    ReproductionField("model.digest", "REQ-DET-022",
                      ("serving_identity.weights_sha256", "serving_identity.model_sha256",
                       "model.weights_sha256", "model.model_sha256")),
    ReproductionField("model.quantization", "REQ-DET-022",
                      ("serving_identity.model_ftype", "serving_identity.quantization",
                       "model.model_ftype", "model.quantization")),
    ReproductionField("model.backend", "REQ-DET-022",
                      ("serving_identity.build_info", "serving_identity.backend",
                       "model.build_info", "model.backend")),
    ReproductionField("model.total_slots", "REQ-DET-022",
                      ("serving_identity.total_slots", "model.total_slots")),
    ReproductionField("model.n_ctx", "REQ-DET-022",
                      ("serving_identity.n_ctx", "model.n_ctx")),
    ReproductionField("sampling.mode", "REQ-DET-021", ("sampling.mode",)),
    ReproductionField("sampling.seed", "REQ-DET-021",
                      ("sampling.seed", "parameters.seed")),
    ReproductionField("sampling.temperature", "REQ-DET-021",
                      ("sampling.temperature", "parameters.temperature")),
    ReproductionField("corpus.suite", "REQ-DET-022",
                      ("suite_id", "suite.suite_id", "workload")),
    ReproductionField("corpus.revision", "REQ-DET-022",
                      ("revision.revision_id", "suite.revision_id",
                       "suites[].revision_id")),
    ReproductionField("corpus.scorer", "REQ-DET-022",
                      ("scorer", "scorer_version", "suite.scorer")),
    ReproductionField("environment.host", "REQ-DET-023",
                      ("environment.platform", "host.platform", "machine",
                       "host_memory_start.platform")),
    ReproductionField("environment.python", "REQ-DET-023",
                      ("environment.python", "python_version", "host.python")),
    ReproductionField("environment.tool_snapshots", "REQ-DET-023",
                      ("environment.tool_snapshot_refs", "tool_snapshot_refs",
                       "tool_snapshots")),
)


@dataclass(slots=True, frozen=True)
class ArtifactCoverage:
    """Which reproduction fields one artifact carries.

    ``unreadable`` is a real state: an artifact that could not be parsed is
    reported as unreadable, never as a clean result, because "we could not
    look" and "there was nothing wrong" must never read the same way.
    """

    label: str
    present: tuple[tuple[str, str], ...] = ()
    missing: tuple[str, ...] = ()
    unreadable: str | None = None

    @property
    def is_unreadable(self) -> bool:
        return self.unreadable is not None

    @property
    def covered(self) -> int:
        return len(self.present)

    @property
    def coverage_ratio(self) -> float:
        """Fraction of required fields present, or ``0.0`` when unreadable."""
        if self.is_unreadable or not REPRODUCTION_FIELDS:
            return 0.0
        return len(self.present) / len(REPRODUCTION_FIELDS)

    def blockers(self) -> tuple[str, ...]:
        return self.missing


def inspect_artifact(label: str, document: object) -> ArtifactCoverage:
    """Measure one parsed artifact against :data:`REPRODUCTION_FIELDS`."""
    if not isinstance(document, dict):
        return ArtifactCoverage(
            label=label, unreadable=f"expected a JSON object, got {type(document).__name__}"
        )
    present: list[tuple[str, str]] = []
    missing: list[str] = []
    for spec in REPRODUCTION_FIELDS:
        path = spec.locate(document)
        if path is None:
            missing.append(spec.name)
        else:
            present.append((spec.name, path))
    return ArtifactCoverage(label=label, present=tuple(present), missing=tuple(missing))


@dataclass(slots=True, frozen=True)
class CoverageSummary:
    """Aggregate coverage across every artifact inspected."""

    coverages: tuple[ArtifactCoverage, ...] = ()

    @property
    def readable(self) -> tuple[ArtifactCoverage, ...]:
        return tuple(c for c in self.coverages if not c.is_unreadable)

    @property
    def unreadable(self) -> tuple[ArtifactCoverage, ...]:
        return tuple(c for c in self.coverages if c.is_unreadable)

    def per_field_presence(self) -> tuple[tuple[str, int], ...]:
        """How many readable artifacts carry each field, out of the readable total."""
        readable = self.readable
        return tuple(
            (spec.name, sum(1 for c in readable if spec.name not in c.missing))
            for spec in REPRODUCTION_FIELDS
        )

    def artifacts_missing(self, field_name: str) -> tuple[str, ...]:
        return tuple(c.label for c in self.readable if field_name in c.missing)

    def mean_coverage(self) -> float | None:
        """Mean coverage across readable artifacts, or ``None`` if none were read."""
        readable = self.readable
        if not readable:
            return None
        return sum(c.coverage_ratio for c in readable) / len(readable)

    def fully_reproducible(self) -> tuple[str, ...]:
        return tuple(c.label for c in self.readable if not c.missing)


def summarise(coverage: Sequence[ArtifactCoverage]) -> CoverageSummary:
    """Aggregate artifact coverage. Unreadable artifacts are counted, not dropped."""
    if not coverage:
        raise ValueError("no artifacts to summarise")
    return CoverageSummary(coverages=tuple(coverage))
