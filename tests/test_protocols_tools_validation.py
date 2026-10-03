"""Structural / validation tests for ``oai2.protocols.tools``.

Pin the tool-calling wire shapes: the :class:`ToolArgument`,
:class:`ToolDefinition`, :class:`ToolPolicy`, :class:`ToolCall`, and
:class:`ToolResult` Pydantic v2 :class:`BaseModel` constraints
(``extra="forbid"``, length bounds, capability enum, frozen-ness,
budget bounds), plus the :meth:`ToolDefinition.signature` stable-string
helper.

Behavioural end-to-end coverage lives in ``tests/test_protocols.py``;
this file pins the *shape* of the API and the invariants the policy
pipeline relies on.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from oai2.core import ToolId
from oai2.protocols import (
    ToolArgument,
    ToolCall,
    ToolDefinition,
    ToolPolicy,
    ToolResult,
)
from oai2.protocols.tools import (
    ToolArgument as ToolArgumentFromModule,
)
from oai2.protocols.tools import (
    ToolCall as ToolCallFromModule,
)
from oai2.protocols.tools import (
    ToolDefinition as ToolDefinitionFromModule,
)
from oai2.protocols.tools import (
    ToolPolicy as ToolPolicyFromModule,
)
from oai2.protocols.tools import (
    ToolResult as ToolResultFromModule,
)

_MODULE_PATH = Path(__file__).resolve().parent.parent / "oai2" / "protocols" / "tools.py"
_MODULE_SOURCE = _MODULE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Module docstring + import surface
# ---------------------------------------------------------------------------


def test_module_source_is_non_empty() -> None:
    """Sanity: the source file was read successfully."""

    assert _MODULE_SOURCE, "protocols/tools.py is unexpectedly empty"


def test_module_has_docstring() -> None:
    """The module ships an overview of the wire shapes."""

    assert _MODULE_SOURCE.startswith('"""')
    first_para = _MODULE_SOURCE.split('"""', 2)[1]
    assert first_para, "module docstring is empty"


def test_module_docstring_mentions_tool_definition_and_tool_call() -> None:
    """The overview mentions ``ToolDefinition`` and ``ToolCall``."""

    first_para = _MODULE_SOURCE.split('"""', 2)[1]
    assert "ToolDefinition" in first_para
    assert "ToolCall" in first_para


def test_module_uses_future_annotations() -> None:
    """``from __future__ import annotations`` is present (UP006-clean)."""

    assert "from __future__ import annotations" in _MODULE_SOURCE


def test_module_imports_use_relative_tool_id() -> None:
    """``ToolId`` is imported relatively from ``..core``; no absolute ``from oai2`` imports."""

    assert "from ..core import ToolId" in _MODULE_SOURCE
    # No accidental absolute import of core
    assert not re.search(r"^import oai2\b", _MODULE_SOURCE, re.MULTILINE)
    assert not re.search(r"^from oai2\.core\b", _MODULE_SOURCE, re.MULTILINE)
    # No accidental absolute import of protocols at all
    assert not re.search(r"^from oai2\b", _MODULE_SOURCE, re.MULTILINE)


def test_module_imports_pydantic_typing() -> None:
    """The module pulls the Pydantic symbols it needs (BaseModel,
    ConfigDict, Field, field_validator)."""

    for symbol in ("BaseModel", "ConfigDict", "Field", "field_validator"):
        assert symbol in _MODULE_SOURCE, f"missing Pydantic import: {symbol}"


def test_module_has_no_wildcard_imports() -> None:
    """No ``from X import *`` — the public surface is enumerated in ``__all__``."""

    assert "import *" not in _MODULE_SOURCE


# ---------------------------------------------------------------------------
# __all__ completeness + package re-export
# ---------------------------------------------------------------------------


def test_dunder_all_lists_exactly_five_public_names() -> None:
    """The module's public surface is exactly 5 names, no more, no less."""

    import oai2.protocols.tools as mod

    assert isinstance(mod.__all__, list)
    assert set(mod.__all__) == {
        "ToolArgument",
        "ToolDefinition",
        "ToolPolicy",
        "ToolCall",
        "ToolResult",
    }
    assert len(mod.__all__) == 5


def test_each_all_name_is_importable_from_module() -> None:
    """Every name in ``__all__`` resolves to a real attribute on the module."""

    import oai2.protocols.tools as mod

    for name in mod.__all__:
        assert hasattr(mod, name), f"__all__ name missing: {name}"


def test_public_names_are_reexported_from_protocols_package() -> None:
    """Top-level ``oai2.protocols`` re-exports the 5 tool public names."""

    import oai2.protocols as pkg

    for name in (
        "ToolArgument",
        "ToolDefinition",
        "ToolPolicy",
        "ToolCall",
        "ToolResult",
    ):
        assert name in pkg.__all__, f"package re-export missing: {name}"


def test_package_re_export_preserves_identity() -> None:
    """Package re-exports point at the *same* classes as the module."""

    import oai2.protocols as pkg
    import oai2.protocols.tools as mod

    assert pkg.ToolArgument is mod.ToolArgument
    assert pkg.ToolDefinition is mod.ToolDefinition
    assert pkg.ToolPolicy is mod.ToolPolicy
    assert pkg.ToolCall is mod.ToolCall
    assert pkg.ToolResult is mod.ToolResult


def test_module_imports_match_top_level() -> None:
    """``from oai2.protocols.tools import X`` matches the top-level re-export."""

    assert ToolArgument is ToolArgumentFromModule
    assert ToolDefinition is ToolDefinitionFromModule
    assert ToolPolicy is ToolPolicyFromModule
    assert ToolCall is ToolCallFromModule
    assert ToolResult is ToolResultFromModule


# ---------------------------------------------------------------------------
# ToolArgument
# ---------------------------------------------------------------------------


def test_tool_argument_is_a_pydantic_basemodel() -> None:
    """``ToolArgument`` subclasses :class:`pydantic.BaseModel`."""

    assert issubclass(ToolArgument, BaseModel)


def test_tool_argument_rejects_extra_fields() -> None:
    """``extra="forbid"`` — unknown fields raise ``ValidationError``."""

    with pytest.raises(ValidationError):
        ToolArgument(name="x", type="string", unknown_field="oops")  # type: ignore[call-arg]


def test_tool_argument_is_frozen() -> None:
    """``ToolArgument`` is frozen — assignment raises ``ValidationError``."""

    arg = ToolArgument(name="path", type="path", value="/tmp/foo")
    with pytest.raises(ValidationError):
        arg.value = "/tmp/bar"


def test_tool_argument_name_min_length() -> None:
    """``name`` ``min_length=1`` — empty raises ``ValidationError``."""

    with pytest.raises(ValidationError):
        ToolArgument(name="", type="string")


def test_tool_argument_name_max_length() -> None:
    """``name`` ``max_length=128`` — overlong raises ``ValidationError``."""

    with pytest.raises(ValidationError):
        ToolArgument(name="x" * 129, type="string")


def test_tool_argument_name_accepts_max_length_boundary() -> None:
    """A name of exactly 128 chars is accepted (boundary inclusive)."""

    arg = ToolArgument(name="x" * 128, type="string")
    assert len(arg.name) == 128


def test_tool_argument_type_min_length() -> None:
    """``type`` ``min_length=1`` — empty raises ``ValidationError``."""

    with pytest.raises(ValidationError):
        ToolArgument(name="x", type="")


def test_tool_argument_type_max_length() -> None:
    """``type`` ``max_length=64`` — overlong raises ``ValidationError``."""

    with pytest.raises(ValidationError):
        ToolArgument(name="x", type="y" * 65)


def test_tool_argument_type_accepts_known_types() -> None:
    """The ``type`` validator accepts the documented set:

    ``{"string", "integer", "number", "boolean", "array", "object",
    "null", "uri", "path", "binary"}``."""

    for t in (
        "string",
        "integer",
        "number",
        "boolean",
        "array",
        "object",
        "null",
        "uri",
        "path",
        "binary",
    ):
        arg = ToolArgument(name="x", type=t)
        assert arg.type == t


def test_tool_argument_type_rejects_unknown_values() -> None:
    """The ``type`` validator rejects unknown values."""

    with pytest.raises(ValidationError):
        ToolArgument(name="x", type="decimal-foo")  # not in the allowed set

    with pytest.raises(ValidationError):
        ToolArgument(name="x", type="string-extra")  # close-but-no-cigar

    with pytest.raises(ValidationError):
        ToolArgument(name="x", type="STRING")  # case-sensitive


def test_tool_argument_value_default_is_none() -> None:
    """``value`` defaults to ``None`` when omitted."""

    arg = ToolArgument(name="x", type="string")
    assert arg.value is None


def test_tool_argument_value_accepts_any_type() -> None:
    """``value`` is annotated ``Any`` — it accepts arbitrary Python values."""

    for v in ("hello", 42, 3.14, True, None, {"k": 1}, [1, 2, 3], b"bytes"):
        arg = ToolArgument(name="x", type="string", value=v)
        assert arg.value == v


def test_tool_argument_fields() -> None:
    """``ToolArgument`` has exactly ``(name, type, value)`` fields."""

    assert set(ToolArgument.model_fields.keys()) == {"name", "type", "value"}


# ---------------------------------------------------------------------------
# ToolDefinition
# ---------------------------------------------------------------------------


def test_tool_definition_is_a_pydantic_basemodel() -> None:
    """``ToolDefinition`` subclasses :class:`pydantic.BaseModel`."""

    assert issubclass(ToolDefinition, BaseModel)


def test_tool_definition_rejects_extra_fields() -> None:
    """``extra="forbid"`` — unknown fields raise ``ValidationError``."""

    with pytest.raises(ValidationError):
        ToolDefinition(
            id=ToolId("read"),
            description="read a file",
            capability="fs.read",
            unknown_field="oops",  # type: ignore[call-arg]
        )


def test_tool_definition_is_not_frozen() -> None:
    """``ToolDefinition`` is NOT frozen — attribute assignment succeeds."""

    td = ToolDefinition(
        id=ToolId("read"), name="read", description="read a file", capability="fs.read"
    )
    # Reassignment is permitted (no ``frozen=True`` on ``model_config``).
    td.scoped = True
    assert td.scoped is True


def test_tool_definition_id_accepts_tool_id() -> None:
    """``id`` accepts a ``ToolId``-typed value."""

    td = ToolDefinition(
        id=ToolId("read"), name="read", description="read a file", capability="fs.read"
    )
    assert td.id == "read"


def test_tool_definition_id_accepts_bare_string() -> None:
    """``ToolId = NewType("ToolId", str)`` is a static-only hint; runtime is plain ``str``."""

    td = ToolDefinition(id="read", name="read", description="read a file", capability="fs.read")  # type: ignore[arg-type]
    assert td.id == "read"


def test_tool_definition_name_min_length() -> None:
    """``name`` ``min_length=1`` — empty raises ``ValidationError``."""

    with pytest.raises(ValidationError):
        ToolDefinition(id=ToolId("read"), name="", description="x", capability="fs.read")


def test_tool_definition_name_max_length() -> None:
    """``name`` ``max_length=64`` — overlong raises ``ValidationError``."""

    with pytest.raises(ValidationError):
        ToolDefinition(
            id=ToolId("read"),
            name="x" * 65,
            description="x",
            capability="fs.read",
        )


def test_tool_definition_description_min_length() -> None:
    """``description`` ``min_length=1`` — empty raises ``ValidationError``."""

    with pytest.raises(ValidationError):
        ToolDefinition(id=ToolId("read"), name="read", description="", capability="fs.read")


def test_tool_definition_description_max_length() -> None:
    """``description`` ``max_length=4096`` — overlong raises ``ValidationError``."""

    with pytest.raises(ValidationError):
        ToolDefinition(
            id=ToolId("read"),
            name="read",
            description="x" * 4097,
            capability="fs.read",
        )


def test_tool_definition_arguments_default_is_empty_tuple() -> None:
    """A fresh ``ToolDefinition`` has no arguments."""

    td = ToolDefinition(
        id=ToolId("read"), name="read", description="read a file", capability="fs.read"
    )
    assert td.arguments == ()


def test_tool_definition_arguments_default_is_an_empty_tuple_type() -> None:
    """The default ``arguments`` is an empty ``tuple`` (the singleton is immutable,
    so identity sharing across instances is safe)."""

    td = ToolDefinition(id=ToolId("read"), name="read", description="x", capability="fs.read")
    assert isinstance(td.arguments, tuple)
    assert len(td.arguments) == 0


def test_tool_definition_arguments_accept_tool_arguments() -> None:
    """``arguments`` accepts a tuple of ``ToolArgument`` instances."""

    td = ToolDefinition(
        id=ToolId("read"),
        name="read",
        description="x",
        capability="fs.read",
        arguments=(ToolArgument(name="path", type="path"),),
    )
    assert len(td.arguments) == 1
    assert td.arguments[0].name == "path"


def test_tool_definition_capability_min_length() -> None:
    """``capability`` ``min_length=1`` — empty raises ``ValidationError``."""

    with pytest.raises(ValidationError):
        ToolDefinition(id=ToolId("read"), name="read", description="x", capability="")


def test_tool_definition_capability_max_length() -> None:
    """``capability`` ``max_length=128`` — overlong raises ``ValidationError``."""

    with pytest.raises(ValidationError):
        ToolDefinition(
            id=ToolId("read"),
            name="read",
            description="x",
            capability="x" * 129,
        )


def test_tool_definition_scoped_default_is_false() -> None:
    """``scoped`` defaults to ``False``."""

    td = ToolDefinition(id=ToolId("read"), name="read", description="x", capability="fs.read")
    assert td.scoped is False


def test_tool_definition_high_impact_default_is_false() -> None:
    """``high_impact`` defaults to ``False``."""

    td = ToolDefinition(id=ToolId("read"), name="read", description="x", capability="fs.read")
    assert td.high_impact is False


def test_tool_definition_signature_is_stable_for_empty_arguments() -> None:
    """``signature()`` produces ``name()`` when there are no arguments."""

    td = ToolDefinition(id=ToolId("noop"), name="noop", description="noop", capability="x.noop")
    assert td.signature() == "noop()"


def test_tool_definition_signature_lists_arguments_in_order() -> None:
    """``signature()`` lists arguments as ``name:type`` joined by ``, ``."""

    td = ToolDefinition(
        id=ToolId("read"),
        name="read",
        description="read a file",
        capability="fs.read",
        arguments=(
            ToolArgument(name="path", type="path"),
            ToolArgument(name="limit", type="integer"),
        ),
    )
    assert td.signature() == "read(path:path, limit:integer)"


def test_tool_definition_fields() -> None:
    """``ToolDefinition`` has exactly the documented field names."""

    assert set(ToolDefinition.model_fields.keys()) == {
        "id",
        "name",
        "description",
        "arguments",
        "capability",
        "scoped",
        "high_impact",
    }


# ---------------------------------------------------------------------------
# ToolPolicy
# ---------------------------------------------------------------------------


def test_tool_policy_is_a_pydantic_basemodel() -> None:
    """``ToolPolicy`` subclasses :class:`pydantic.BaseModel`."""

    assert issubclass(ToolPolicy, BaseModel)


def test_tool_policy_rejects_extra_fields() -> None:
    """``extra="forbid"`` — unknown fields raise ``ValidationError``."""

    with pytest.raises(ValidationError):
        ToolPolicy(unknown_field="oops")  # type: ignore[call-arg]


def test_tool_policy_is_frozen() -> None:
    """``ToolPolicy`` is frozen — assignment raises ``ValidationError``."""

    policy = ToolPolicy()
    with pytest.raises(ValidationError):
        policy.high_impact_approved = True


def test_tool_policy_default_allow_capabilities_is_empty_frozenset() -> None:
    """``allow_capabilities`` defaults to an empty ``frozenset``."""

    policy = ToolPolicy()
    assert policy.allow_capabilities == frozenset()
    assert isinstance(policy.allow_capabilities, frozenset)


def test_tool_policy_default_deny_capabilities_is_empty_frozenset() -> None:
    """``deny_capabilities`` defaults to an empty ``frozenset``."""

    policy = ToolPolicy()
    assert policy.deny_capabilities == frozenset()
    assert isinstance(policy.deny_capabilities, frozenset)


def test_tool_policy_default_factories_are_independent() -> None:
    """Two fresh ``ToolPolicy`` instances must not share the same frozensets."""

    a = ToolPolicy()
    b = ToolPolicy()
    # frozensets are immutable but the default_factory should produce
    # independent instances so that constructing with overrides does not
    # leak across instances.
    assert a.allow_capabilities is not b.allow_capabilities
    assert a.deny_capabilities is not b.deny_capabilities


def test_tool_policy_default_budget_calls_is_64() -> None:
    """``budget_calls`` defaults to 64."""

    policy = ToolPolicy()
    assert policy.budget_calls == 64


def test_tool_policy_default_budget_seconds_is_60() -> None:
    """``budget_seconds`` defaults to 60.0."""

    policy = ToolPolicy()
    assert policy.budget_seconds == 60.0


def test_tool_policy_default_high_impact_approved_is_false() -> None:
    """``high_impact_approved`` defaults to ``False``."""

    policy = ToolPolicy()
    assert policy.high_impact_approved is False


def test_tool_policy_budget_calls_rejects_negative() -> None:
    """``budget_calls`` uses ``ge=0`` — negatives raise ``ValidationError``."""

    with pytest.raises(ValidationError):
        ToolPolicy(budget_calls=-1)


def test_tool_policy_budget_calls_accepts_zero() -> None:
    """``budget_calls=0`` is the documented boundary (ge=0)."""

    policy = ToolPolicy(budget_calls=0)
    assert policy.budget_calls == 0


def test_tool_policy_budget_calls_rejects_over_ceiling() -> None:
    """``budget_calls`` uses ``le=100_000`` — over the ceiling raises ``ValidationError``."""

    with pytest.raises(ValidationError):
        ToolPolicy(budget_calls=100_001)


def test_tool_policy_budget_calls_accepts_max_boundary() -> None:
    """``budget_calls=100_000`` is accepted (le=100_000 boundary inclusive)."""

    policy = ToolPolicy(budget_calls=100_000)
    assert policy.budget_calls == 100_000


def test_tool_policy_budget_seconds_rejects_negative() -> None:
    """``budget_seconds`` uses ``ge=0.0`` — negatives raise ``ValidationError``."""

    with pytest.raises(ValidationError):
        ToolPolicy(budget_seconds=-1.0)


def test_tool_policy_budget_seconds_rejects_over_ceiling() -> None:
    """``budget_seconds`` uses ``le=86_400.0`` — over the ceiling raises ``ValidationError``."""

    with pytest.raises(ValidationError):
        ToolPolicy(budget_seconds=86_401.0)


def test_tool_policy_budget_seconds_accepts_max_boundary() -> None:
    """``budget_seconds=86_400.0`` is accepted (le=86_400.0 boundary inclusive)."""

    policy = ToolPolicy(budget_seconds=86_400.0)
    assert policy.budget_seconds == 86_400.0


def test_tool_policy_accepts_capability_frozensets() -> None:
    """``allow_capabilities`` / ``deny_capabilities`` accept arbitrary frozensets of strings."""

    policy = ToolPolicy(
        allow_capabilities=frozenset({"fs.read", "net.fetch"}),
        deny_capabilities=frozenset({"shell.exec"}),
    )
    assert "fs.read" in policy.allow_capabilities
    assert "shell.exec" in policy.deny_capabilities


def test_tool_policy_fields() -> None:
    """``ToolPolicy`` has exactly the documented field names."""

    assert set(ToolPolicy.model_fields.keys()) == {
        "allow_capabilities",
        "deny_capabilities",
        "budget_calls",
        "budget_seconds",
        "high_impact_approved",
    }


# ---------------------------------------------------------------------------
# ToolCall
# ---------------------------------------------------------------------------


def test_tool_call_is_a_pydantic_basemodel() -> None:
    """``ToolCall`` subclasses :class:`pydantic.BaseModel`."""

    assert issubclass(ToolCall, BaseModel)


def test_tool_call_rejects_extra_fields() -> None:
    """``extra="forbid"`` — unknown fields raise ``ValidationError``."""

    with pytest.raises(ValidationError):
        ToolCall(
            id="c1",
            tool_id=ToolId("read"),
            unknown_field="oops",  # type: ignore[call-arg]
        )


def test_tool_call_is_not_frozen() -> None:
    """``ToolCall`` is NOT frozen — attribute assignment succeeds."""

    tc = ToolCall(id="c1", tool_id=ToolId("read"))
    tc.arguments = {"path": "/x"}
    assert tc.arguments == {"path": "/x"}


def test_tool_call_id_min_length() -> None:
    """``id`` ``min_length=1`` — empty raises ``ValidationError``."""

    with pytest.raises(ValidationError):
        ToolCall(id="", tool_id=ToolId("read"))


def test_tool_call_id_max_length() -> None:
    """``id`` ``max_length=128`` — overlong raises ``ValidationError``."""

    with pytest.raises(ValidationError):
        ToolCall(id="x" * 129, tool_id=ToolId("read"))


def test_tool_call_tool_id_accepts_tool_id() -> None:
    """``tool_id`` accepts a ``ToolId``-typed value."""

    tc = ToolCall(id="c1", tool_id=ToolId("read"))
    assert tc.tool_id == "read"


def test_tool_call_tool_id_accepts_bare_string() -> None:
    """``ToolId = NewType("ToolId", str)`` is a static-only hint; runtime is plain ``str``."""

    tc = ToolCall(id="c1", tool_id="read")  # type: ignore[arg-type]
    assert tc.tool_id == "read"


def test_tool_call_arguments_default_is_empty_dict() -> None:
    """``arguments`` defaults to an empty dict."""

    tc = ToolCall(id="c1", tool_id=ToolId("read"))
    assert tc.arguments == {}
    assert isinstance(tc.arguments, dict)


def test_tool_call_arguments_default_factory_is_independent() -> None:
    """Two fresh ``ToolCall`` instances must not share the same dict instance."""

    a = ToolCall(id="c1", tool_id=ToolId("read"))
    b = ToolCall(id="c1", tool_id=ToolId("read"))
    assert a.arguments is not b.arguments
    a.arguments["k"] = "v"
    assert "k" not in b.arguments


def test_tool_call_policy_default_is_tool_policy() -> None:
    """``policy`` defaults to a fresh ``ToolPolicy`` instance."""

    tc = ToolCall(id="c1", tool_id=ToolId("read"))
    assert isinstance(tc.policy, ToolPolicy)
    assert tc.policy.budget_calls == 64
    assert tc.policy.budget_seconds == 60.0
    assert tc.policy.high_impact_approved is False


def test_tool_call_fields() -> None:
    """``ToolCall`` has exactly the documented field names."""

    assert set(ToolCall.model_fields.keys()) == {
        "id",
        "tool_id",
        "arguments",
        "policy",
    }


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------


def test_tool_result_is_a_pydantic_basemodel() -> None:
    """``ToolResult`` subclasses :class:`pydantic.BaseModel`."""

    assert issubclass(ToolResult, BaseModel)


def test_tool_result_rejects_extra_fields() -> None:
    """``extra="forbid"`` — unknown fields raise ``ValidationError``."""

    with pytest.raises(ValidationError):
        ToolResult(call_id="c1", ok=True, unknown_field="oops")  # type: ignore[call-arg]


def test_tool_result_is_not_frozen() -> None:
    """``ToolResult`` is NOT frozen — attribute assignment succeeds."""

    tr = ToolResult(call_id="c1", ok=True)
    tr.error = "boom"
    assert tr.error == "boom"


def test_tool_result_call_id_is_required() -> None:
    """``call_id`` has no default — omitting it raises ``ValidationError``."""

    with pytest.raises(ValidationError):
        ToolResult(ok=True)  # type: ignore[call-arg]


def test_tool_result_ok_is_required() -> None:
    """``ok`` has no default — omitting it raises ``ValidationError``."""

    with pytest.raises(ValidationError):
        ToolResult(call_id="c1")  # type: ignore[call-arg]


def test_tool_result_output_default_is_none() -> None:
    """``output`` defaults to ``None``."""

    tr = ToolResult(call_id="c1", ok=True)
    assert tr.output is None


def test_tool_result_output_accepts_any_type() -> None:
    """``output`` is annotated ``Any`` — arbitrary values accepted."""

    for v in ("hello", 42, {"k": 1}, [1, 2, 3], None, b"bytes"):
        tr = ToolResult(call_id="c1", ok=True, output=v)
        assert tr.output == v


def test_tool_result_error_default_is_none() -> None:
    """``error`` defaults to ``None``."""

    tr = ToolResult(call_id="c1", ok=True)
    assert tr.error is None


def test_tool_result_error_accepts_string() -> None:
    """``error`` accepts a non-empty string."""

    tr = ToolResult(call_id="c1", ok=False, error="timeout")
    assert tr.error == "timeout"


def test_tool_result_elapsed_ms_default_is_zero() -> None:
    """``elapsed_ms`` defaults to 0.0."""

    tr = ToolResult(call_id="c1", ok=True)
    assert tr.elapsed_ms == 0.0


def test_tool_result_elapsed_ms_rejects_negative() -> None:
    """``elapsed_ms`` uses ``ge=0.0`` — negatives raise ``ValidationError``."""

    with pytest.raises(ValidationError):
        ToolResult(call_id="c1", ok=True, elapsed_ms=-0.1)


def test_tool_result_elapsed_ms_accepts_zero() -> None:
    """``elapsed_ms=0.0`` is the documented boundary (ge=0.0 inclusive)."""

    tr = ToolResult(call_id="c1", ok=True, elapsed_ms=0.0)
    assert tr.elapsed_ms == 0.0


def test_tool_result_fields() -> None:
    """``ToolResult`` has exactly the documented field names."""

    assert set(ToolResult.model_fields.keys()) == {
        "call_id",
        "ok",
        "output",
        "error",
        "elapsed_ms",
    }


# ---------------------------------------------------------------------------
# Module source — public-safety boundary
# ---------------------------------------------------------------------------


def test_module_source_has_no_cloud_sdk_reference() -> None:
    """The wire-shape module must not pull any cloud SDK or vendor helper."""

    forbidden = (
        "boto3",
        "azure",
        "google.cloud",
        "gcp",
        "aws_access_key",
        "kubernetes",
        "docker",
    )
    for needle in forbidden:
        assert needle not in _MODULE_SOURCE, f"forbidden cloud reference: {needle}"


def test_module_source_has_no_hardcoded_api_key() -> None:
    """No long alphanumeric secret is hardcoded in the module source."""

    code_only = "\n".join(
        line for line in _MODULE_SOURCE.splitlines() if not line.lstrip().startswith("#")
    )
    assert not re.search(r"api_key\s*=\s*[\"']sk-[A-Za-z0-9]{16,}", code_only)
    assert not re.search(r"[\"']sk-[A-Za-z0-9]{16,}[\"']", code_only)
    assert "BEGIN PRIVATE KEY" not in code_only


def test_module_source_has_no_print_or_pprint() -> None:
    """The module is silent — no ``print`` / ``pprint``."""

    assert "print(" not in _MODULE_SOURCE
    assert "pprint(" not in _MODULE_SOURCE


def test_module_source_has_no_subprocess_or_os_system() -> None:
    """No subprocess / os.system — the module is pure-Python schemas."""

    assert "subprocess" not in _MODULE_SOURCE
    assert "os.system" not in _MODULE_SOURCE


def test_module_source_has_no_direct_network_imports() -> None:
    """No direct ``requests`` / ``urllib`` / ``httpx`` imports."""

    assert "import requests" not in _MODULE_SOURCE
    assert "from urllib" not in _MODULE_SOURCE
    assert "import urllib" not in _MODULE_SOURCE
    assert "import httpx" not in _MODULE_SOURCE
    assert "from httpx" not in _MODULE_SOURCE


def test_module_source_has_no_eval_or_exec() -> None:
    """No ``eval`` / ``exec`` — the schemas are static."""

    assert "eval(" not in _MODULE_SOURCE
    assert "exec(" not in _MODULE_SOURCE


def test_module_source_has_no_wildcard_imports_in_source() -> None:
    """No ``from X import *`` — the public surface is enumerated in ``__all__``."""

    assert "import *" not in _MODULE_SOURCE


def test_module_source_does_not_read_environment_directly() -> None:
    """The module does not read ``os.environ`` — config is constructor-injected."""

    assert "os.environ" not in _MODULE_SOURCE
    assert "os.getenv" not in _MODULE_SOURCE


def test_module_source_has_no_outstanding_todo_markers() -> None:
    """No outstanding ``TODO`` / ``FIXME`` / ``XXX`` — the file is shipped
    as production code."""

    for marker in (r"\bTODO\b", r"\bFIXME\b", r"\bXXX\b"):
        assert not re.search(marker, _MODULE_SOURCE), f"forbidden marker in source: {marker}"


def test_module_source_uses_extra_forbid_on_all_five_models() -> None:
    """All five Pydantic models declare ``extra="forbid"``."""

    assert _MODULE_SOURCE.count('extra="forbid"') == 5


def test_module_source_uses_frozen_config_on_arg_and_policy() -> None:
    """``ToolArgument`` and ``ToolPolicy`` are the two frozen models."""

    # ``frozen=True`` appears exactly twice in the module: on ToolArgument and
    # on ToolPolicy. ``ToolDefinition`` / ``ToolCall`` / ``ToolResult`` are
    # NOT frozen (so the dispatch layer can mutate their ``arguments`` dict).
    assert _MODULE_SOURCE.count("frozen=True") == 2


def test_module_source_uses_pydantic_basemodel() -> None:
    """All five models subclass :class:`BaseModel`."""

    assert _MODULE_SOURCE.count("(BaseModel)") == 5


def test_module_source_defines_field_validator_for_argument_type() -> None:
    """The ``ToolArgument.type`` field is guarded by a ``field_validator``."""

    assert "field_validator" in _MODULE_SOURCE
    assert '@field_validator("type")' in _MODULE_SOURCE


def test_module_source_exposes_signature_method() -> None:
    """``ToolDefinition`` exposes a stable string ``signature()`` method used
    for caches and prompt hashing."""

    assert "def signature" in _MODULE_SOURCE
    # The f-string format is ``f"{self.name}({', '.join(parts)})"`` — the
    # literal ``self.name`` is followed by the closing f-string brace ``}``,
    # then the literal opening paren of the signature.
    assert "self.name}" in _MODULE_SOURCE


def test_module_source_does_not_use_unittest_mock() -> None:
    """Production code does not import a test-time helper."""

    assert "unittest.mock" not in _MODULE_SOURCE
    assert "import unittest" not in _MODULE_SOURCE


class TestToolResultErrorMatchesOk:
    """`ok` and `error` must agree, as they already do on the knowledge transport.

    `knowledge/transport.py` enforces "successful response must not contain
    error" and "failed response requires error" on its sibling response
    shape. `ToolResult` enforced neither, so both states a caller most needs
    to tell apart were constructible -- and `_format_tool_result`, the only
    renderer, branches on `result.ok` alone:

        ToolResult(call_id="c1", ok=True, output="", error="exit 1")
          accepted; the model is shown '' and the error is dropped, so a
          FAILED call arrives as an empty SUCCESS

        ToolResult(call_id="c1", ok=False, error=None)
          accepted; the model is shown 'ERROR: unknown failure' -- a failure
          whose cause was never measured

    The second is an absent measurement rendered as a definite one; the first
    is a leak rendered as success.
    """

    def test_a_successful_result_may_not_carry_an_error(self) -> None:
        with pytest.raises(ValidationError, match="must not contain error"):
            ToolResult(call_id="c1", ok=True, output="", error="exit 1")

    def test_a_failed_result_requires_an_error(self) -> None:
        with pytest.raises(ValidationError, match="requires error"):
            ToolResult(call_id="c1", ok=False, error=None)

    def test_opposite_direction_a_well_formed_success_is_accepted(self) -> None:
        """Guard: the common case must not be made harder to construct."""
        tr = ToolResult(call_id="c1", ok=True, output="file contents")
        assert tr.ok is True
        assert tr.error is None

    def test_opposite_direction_a_well_formed_failure_is_accepted(self) -> None:
        tr = ToolResult(call_id="c1", ok=False, error="timeout")
        assert tr.ok is False
        assert tr.error == "timeout"

    def test_opposite_direction_a_failure_may_still_carry_partial_output(self) -> None:
        """Guard: `output` is deliberately NOT constrained on the failure path.

        A tool that produced partial output before failing has something real
        to report, and forbidding it would push producers toward dropping the
        output or inventing a fake success to carry it. The knowledge
        transport forbids the analogous combination because there is no
        meaningful partial object; here there is.
        """
        tr = ToolResult(call_id="c1", ok=False, output="wrote 3 of 10 files", error="disk full")
        assert tr.ok is False
        assert tr.output == "wrote 3 of 10 files"

    def test_the_model_cannot_be_shown_a_failure_as_a_success(self) -> None:
        """End-to-end: the rendering the agent loop performs.

        Asserted on the constructed states, not on a stub, because the
        defect was that the bad state existed at all -- a renderer fix alone
        would leave it constructible by the next producer.
        """
        from oai2.agents.agent_loop import _format_tool_result

        assert _format_tool_result(
            ToolResult(call_id="c1", ok=True, output="file contents")
        ) == "file contents"
        assert _format_tool_result(
            ToolResult(call_id="c1", ok=False, error="timeout")
        ) == "ERROR: timeout"
        # And the two states that used to be reachable cannot be built at all.
        for kwargs in (
            {"ok": True, "output": "", "error": "exit 1"},
            {"ok": False, "error": None},
        ):
            with pytest.raises(ValidationError):
                ToolResult(call_id="c1", **kwargs)  # type: ignore[arg-type]
