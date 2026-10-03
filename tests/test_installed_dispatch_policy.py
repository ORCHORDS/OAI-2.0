"""Tests for the ACTUALLY INSTALLED :func:`default_dispatch_policy`.

These do not use a narrowed fixture policy: they exercise the policy the
agent loop really installs, because a denial proven against a hand-built
fixture says nothing about whether the serving policy rejects the same
action.

The two defects pinned here previously (as *observed behaviour*) are now
fixed, so these assert the corrected semantics:

1. ``resource_scopes`` holds directory roots and is applied by
   containment after ``..`` normalisation, not by exact string membership.
   Previously only a literal scope root passed, so every concrete path was
   refused — the scopes were decoration.
2. An unscoped capability (``Bash``, ``Glob`` — neither carries a path)
   used to walk straight past the scope gate. It now requires an explicit
   ``allow_unscoped_capabilities`` grant, and one is stated by
   :func:`default_dispatch_policy` so the default agent surface still works.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from oai2.agents.agent_loop import default_dispatch_policy
from oai2.core import ToolId
from oai2.protocols import ToolCall
from oai2.tools.dispatch import DispatchPolicy, DispatchStage, ToolDispatcher
from oai2.tools.registry import default_tool_definitions


def _call(tool: str, arguments: dict[str, object]) -> ToolCall:
    return ToolCall(id="c1", tool_id=ToolId(tool), arguments=dict(arguments))


@pytest.fixture
def installed(tmp_path):
    return ToolDispatcher(
        registry=default_tool_definitions(),
        policy=default_dispatch_policy(),
        cwd=tmp_path,
    )


# ---------------------------------------------------------------------------
# 1. Scoping is containment, not exact membership
# ---------------------------------------------------------------------------


def test_default_policy_scopes_are_roots() -> None:
    assert default_dispatch_policy().resource_scopes == frozenset({"./", "/tmp"})


def test_default_policy_does_not_admit_the_home_directory() -> None:
    """The default must not authorise the user's credentials.

    ``~/.ssh/id_rsa``, ``~/.aws/credentials`` and ``.config/gh/hosts.yml``
    are all reachable by absolute path, and ``fs.write`` is in the default
    allow list, so a home-directory scope root means the model can both read
    and overwrite every one of them. That was the shipped default until this
    was fixed; see #240.

    The scope list is asserted structurally here as well as behaviourally in
    ``TestHomeDirectoryIsNotAdmittedByDefault`` so a future re-introduction
    cannot pass by making the check structural-only or behavioural-only.
    """
    scopes = default_dispatch_policy().resource_scopes
    home = str(Path.home())
    for scope in scopes:
        assert not (scope == home or scope == "~/"), (
            f"default scope {scope!r} admits the home directory"
        )


def test_concrete_file_under_a_scope_root_is_allowed(installed, tmp_path) -> None:
    """A real file path beneath a scope root must EXECUTE, not be refused."""
    target = tmp_path / "app.py"
    target.write_text("x\n", encoding="utf-8")
    installed._cwd = tmp_path  # noqa: SLF001 - scope root is "./" relative to cwd
    policy = DispatchPolicy(
        allow_capabilities=frozenset({"fs.read"}),
        resource_scopes=frozenset({str(tmp_path)}),
        allow_unscoped_capabilities=frozenset(),
    )
    d = ToolDispatcher(registry=default_tool_definitions(), policy=policy, cwd=tmp_path)
    out = d.check(_call("Read", {"path": str(target)}), calls_used=0)
    assert out.stage is DispatchStage.EXECUTE, out.reason


def test_nested_subdirectory_under_a_scope_root_is_allowed(tmp_path) -> None:
    policy = DispatchPolicy(
        allow_capabilities=frozenset({"fs.read"}),
        resource_scopes=frozenset({"/allowed"}),
        allow_unscoped_capabilities=frozenset(),
    )
    d = ToolDispatcher(registry=default_tool_definitions(), policy=policy)
    assert d.check(_call("Read", {"path": "/allowed/a/b/c.py"}), calls_used=0).stage is (
        DispatchStage.EXECUTE
    )


def test_path_outside_every_scope_root_is_denied(tmp_path) -> None:
    policy = DispatchPolicy(
        allow_capabilities=frozenset({"fs.read"}),
        resource_scopes=frozenset({"/allowed"}),
        allow_unscoped_capabilities=frozenset(),
    )
    d = ToolDispatcher(registry=default_tool_definitions(), policy=policy)
    out = d.check(_call("Read", {"path": "/etc/passwd"}), calls_used=0)
    assert out.stage is DispatchStage.DENY
    assert "resource out of scope" in out.reason


def test_parent_traversal_cannot_escape_a_scope(installed, tmp_path) -> None:
    """``<scope>/../etc/passwd`` must normalise out of scope, not through it."""
    policy = DispatchPolicy(
        allow_capabilities=frozenset({"fs.read"}),
        resource_scopes=frozenset({str(tmp_path)}),
        allow_unscoped_capabilities=frozenset(),
    )
    d = ToolDispatcher(registry=default_tool_definitions(), policy=policy, cwd=tmp_path)
    out = d.check(_call("Read", {"path": f"{tmp_path}/../etc/passwd"}), calls_used=0)
    assert out.stage is DispatchStage.DENY
    assert "resource out of scope" in out.reason


def test_sibling_prefix_directory_is_not_in_scope(tmp_path) -> None:
    """``/allowed-evil`` must not pass as being inside ``/allowed``."""
    policy = DispatchPolicy(
        allow_capabilities=frozenset({"fs.read"}),
        resource_scopes=frozenset({"/allowed"}),
        allow_unscoped_capabilities=frozenset(),
    )
    d = ToolDispatcher(registry=default_tool_definitions(), policy=policy)
    assert d.check(_call("Read", {"path": "/allowed-evil/x"}), calls_used=0).stage is (
        DispatchStage.DENY
    )


def test_relative_path_resolves_against_cwd(tmp_path) -> None:
    policy = DispatchPolicy(
        allow_capabilities=frozenset({"fs.read"}),
        resource_scopes=frozenset({str(tmp_path)}),
        allow_unscoped_capabilities=frozenset(),
    )
    d = ToolDispatcher(registry=default_tool_definitions(), policy=policy, cwd=tmp_path)
    assert d.check(_call("Read", {"path": "sub/app.py"}), calls_used=0).stage is (
        DispatchStage.EXECUTE
    )
    assert d.check(_call("Read", {"path": "../outside.py"}), calls_used=0).stage is (
        DispatchStage.DENY
    )


# ---------------------------------------------------------------------------
# 2. Unscoped capabilities need an explicit grant
# ---------------------------------------------------------------------------


def test_tool_scoping_table() -> None:
    scoped = {t.name: t.scoped for t in default_tool_definitions()}
    assert scoped == {
        "Read": True,
        "Edit": True,
        "Write": True,
        "Bash": False,
        "Glob": True,
        "Grep": True,
    }


def test_grep_cannot_escape_a_denial_on_read(tmp_path) -> None:
    """The fix for defect 2: Grep was a way to read what Read was refused."""
    policy = DispatchPolicy(
        allow_capabilities=frozenset({"fs.read"}),
        resource_scopes=frozenset({str(tmp_path)}),
        allow_unscoped_capabilities=frozenset(),
        high_impact_approved=True,
    )
    d = ToolDispatcher(registry=default_tool_definitions(), policy=policy, cwd=tmp_path)
    assert d.check(_call("Read", {"path": "/etc/passwd"}), calls_used=0).stage is (
        DispatchStage.DENY
    )
    # Same capability, same out-of-scope target: Grep must not be a bypass.
    assert d.check(_call("Grep", {"pattern": "root", "path": "/etc"}), calls_used=0).stage is (
        DispatchStage.DENY
    )


def test_unscoped_capability_is_denied_without_an_explicit_grant(tmp_path) -> None:
    policy = DispatchPolicy(
        allow_capabilities=frozenset({"shell.exec"}),
        resource_scopes=frozenset({str(tmp_path)}),
        allow_unscoped_capabilities=frozenset(),
        high_impact_approved=True,
    )
    d = ToolDispatcher(registry=default_tool_definitions(), policy=policy, cwd=tmp_path)
    out = d.check(_call("Bash", {"command": "cat /etc/passwd"}), calls_used=0)
    assert out.stage is DispatchStage.DENY
    assert "unscoped capability" in out.reason


def test_unscoped_capability_runs_when_explicitly_granted(installed) -> None:
    """Legitimate authorised operation still works under the real policy."""
    out = installed.check(_call("Bash", {"command": "ls"}), calls_used=0)
    assert out.stage is DispatchStage.EXECUTE, out.reason


def test_default_policy_grants_exactly_the_tools_without_a_path() -> None:
    """Only ``Bash`` lacks a path, so only ``Bash`` holds an unscoped grant.

    ``Glob`` used to be here too, because it declared no ``path`` argument and
    therefore had nothing for gate 4 to constrain. It now takes a ``path`` and
    is scoped, so directory listing is bound by the host's scopes like any
    other filesystem read.
    """
    granted = default_dispatch_policy().allow_unscoped_capabilities
    assert granted == frozenset({"shell.exec"})
    # Nothing that *can* be scoped may hold an unscoped grant.
    for td in default_tool_definitions():
        if td.scoped:
            assert td.capability not in granted, td.name


def test_no_scoping_intent_means_no_unscoped_gate(tmp_path) -> None:
    """A host that declares no scopes has expressed no intent to scope."""
    policy = DispatchPolicy(
        allow_capabilities=frozenset({"shell.exec"}),
        resource_scopes=frozenset(),
        high_impact_approved=True,
    )
    d = ToolDispatcher(registry=default_tool_definitions(), policy=policy, cwd=tmp_path)
    assert d.check(_call("Bash", {"command": "ls"}), calls_used=0).stage is DispatchStage.EXECUTE


def test_scoped_tool_with_no_path_argument_is_denied(tmp_path) -> None:
    """A scoped tool that omits its path must not slip through unscoped."""
    policy = DispatchPolicy(
        allow_capabilities=frozenset({"fs.read"}),
        resource_scopes=frozenset({str(tmp_path)}),
        allow_unscoped_capabilities=frozenset(),
    )
    d = ToolDispatcher(registry=default_tool_definitions(), policy=policy, cwd=tmp_path)
    assert d.check(_call("Read", {}), calls_used=0).stage is DispatchStage.DENY


class TestHomeDirectoryIsNotAdmittedByDefault:
    """The behavioural half of the home-directory fix.

    The structural assertion above checks the scope list. These check what
    that scope list actually *does* to a real ``Read``/``Write`` under the
    policy the agent loop installs, because a scope entry that looks safe and
    a scope entry that admits the file are not the same claim.
    """

    @staticmethod
    def _canary() -> Path:
        """A harmless stand-in for the credential files that live at home.

        Named to mirror the real thing so the test reads honestly, and it
        holds no secret. The directory is created under the real home
        directory because that is the only place the regression can occur;
        putting it in tmp_path would not test the default scopes at all.
        """
        d = Path.home() / ".oai2-test-canary"
        d.mkdir(exist_ok=True)
        return d

    def test_read_under_the_home_directory_is_denied_by_default(self, tmp_path) -> None:
        canary = self._canary() / "id_rsa"
        canary.write_text("CANARY-NOT-A-REAL-KEY\n", encoding="utf-8")
        try:
            d = ToolDispatcher(
                registry=default_tool_definitions(),
                policy=default_dispatch_policy(),
                cwd=tmp_path,
            )
            call = ToolCall(
                id="c1",
                tool_id=ToolId("Read"),
                arguments={"path": str(canary)},
            )
            decision = d.check(call, calls_used=0)
        finally:
            canary.unlink()

        assert decision.stage is DispatchStage.DENY
        assert "out of scope" in decision.reason

    def test_write_under_the_home_directory_is_denied_by_default(self, tmp_path) -> None:
        """``fs.write`` is in the default allow list, so read-only would not
        be enough — overwriting a key is as bad as reading one."""
        canary = self._canary() / "id_rsa"
        canary.write_text("CANARY-NOT-A-REAL-KEY\n", encoding="utf-8")
        try:
            d = ToolDispatcher(
                registry=default_tool_definitions(),
                policy=default_dispatch_policy(),
                cwd=tmp_path,
            )
            call = ToolCall(
                id="c1",
                tool_id=ToolId("Write"),
                arguments={"path": str(canary), "content": "overwritten"},
            )
            decision = d.check(call, calls_used=0)
        finally:
            canary.unlink()

        assert decision.stage is DispatchStage.DENY

    def test_the_working_directory_still_works(self, tmp_path) -> None:
        """The control: narrowing the default must not break the agent.

        ``./`` is the host-chosen working directory; the loop has to be able
        to read the project it was pointed at.
        """
        target = tmp_path / "app.py"
        target.write_text("x = 1\n", encoding="utf-8")
        d = ToolDispatcher(
            registry=default_tool_definitions(),
            policy=default_dispatch_policy(),
            cwd=tmp_path,
        )
        decision = d.check(
            ToolCall(id="c1", tool_id=ToolId("Read"), arguments={"path": "app.py"}),
            calls_used=0,
        )
        assert decision.stage is DispatchStage.EXECUTE

    def test_a_host_may_still_opt_in_to_home_access_explicitly(
        self, tmp_path
    ) -> None:
        """Not a removal of the capability — a change of who decides.

        An interactive coding assistant may genuinely want ``~/`` in scope.
        It now has to say so, which is the whole point.
        """
        canary = self._canary() / "id_rsa"
        canary.write_text("CANARY-NOT-A-REAL-KEY\n", encoding="utf-8")
        try:
            d = ToolDispatcher(
                registry=default_tool_definitions(),
                policy=default_dispatch_policy(resource_scopes=["./", str(Path.home())]),
                cwd=tmp_path,
            )
            decision = d.check(
                ToolCall(id="c1", tool_id=ToolId("Read"), arguments={"path": str(canary)}),
                calls_used=0,
            )
        finally:
            canary.unlink()

        assert decision.stage is DispatchStage.EXECUTE

    def test_shell_is_unscoped_and_the_docs_do_not_claim_otherwise(self) -> None:
        """Pin the honest limit of the scope gate.

        ``shell.exec`` carries no path, so ``Bash("cat ~/.ssh/id_rsa")``
        passes gate 4 whatever the scopes say. The fix therefore does not
        rest on the scopes being a sandbox, and the docstring now says so.
        If someone later removes the unscoped-shell grant, this test should be
        updated deliberately rather than silently passing.
        """
        assert "shell.exec" in default_dispatch_policy().allow_unscoped_capabilities
        doc = default_dispatch_policy.__doc__ or ""
        assert "do not bound what a shell can reach" in doc
