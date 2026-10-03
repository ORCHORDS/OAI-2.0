"""Incremental, source-derived repository file and symbol graph.

The graph is deliberately derived context, not persistent knowledge.  It uses
Python's standard-library AST parser when available and falls back to a
conservative import scan for malformed Python.  Unsupported languages remain
visible as file nodes but never receive invented symbols.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path


class ParserStatus(StrEnum):
    PARSED = "parsed"
    FALLBACK = "fallback"
    UNSUPPORTED = "unsupported"
    BINARY = "binary"


@dataclass(frozen=True)
class ImportSpec:
    """One syntactic import, retained even when it cannot resolve locally."""

    module: str
    names: tuple[str, ...] = ()
    level: int = 0
    kind: str = "import"


@dataclass(frozen=True)
class GraphSymbol:
    """A stable definition/import symbol within a source file."""

    symbol_id: str
    path: str
    name: str
    qualified_name: str
    kind: str
    line: int
    column: int


@dataclass(frozen=True)
class GraphReference:
    """A name read in source and any conservatively resolved local symbols."""

    path: str
    name: str
    line: int
    column: int
    target_symbol_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class FileNode:
    """A file and the parser output safely attributable to that file."""

    file_id: str
    path: str
    content_hash: str
    size_bytes: int
    language: str | None
    parser_status: ParserStatus
    parser_error: str | None = None
    symbols: tuple[GraphSymbol, ...] = ()
    imports: tuple[ImportSpec, ...] = ()
    references: tuple[GraphReference, ...] = ()


@dataclass(frozen=True)
class DependencyEdge:
    source: str
    target: str
    import_name: str
    kind: str


@dataclass(frozen=True)
class RepositoryIdentity:
    """Identity of the indexed source state, including dirty working-tree data."""

    root: str
    git_revision: str | None
    # Tri-state on purpose, matching ``git_revision``: ``None`` means the
    # working tree was never measured, which is NOT the same as clean. See
    # :func:`_git_metadata`.
    dirty: bool | None
    source_digest: str
    snapshot_id: str


@dataclass(frozen=True)
class IndexStats:
    parsed_files: int = 0
    reused_files: int = 0
    removed_files: int = 0
    fallback_files: int = 0
    unsupported_files: int = 0


@dataclass(frozen=True)
class RepositoryGraph:
    """Immutable graph snapshot suitable for checkpointing and comparison."""

    identity: RepositoryIdentity
    files: Mapping[str, FileNode]
    dependencies: tuple[DependencyEdge, ...]
    stats: IndexStats = field(default_factory=IndexStats)

    @classmethod
    def build(
        cls,
        root: str | os.PathLike[str],
        *,
        previous: RepositoryGraph | None = None,
        changed_paths: Iterable[str | os.PathLike[str]] | None = None,
    ) -> RepositoryGraph:
        return _build_graph(Path(root), previous=previous, changed_paths=changed_paths)

    @classmethod
    def update(
        cls,
        previous: RepositoryGraph,
        root: str | os.PathLike[str] | None = None,
        *,
        changed_paths: Iterable[str | os.PathLike[str]] | None = None,
    ) -> RepositoryGraph:
        return _build_graph(
            Path(root) if root is not None else Path(previous.identity.root),
            previous=previous,
            changed_paths=changed_paths,
        )

    @property
    def symbols(self) -> tuple[GraphSymbol, ...]:
        return tuple(symbol for node in self.files.values() for symbol in node.symbols)

    @property
    def references(self) -> tuple[GraphReference, ...]:
        return tuple(reference for node in self.files.values() for reference in node.references)

    @property
    def fingerprint(self) -> str:
        """Stable graph digest excluding filesystem location and parse counters."""
        payload = {
            "files": {path: _file_payload(node) for path, node in sorted(self.files.items())},
            "dependencies": [edge.__dict__ for edge in self.dependencies],
        }
        return _digest_json(payload)

    def definitions(self, name: str, *, path: str | None = None) -> tuple[GraphSymbol, ...]:
        """Locate definitions by simple or qualified name."""
        normalized = _normalize_query_path(path) if path else None
        matches = [
            symbol
            for symbol in self.symbols
            if (symbol.name == name or symbol.qualified_name == name)
            and (normalized is None or symbol.path == normalized)
        ]
        return tuple(
            sorted(matches, key=lambda symbol: (symbol.path, symbol.line, symbol.symbol_id))
        )

    def references_to(self, name_or_symbol_id: str) -> tuple[GraphReference, ...]:
        """Locate references by name or resolved stable symbol ID."""
        matches = []
        for reference in self.references:
            if (
                reference.name == name_or_symbol_id
                or name_or_symbol_id in reference.target_symbol_ids
            ):
                matches.append(reference)
        return tuple(
            sorted(
                matches, key=lambda reference: (reference.path, reference.line, reference.column)
            )
        )

    def dependencies_of(self, path: str, *, transitive: bool = False) -> tuple[str, ...]:
        """Return local modules imported by ``path``."""
        path = _normalize_query_path(path)
        direct = {edge.target for edge in self.dependencies if edge.source == path}
        if not transitive:
            return tuple(sorted(direct))
        return tuple(sorted(self._walk(direct, forward=True)))

    def dependents_of(self, path: str, *, transitive: bool = False) -> tuple[str, ...]:
        """Return local modules that depend on ``path``."""
        path = _normalize_query_path(path)
        direct = {edge.source for edge in self.dependencies if edge.target == path}
        if not transitive:
            return tuple(sorted(direct))
        return tuple(sorted(self._walk(direct, forward=False)))

    def affected_paths(self, changed_paths: Iterable[str]) -> tuple[str, ...]:
        """Return changed files plus all reverse dependency ancestors."""
        changed = {_normalize_query_path(path) for path in changed_paths}
        affected = set(changed)
        for path in changed:
            affected.update(self.dependents_of(path, transitive=True))
        return tuple(sorted(affected))

    def likely_affected_tests(self, changed_paths: Iterable[str]) -> tuple[str, ...]:
        """Return affected paths that look like tests, in deterministic order."""
        return tuple(path for path in self.affected_paths(changed_paths) if _looks_like_test(path))

    def canonical(self) -> dict[str, object]:
        """Return a JSON-serializable graph representation for evidence/checkpoints."""
        return {
            "identity": {
                "git_revision": self.identity.git_revision,
                "dirty": self.identity.dirty,
                "source_digest": self.identity.source_digest,
                "snapshot_id": self.identity.snapshot_id,
            },
            "files": {path: _file_payload(node) for path, node in sorted(self.files.items())},
            "dependencies": [edge.__dict__ for edge in self.dependencies],
        }

    def _walk(self, seeds: set[str], *, forward: bool) -> set[str]:
        seen = set(seeds)
        queue = deque(seeds)
        while queue:
            current = queue.popleft()
            neighbors = self.dependencies_of(current) if forward else self.dependents_of(current)
            for neighbor in neighbors:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        return seen


def build_repository_graph(
    root: str | os.PathLike[str],
    *,
    previous: RepositoryGraph | None = None,
    changed_paths: Iterable[str | os.PathLike[str]] | None = None,
) -> RepositoryGraph:
    """Build a graph, reusing unchanged parsed files from ``previous``."""
    return RepositoryGraph.build(root, previous=previous, changed_paths=changed_paths)


def update_repository_graph(
    previous: RepositoryGraph,
    root: str | os.PathLike[str] | None = None,
    *,
    changed_paths: Iterable[str | os.PathLike[str]] | None = None,
) -> RepositoryGraph:
    """Incrementally update a prior graph snapshot."""
    return RepositoryGraph.update(previous, root, changed_paths=changed_paths)


_IGNORED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        "dist",
        "build",
    }
)
_LANGUAGE_BY_SUFFIX = {".py": "python", ".pyi": "python"}
_IMPORT_FALLBACK_RE = re.compile(
    r"^\s*(?:from\s+(?P<from>[.\w]+)\s+import\s+|import\s+(?P<import>[.\w]+))"
)


def _build_graph(
    root: Path,
    *,
    previous: RepositoryGraph | None,
    changed_paths: Iterable[str | os.PathLike[str]] | None,
) -> RepositoryGraph:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    requested_changes = {_relative_path(root, Path(path)) for path in (changed_paths or ())}
    old_files = dict(previous.files) if previous is not None else {}
    current_files = _discover_files(root)
    files: dict[str, FileNode] = {}
    parsed = reused = removed = fallback = unsupported = 0

    for relative_path, absolute_path in current_files.items():
        raw = absolute_path.read_bytes()
        content_hash = hashlib.sha256(raw).hexdigest()
        old = old_files.get(relative_path)
        language = _LANGUAGE_BY_SUFFIX.get(absolute_path.suffix.lower())
        if old is not None and old.content_hash == content_hash and old.language == language:
            node = old
            reused += 1
        else:
            node = _parse_file(relative_path, raw, content_hash, language)
            parsed += 1
        files[relative_path] = node
        if node.parser_status is ParserStatus.FALLBACK:
            fallback += 1
        elif node.parser_status in (ParserStatus.UNSUPPORTED, ParserStatus.BINARY):
            unsupported += 1

    removed = len(set(old_files) - set(files))
    # ``requested_changes`` is intentionally not used to skip hash checks: a
    # caller may omit a path, and exact source-state identity must still win.
    del requested_changes

    dependencies = _resolve_dependencies(files)
    files = _resolve_references(files, dependencies)
    git_revision, dirty = _git_metadata(root)
    source_digest = _source_digest(files)
    snapshot_id = _digest_json(
        {
            "git_revision": git_revision,
            "dirty": dirty,
            "source_digest": source_digest,
        }
    )
    identity = RepositoryIdentity(
        root=str(root),
        git_revision=git_revision,
        dirty=dirty,
        source_digest=source_digest,
        snapshot_id=snapshot_id,
    )
    return RepositoryGraph(
        identity=identity,
        files=dict(sorted(files.items())),
        dependencies=tuple(
            sorted(
                dependencies,
                key=lambda edge: (edge.source, edge.target, edge.import_name, edge.kind),
            )
        ),
        stats=IndexStats(
            parsed_files=parsed,
            reused_files=reused,
            removed_files=removed,
            fallback_files=fallback,
            unsupported_files=unsupported,
        ),
    )


def _discover_files(root: Path) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(name for name in dirnames if name not in _IGNORED_DIRS)
        for filename in sorted(filenames):
            path = Path(current) / filename
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            discovered[relative] = path
    return dict(sorted(discovered.items()))


def _parse_file(
    relative_path: str,
    raw: bytes,
    content_hash: str,
    language: str | None,
) -> FileNode:
    file_id = f"file:{relative_path}"
    if language is None:
        return FileNode(
            file_id=file_id,
            path=relative_path,
            content_hash=content_hash,
            size_bytes=len(raw),
            language=None,
            parser_status=ParserStatus.UNSUPPORTED,
        )
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return FileNode(
            file_id=file_id,
            path=relative_path,
            content_hash=content_hash,
            size_bytes=len(raw),
            language=language,
            parser_status=ParserStatus.BINARY,
            parser_error=str(exc),
        )

    try:
        tree = ast.parse(source, filename=relative_path)
    except (SyntaxError, ValueError, TypeError) as exc:
        return FileNode(
            file_id=file_id,
            path=relative_path,
            content_hash=content_hash,
            size_bytes=len(raw),
            language=language,
            parser_status=ParserStatus.FALLBACK,
            parser_error=str(exc),
            imports=_scan_imports(source),
        )

    collector = _PythonCollector(relative_path)
    collector.visit(tree)
    return FileNode(
        file_id=file_id,
        path=relative_path,
        content_hash=content_hash,
        size_bytes=len(raw),
        language=language,
        parser_status=ParserStatus.PARSED,
        symbols=tuple(collector.symbols),
        imports=tuple(collector.imports),
        references=tuple(collector.references),
    )


class _PythonCollector(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.scope: list[str] = []
        self.symbols: list[GraphSymbol] = []
        self.imports: list[ImportSpec] = []
        self.references: list[GraphReference] = []

    def _add_symbol(self, node: ast.AST, name: str, kind: str) -> None:
        line = getattr(node, "lineno", 1)
        column = getattr(node, "col_offset", 0)
        qualified_name = ".".join((*self.scope, name))
        symbol_id = "symbol:" + _digest_json(
            {"path": self.path, "qualified_name": qualified_name, "kind": kind}
        )
        self.symbols.append(
            GraphSymbol(
                symbol_id=symbol_id,
                path=self.path,
                name=name,
                qualified_name=qualified_name,
                kind=kind,
                line=line,
                column=column,
            )
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._add_symbol(node, node.name, "function")
        self.scope.append(node.name)
        for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            self._add_symbol(argument, argument.arg, "parameter")
        if node.args.vararg:
            self._add_symbol(node.args.vararg, node.args.vararg.arg, "parameter")
        if node.args.kwarg:
            self._add_symbol(node.args.kwarg, node.args.kwarg.arg, "parameter")
        self.generic_visit(node)
        self.scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._add_symbol(node, node.name, "class")
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(ImportSpec(module=alias.name, names=(alias.name,), kind="import"))
            self._add_symbol(node, alias.asname or alias.name.split(".")[0], "import")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        names = tuple(alias.name for alias in node.names)
        self.imports.append(ImportSpec(module=module, names=names, level=node.level, kind="from"))
        for alias in node.names:
            if alias.name != "*":
                self._add_symbol(node, alias.asname or alias.name, "import")

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self._add_symbol(node, node.id, "variable")
        elif isinstance(node.ctx, ast.Load):
            self.references.append(
                GraphReference(
                    path=self.path,
                    name=node.id,
                    line=node.lineno,
                    column=node.col_offset,
                )
            )


def _scan_imports(source: str) -> tuple[ImportSpec, ...]:
    imports: list[ImportSpec] = []
    for line in source.splitlines():
        match = _IMPORT_FALLBACK_RE.match(line)
        if match is None:
            continue
        if match.group("from") is not None:
            raw = match.group("from")
            level = len(raw) - len(raw.lstrip("."))
            module = raw[level:]
            imports.append(ImportSpec(module=module, level=level, kind="from"))
        else:
            imports.append(ImportSpec(module=match.group("import"), kind="import"))
    return tuple(imports)


def _resolve_references(
    files: Mapping[str, FileNode], dependencies: Sequence[DependencyEdge]
) -> dict[str, FileNode]:
    by_name: dict[str, list[GraphSymbol]] = defaultdict(list)
    for node in files.values():
        for symbol in node.symbols:
            by_name[symbol.name].append(symbol)

    dependency_targets: dict[str, set[str]] = defaultdict(set)
    for edge in dependencies:
        dependency_targets[edge.source].add(edge.target)

    resolved: dict[str, FileNode] = {}
    for path, node in files.items():
        references: list[GraphReference] = []
        for reference in node.references:
            local_candidates = [symbol for symbol in by_name[reference.name] if symbol.path == path]
            non_imports = [symbol for symbol in local_candidates if symbol.kind != "import"]
            if non_imports:
                candidates = non_imports
            else:
                imported_targets = [
                    symbol
                    for target in dependency_targets[path]
                    for symbol in files[target].symbols
                    if symbol.name == reference.name and symbol.kind != "import"
                ]
                candidates = imported_targets or local_candidates or by_name[reference.name]
            target_ids = tuple(sorted({symbol.symbol_id for symbol in candidates}))
            references.append(replace(reference, target_symbol_ids=target_ids))
        resolved[path] = replace(node, references=tuple(references))
    return resolved


def _resolve_dependencies(files: Mapping[str, FileNode]) -> list[DependencyEdge]:
    module_index: dict[str, str] = {}
    for path in files:
        if not path.endswith((".py", ".pyi")):
            continue
        parts = path.rsplit("/", 1)
        filename = parts[-1]
        package = parts[0].split("/") if len(parts) == 2 else []
        if filename in {"__init__.py", "__init__.pyi"}:
            module = ".".join(package)
        else:
            module = ".".join((*package, filename.rsplit(".", 1)[0]))
        module_index[module] = path

    edges: set[DependencyEdge] = set()
    for source, node in files.items():
        for spec in node.imports:
            for target in _resolve_import_targets(source, spec, module_index):
                edges.add(
                    DependencyEdge(
                        source=source,
                        target=target,
                        import_name=_import_label(spec),
                        kind=spec.kind,
                    )
                )
    return list(edges)


def _resolve_import_targets(
    source: str,
    spec: ImportSpec,
    module_index: Mapping[str, str],
) -> set[str]:
    module_parts = source.rsplit("/", 1)
    filename = module_parts[-1]
    package_parts: tuple[str, ...] = (
        tuple(module_parts[0].split("/")) if len(module_parts) == 2 else ()
    )
    if filename.startswith("__init__."):
        package_parts = (*package_parts, "__init__")
        package_parts = package_parts[:-1]
    if spec.level:
        base_parts = package_parts[: max(0, len(package_parts) - spec.level + 1)]
        requested = (
            *base_parts,
            *([part for part in spec.module.split(".") if part] if spec.module else ()),
        )
    else:
        requested = tuple(part for part in spec.module.split(".") if part)
    base = ".".join(requested)
    candidates = [base] if base else []
    if spec.kind == "from":
        candidates.extend(f"{base}.{name}" for name in spec.names if name != "*")
    return {module_index[candidate] for candidate in candidates if candidate in module_index}


def _import_label(spec: ImportSpec) -> str:
    prefix = "." * spec.level
    module = f"{prefix}{spec.module}"
    if spec.kind == "from" and spec.names:
        return f"{module}:{','.join(spec.names)}"
    return module


def _source_digest(files: Mapping[str, FileNode]) -> str:
    entries = [(path, node.content_hash) for path, node in sorted(files.items())]
    return _digest_json(entries)


def _file_payload(node: FileNode) -> dict[str, object]:
    return {
        "file_id": node.file_id,
        "content_hash": node.content_hash,
        "size_bytes": node.size_bytes,
        "language": node.language,
        "parser_status": node.parser_status.value,
        "parser_error": node.parser_error,
        "symbols": [symbol.__dict__ for symbol in node.symbols],
        "imports": [import_spec.__dict__ for import_spec in node.imports],
        "references": [reference.__dict__ for reference in node.references],
    }


def _digest_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _relative_path(root: Path, path: Path) -> str:
    if path.is_absolute():
        return path.resolve().relative_to(root.resolve()).as_posix()
    return path.as_posix().lstrip("./")


def _normalize_query_path(path: str | os.PathLike[str]) -> str:
    return Path(path).as_posix().lstrip("./")


def _looks_like_test(path: str) -> bool:
    parts = Path(path).parts
    name = parts[-1]
    return "tests" in parts or name.startswith("test_") or name.endswith("_test.py")


def _git_metadata(root: Path) -> tuple[str | None, bool | None]:
    """Return ``(revision, dirty)`` for ``root``.

    ``dirty`` is a tri-state, matching ``revision`` in the same tuple:
    ``True``/``False`` when ``git status`` ran and answered, and ``None`` when
    it never could. A missing ``git`` binary, a corrupt index, a permissions
    problem or a dubious-ownership refusal all make the probe fail, and
    reporting ``False`` for those says "this working tree is clean" about a
    tree whose cleanliness was never established.

    That matters more than usual here because the value is hashed into
    ``snapshot_id``: an unmeasurable source state and a genuinely clean one
    produced the same identity input, so a snapshot could claim to describe a
    clean tree on the strength of a probe that failed.
    """
    try:
        revision_result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        status_result = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None, None
    revision = revision_result.stdout.strip() if revision_result.returncode == 0 else None
    # None, not False, when the probe could not run: see the docstring.
    dirty = bool(status_result.stdout.strip()) if status_result.returncode == 0 else None
    return revision, dirty
