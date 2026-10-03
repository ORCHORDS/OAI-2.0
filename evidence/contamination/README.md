# Contamination audit — WI-BENCH-001 (#171)

`audit_4fb4112.json` is the raw artifact from `scripts/contamination_audit.py`
run against this repository at source SHA
`4fb4112c9241493879af05358e8a52accd9e3207`, working tree clean
(`harness.dirty == "false"`).

## The finding

**All 12 cases of the `deterministic_hard` held-out suite appear verbatim in a
committed repository file.**

| suite | cases | findings | quarantined |
|---|---:|---:|---|
| abstention | 1 | 0 | — |
| bug_diagnosis | 2 | 0 | — |
| coding | 3 | 0 | — |
| conflicting_evidence | 1 | 0 | — |
| **deterministic_hard** | **12** | **12** | **all 12** |
| multi_file_reasoning | 1 | 0 | — |
| orchestration | 1 | 0 | — |
| reasoning | 2 | 0 | — |
| tool_use | 1 | 0 | — |
| verification | 1 | 0 | — |
| vision | 1 | 0 | — |

Every finding is `near_prompt` at score 1.000 against
`evidence/accuracy/deterministic_hard_12_a133744.json`.

The other 26 cases across ten suites report clean. That is the negative
control: the audit is not simply reporting contamination everywhere.

## What this means

`evidence/accuracy/deterministic_hard_12_a133744.json` is a real measurement
artifact from #240, committed deliberately as evidence. It contains each
held-out prompt and the model's reply, and it also contains the suite's
`expected_patterns`. So:

- the suite is not held out from this repository, and therefore not from
  anything built out of this repository;
- the 4/12 figure recorded for that suite can be reached by memorisation
  without any reasoning;
- any future promotion gate that treats `deterministic_hard` as held-out is
  measuring recall of a committed file.

The artifact is **not** deleted. It is the evidence behind a published
accuracy number, and destroying it would break #240's record to hide a
governance problem. The correct consequence is that the suite is retired as a
held-out gate, which is an owner decision recorded on #171 rather than one to
take unilaterally here.

### Why the verifier-leak check did not fire on the answer keys

The file does contain the expected patterns, but the check deliberately does
not flag them: the suite's oracles are mostly single tokens (`\b7\b`,
`\b372\b`), and `min_verifier_tokens=3` refuses to accuse a corpus on a
one-token signature, because such a pattern matches ordinary prose everywhere.
`test_short_oracle_cannot_accuse_a_corpus` pins that behaviour. The prompt
contamination alone already condemns the file, so no signal is lost here.

## The detector was wrong first

The first version of the near-overlap score was Jaccard similarity. Run
against the real tree it reported **0 findings for all 12 cases** — a verbatim
copy scored as clean.

The cause is direction: Jaccard divides by the union, so a 13-token case
inside a 7 KB artifact is diluted toward zero. Measured on the real file:

| score | result on all 12 cases |
|---|---|
| Jaccard over trigram digests | 0 findings |
| containment of case shingles in entry | 1.000, all 12 |

The unrelated control scored 0.000 under containment, so the change cost no
precision. The switch is documented in `_overlap`, and
`test_audit_detects_a_planted_verbatim_copy` plus
`test_real_repository_artifact_is_flagged` pin both the detection and the real
finding.

## Shingle size, chosen from measurement

| k | lightly-edited copy | unrelated | same-topic-different |
|---|---:|---:|---:|
| 2 | 0.88 | 0.00 | 0.00 |
| 3 | 0.88 | 0.00 | 0.00 |
| 5 | 0.86 | 0.00 | 0.00 |

Three is the default because it is the smallest k that still separates the
unrelated controls. The honest limit is pinned by
`test_heavy_rewrite_of_a_short_prompt_is_not_detected`: a *rewritten* short
prompt is not detected, and a threshold low enough to catch it would also flag
the controls.

## Reproducing

```sh
.venv/bin/python scripts/contamination_audit.py
.venv/bin/python scripts/contamination_audit.py --json out.json --fail-on-findings
```

The corpus is enumerated with `git ls-files` over `evidence/`, `docs/` and
`evals/`, so the verdict covers exactly what is committed and cannot change
because a local `.venv` exists.
