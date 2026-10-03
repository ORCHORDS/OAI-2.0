# Eval governance report — WI-BENCH-002 (#172)

`eval_governance_ac23ce2.json` is the raw artifact from
`scripts/eval_governance_report.py` run at source SHA
`ac23ce2c4f2d9c2215c10229c25657d904ea8751`, working tree clean
(`harness.dirty == "false"`), scorer `regex_or@2026-10-04`, corpus of 89
committed files from `evidence/`, `docs/` and `evals/`.

## What it reports

| suite | revision | cases | audit | findings |
|---|---|---:|---|---:|
| abstention | `fa704671d9df425e` | 1 | clean | 0 |
| bug_diagnosis | `18266f6cbf978de1` | 2 | clean | 0 |
| coding | `e23b089ab2e1b695` | 3 | clean | 0 |
| conflicting_evidence | `fc290d552700438f` | 1 | clean | 0 |
| **deterministic_hard** | `95b991d3b6dc7b3e` | 12 | **findings** | **12** |
| multi_file_reasoning | `cee898ded312df79` | 1 | clean | 0 |
| orchestration | `c07a285a46ef70b5` | 1 | clean | 0 |
| reasoning | `8c8e1a1da83bbb87` | 2 | clean | 0 |
| tool_use | `d345d7a18d1b0b37` | 3 | clean | 0 |
| verification | `1b9dc74c8b80328c` | 1 | clean | 0 |
| vision | `caefc130a20c5702` | 1 | clean | 0 |

**11 suites, 11 distinct revisions.** Two results are comparable only when
their revision, scorer, scorer version and candidate all match, and
`build_trend` groups rather than averages — so these eleven lines can never be
drawn as one curve.

**One suite cannot gate a promotion:** `deterministic_hard`, carrying the #171
contamination finding forward into the promotion path. That is REQ-BENCH-024
doing its job on real data rather than in a fixture.

## The gap this closes

`evaluate_held_out_promotion` — the existing gate — checks that baseline and
candidate cover the same `case_id` set and that none of those ids appear in
training inputs. Reproduced on `main` before `oai2.evals.governance` existed:

```
two SuiteReport objects, same case_id, completely different prompt behind it
  -> evaluate_held_out_promotion(...)  passed=True   failures=[]
```

The gate could not see the difference, because the prompt is not an input to
the comparison. A scorer change was equally invisible: `scorer` never reached
the function. `test_governed_gate_refuses_the_comparison_the_raw_gate_accepted`
pins exactly this, and asserts that the raw gate still accepts it — so the test
would notice if the underlying defect were ever fixed and the governed path
became redundant.

## Controls

- `test_unchanged_suite_yields_a_stable_revision` (determinism) and
  `test_changing_one_prompt_changes_the_revision` (sensitivity) are both
  required. A revision id that is constant makes nothing comparable; one that is
  unstable makes nothing comparable either.
- `test_absent_audit_blocks_promotion`: a record with no audit is `UNKNOWN`
  and blocks. Without `AuditStatus.UNKNOWN` the oldest, least-governed numbers
  would gate a promotion.
- `test_clean_and_matching_records_reach_the_numeric_gate` proves the governed
  path is a *pre-gate*, not a replacement: the numeric decision still comes
  from `evaluate_held_out_promotion`, unchanged.
- `test_different_revisions_do_not_merge_into_one_trend` asserts the absence
  of any blended mean, which is the thing REQ-BENCH-023 forbids.

## Reproducing

```sh
.venv/bin/python scripts/eval_governance_report.py
.venv/bin/python scripts/eval_governance_report.py --json out.json
```

Deterministic and local: no runner, no network, no model.
