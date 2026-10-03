# Reproducibility coverage audit — WI-DET-002 (#176)

`audit.json` is the raw artifact from `scripts/reproducibility_audit.py` run at
source SHA `0fc3bd61907faad1fea76924348363f505d33b93`, working tree clean
(`harness.dirty == "false"`), against all 7 JSON artifacts committed under
`evidence/` at that SHA.

## The finding

**Mean coverage 25.0%. 0 of 7 artifacts are fully reproducible.** This is a
measurement of this repository's own evidence, not a hypothetical gap.

| artifact | coverage | missing |
|---|---:|---:|
| `evidence/accuracy/cache_invalidation_probe.json` | 0.0% | 20 |
| `evidence/accuracy/deterministic_hard_12_a133744.json` | 60.0% | 8 |
| `evidence/admission/admission_probe_b7399ae.json` | 30.0% | 14 |
| `evidence/contamination/audit_4fb4112.json` | 15.0% | 17 |
| `evidence/governance/eval_governance_ac23ce2.json` | 20.0% | 16 |
| `evidence/numerical/backend_noran_58b5d1b0e429.json` | 40.0% | 12 |
| `evidence/numerical/tolerance_matrix_87d35007693d.json` | 10.0% | 18 |

**Seven fields are recorded by no artifact at all:**

| field | requirement | why it matters |
|---|---|---|
| `model.quantization` | REQ-DET-022 | the same path can hold different quants |
| `model.backend` | REQ-DET-022 | `/props` reports it as `build_info`; nobody copied it |
| `sampling.mode` | REQ-DET-021 | determinism is *implied* by `temperature: 0.0`, never declared |
| `corpus.revision` | REQ-DET-022 | every artifact predates #172's content-addressed revision |
| `environment.host` | REQ-DET-023 | no platform/machine is recorded anywhere |
| `environment.python` | REQ-DET-023 | no interpreter version is recorded anywhere |
| `environment.tool_snapshots` | REQ-DET-023 | no artifact references a `uv.lock` or any tool snapshot |

Two specific cases worth naming:

- **`cache_invalidation_probe.json` has no source identity whatsoever** — no
  commit, no branch, no dirty flag. Its verdict cannot be tied to any revision
  of the code that produced it, so a verifier replaying it would have no way to
  know what they were replaying.
- **`model.digest` is present in exactly 1 of 7** (`backend_noran`). That one
  artifact proves the digest is obtainable — it was computed by hashing the
  weights file on disk, not read from `/props`. The other six record the model
  *path* only, and a path is a claim, not a digest.

`model.quantization` and `model.backend` being absent everywhere is the direct
consequence of nobody copying the two `/props` keys that carry them.

## Why the probe list accepts several spellings

The committed artifacts were written by different harnesses and name the same
fact differently: `harness.commit` vs `harness.source_sha`,
`serving_identity.weights_sha256` vs `model_sha256`, `serving_identity.model_ftype`
vs `quantization`. A single-key probe would score all of those as missing and
report this real gap as a false alarm.

A detector that cries wolf is worse than no detector, because it gets trusted.
So each field declares every accepted path, a field is present if *any* probe
resolves to a non-null value, and the resolving path is recorded so a reader
can see which spelling satisfied it. The negative control is
`test_artifact_written_with_alternative_key_spellings_is_fully_covered`, which
scores a fully-populated artifact written entirely in the alternative spellings
at 100% — and asserts the specific paths it matched.

A `null` never satisfies a field (`test_a_null_value_does_not_satisfy_a_field`),
so an artifact cannot pass by carrying `"commit": null`.

## What this does not say

- It does not re-run any measurement, and it does not check that a recorded
  value is *true*. A field can be present and wrong.
- It does not say any artifact should be deleted. The point is that a partial
  record must not be cited as a reproducible one, and
  `ReproductionRecord.blockers()` is what enforces that.
- `corpus.revision` being absent everywhere is a *timing* fact: #172's
  revisions landed after these artifacts were written. It is not a defect in
  them.

## Reproducing

```sh
.venv/bin/python scripts/reproducibility_audit.py
.venv/bin/python scripts/reproducibility_audit.py --json out.json
.venv/bin/python scripts/reproducibility_audit.py --fail-on-gap   # for a gate
```

Deterministic and local: no runner, no network, no model. Exit status is 0 when
the audit ran, regardless of findings — a gap is a result to report, not a tool
failure.
