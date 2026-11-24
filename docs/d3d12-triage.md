# D3D12.X crash-report triage

## Design

Chosen blocker: **6, crash-report triage for unknown D3D12.X slots/interfaces**.
The runtime already records called slot stubs in `report.json.called_stubs` and
unknown QueryInterface GUIDs in `d3d12.unknown_iids`. `stubs::record_call` logs the
first call, including its caller. Queue stubs include the older/newer layout in
their name. Today these observations must be extracted manually, and a missing
or truncated final JSON report makes its evidence harder to recover.

This is the highest-leverage bounded increment available without proprietary
data: it covers every title's unknown interface paths, needs no GPU assumptions,
and can be tested in CI. Wave64 and mesh already have substantial synthetic
coverage; full MSAA conversion requires broader layout and upload work. Runtime
report-persistence work is already proposed in upstream PR #240; this tool is
independent of it.

Add a dependency-free Python CLI that reads an existing report and optionally its
matching log. Group slot observations by DLL, interface, slot and layout suffix;
preserve distinct callers and source provenance. Unknown IIDs remain separate
observations: their appearance does not establish which interface or slot failed.
Use report counts when available; log first-call lines are evidence, not extra
calls. Sort output deterministically and support JSON for automated comparisons.

If the report cannot be read or decoded, warn and use the explicitly supplied
log. Reject an invocation with no usable input. Warn about malformed relevant
records rather than treating them as a clean run. Do not infer method signatures,
return values, crash causality or gameplay success. Never change runtime dispatch
or pretend an unknown slot succeeded. No executable, shader or game asset is read.

Expected impact: better preservation and grouping of unknown-slot evidence
reduces diagnosis time and the chance that a recurring boot blocker is overlooked.
It does not itself make an unsupported method execute. Synthetic tests will cover
both slot spellings, duplicate report/log evidence, IID normalization, deterministic
output and damaged-report fallback. Retail progression and future title-specific
interfaces remain unvalidated; a report can identify a candidate, not prove that
implementing it reaches gameplay.

## Reproduce and test

### Baseline comparison design (follow-up)

Compare exact slot identities (DLL, interface, number, variant) and canonical
IIDs against an explicitly supplied raw report. Preserve all current evidence;
report only newly observed identities. Count/caller changes and disappearance
of an observation say nothing about a newly missing or newly supported method.
Reject incomplete baselines. An opt-in CI gate returns 2 for incomplete current
evidence, 1 for new observations, and 0 otherwise. The caller must select
comparable title/version/backend runs; this is an evidence comparison, not proof
of a regression or matching run identity. Keep this increment separate from the
initial triage PR, with synthetic tests and no runtime changes.

From the repository root, with Python 3 (standard library only):

```powershell
python scripts/triage_d3d12.py path\report.json --log path\xcl.log
python scripts/triage_d3d12.py path\report.json --log path\xcl.log --json
python -B -m unittest discover -s scripts -p test_triage_d3d12.py -v
.\scripts\test.ps1
```

Use the report and log from **one run**; the tool cannot establish their identity.
`--log` is optional. Supply it explicitly to recover evidence when the report
path is missing or its JSON is damaged. Input files are read only; nothing is
uploaded. Output includes caller module names/offsets: review before sharing.

JSON schema version 1 has `slots`, `unknown_iids` and `warnings`. Slot `calls`
sums counts from report records, which may be an earlier snapshot; `null` means
log-only evidence with no known total. `first_callers` and `sources` are deduplicated.
`variant` preserves bare/null-backend, `not on PC`, older/newer layout and future
suffixes independently. Ordinary named stubs and other DLLs are excluded. IIDs
have no inferred call count, caller or slot association. Warnings also go to
stderr. Exit 0 means inputs were processed (even with warnings/findings), 2 means
no usable input or invalid CLI arguments. An empty result never proves support.

For a comparable earlier raw `report.json`, add `--baseline baseline.json`.
The optional `comparison` object lists `new_slots` and `new_iids` while keeping
all current evidence. `complete` means no recognized input/parser warnings;
it does not establish that either report is final or all title paths ran.
Disappeared observations are not called fixed. A malformed/incomplete baseline
produces exit 2 and a warning, retaining current evidence without a comparison.

```powershell
python scripts/triage_d3d12.py current.json --baseline baseline.json --json --fail-on-new
```

`--fail-on-new` requires `--baseline`. Its exit codes are 0 for no new identities,
1 for newly observed identities, and 2 for incomplete evidence or input errors.
Incomplete evidence takes precedence even if new identities were found. Without
the gate, a valid comparison returns 0 regardless of findings. Choose comparable
title versions, workloads, backends and adapters: different host interface
versions can change slot identities, and the tool cannot verify provenance.

## Validation and limits

On base `508a0c7` plus this scripts/docs/CI change: Windows 11 Home build 26300,
i9-14900K, RTX 4080 SUPER driver 32.0.16.2114, Rust 1.93.0, DXC 1.9.2609.5,
Python 3.13, debug build. No runtime source changed.

- New regression command above: **15 passed, 0 failed, 0 skipped**. Before the
  implementation, discovery failed because the triage module did not exist.
- Existing `scripts/test.ps1`: **680 passed, 0 failed, 41 ignored**. Within the
  passing count, two native-wave64 cases and two missing-debug-layer cases skipped;
  32 sample-gated tests returned without data (guard count, not an emitted total).
  All `XCL_SAMPLE_*` were unset. The synthetic console-title e2e test ran and passed.
- Streaming 100,000 identical synthetic first-call lines: **0.164 s**, one unique
  slot, unknown call total; 325-byte compact JSON. This is a local microbenchmark,
  not a runtime or gameplay speedup.

Regression coverage includes distinct interfaces/layouts, duplicate guarded
records, report/log overlap, IID normalization, malformed records/logs, stable
ordering, Unicode callers, missing/truncated/non-object reports, missing logs,
deeply nested JSON, and CLI/text-output behavior. Windows CI runs these without a GPU or game files.
The baseline follow-up adds 9 regressions (24 total), including exact identity
matching, count/caller independence, incomplete inputs, evidence retention and
all three gate exit codes. Risks: logger/schema drift may lose evidence
(recognized malformed records warn), and mixing runs can merge unrelated
observations. Follow-up: versioned structured runtime events with run identity.
No AMD/Intel GPU or retail run was performed. Future title interfaces,
shader behavior and boot-to-gameplay progression cannot be validated without an
authorized executable; none was requested or used here.
