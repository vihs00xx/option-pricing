# Frozen 1D comparison milestone

The shared protocol, cases, raw predictions and timings are under
`results/shared_comparison/v1/`. All four protocol-pinned neural checkpoints
(anchored PINN and three refined parity seeds) are retained with their original
paths and SHA-256 identities. Source bytes are preserved by `.gitattributes`.
The model registries retain the earlier selection decisions; no model was
selected using the shared test errors.

Run `python -m experiments.comparison.verify_milestone` before using the saved
results. It checks the protocol, source, model, prediction and timing digests.
Run `python -m experiments.comparison.summarize_shared` to rebuild the summaries.
`requirements-benchmark.txt` pins the principal runtime and test dependencies;
the protocol and per-worker environment records identify the actual software
and hardware. Exact training replay is not promised: saved checkpoints are the
reproducible evaluation inputs, and historical training used different budgets.

## Cleanup and recovery

`cleanup_manifest.json` identifies files removed in the first cleanup, their original paths and
SHA-256. A verified pre-cleanup recovery archive is stored locally beside the
project at `../option-pricing-archive/milestone-20260926/before_cleanup.zip`.
It includes prior drafts, legacy experiments and all pre-cleanup checkpoints.
The same verified archive also contains the superseded DeepONet ablation intermediates removed in the later repository cleanup; the three data files referenced by the frozen protocol and its summary remain tracked. The archive is intentionally outside Git; clone-only users cannot restore these
discarded intermediates without that archive. No active benchmark dependency
is intended to require them. Historical registry entries describing archived
legacy runs should be interpreted using the cleanup manifest.

Restore only the required named entries into their original project-relative
paths and verify their hashes. Do not extract the entire archive over a changed
checkout. Existing local documentation remains untracked.

## Scientific scope

Classical numerical methods win on the measured accuracy. Neural inference is
fast relative to the tested independent numerical solves, with unequal errors.
Near-expiry Greeks and financial restrictions remain limitations. No overall
advantage including training, labels and selection, or accuracy-matched
break-even point, has been established. New tuning needs fresh assessment cases.
