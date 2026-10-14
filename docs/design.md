# Design note

## Decision
Curate action-rich and combat-event-rich windows from Counter-Strike gameplay for behavioral-cloning/world-model datasets. Rare events are often diluted by repeated traversal or low-action frames.

## SDK usage
- `TeaPearceHDF5Adapter`: maps upstream frame-keyed HDF5 into `Tick` records and exposes action, keyboard/mouse activity, kill/death, and event streams.
- `input(...).window(96, stride=48)`: produces temporal candidates while overlapping windows increase context coverage.
- `measure` + `bin`: characterize action density, mouse activity, and firing rate.
- `tag("combat.kind")`: retain the event category as a selection signal.
- `sample("equal")` + seeded strict options: reproducible selection across activity bins.
- `split(test=0.2)`: source-level holdout when there are multiple recordings.
- `OutputSpec` + `pipeline.output`: preserve action targets and labels in training-ready NPZ shards.

## Evaluation contract
The script emits the SDK's candidate/selected counts and output metrics. The output metrics are not a full evaluation by themselves. Before claiming success, run a random baseline with the same source files, windowing, sample budget, and seeds. Compare event recall, action-signature diversity, duplicate rate, per-source coverage, runtime, peak memory, and output size. Manually review sampled event windows to estimate label precision.

## Failure handling
Malformed or non-contiguous HDF5 frame keys and unexpected action/helper shapes are rejected by the adapter. Frame export is opt-in because it materially increases disk use. Strict selection can fail if requested sample counts exceed available candidates; lower `--n` when necessary.

## Known gaps
The first version prioritizes annotated kill/death events and behavior signals; it does not perform 3D collision reconstruction. Rare kill/death windows are not equivalent to rare physical events. Perceptual near-duplicate detection, a uniform baseline command, and model-level downstream evaluation are planned follow-ups.
