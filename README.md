# Rare Gameplay Event Curator

**Task 6 — Kuration SDK**

A reproducible pipeline for selecting diverse, event-rich Counter-Strike gameplay windows for world-model and behavioral-cloning training. Instead of allowing common movement to dominate a dataset, the pipeline measures action/mouse activity, bins those signals, tags combat events, and uses Kuration's deterministic selection and shard export.

> **Important:** the built-in `--demo` mode creates synthetic structural fixtures to verify the pipeline. It does not contain real gameplay and is not evidence of real-world model improvement. Use actual recordings for evaluation.

## Why this problem matters

Long gameplay corpora contain many repeated low-information frames. Rare kill/death events and less-common action patterns may be underrepresented. Training on a more balanced sample can improve coverage of important situations, but this must be measured on the target dataset and model.

## Workflow

1. Read TeaPearce Counter-Strike HDF5 recordings through a dataset-specific adapter.
2. Map each frame's 51-value action target and kill/death labels to Kuration's canonical schema.
3. Create 96-frame windows with 48-frame stride.
4. Measure keyboard activity, mouse activity, and firing rate; bin movement signals; tag combat windows.
5. Select windows deterministically with event-tag coverage and export action/label arrays to NPZ shards.
6. Write `curation_report.json` with actual SDK counts and output-only diversity/event metrics.

```text
HDF5 recordings
      |
      v
TeaPearce adapter --> canonical ticks --> 96-frame windows
                                            |
                                            v
                     activity measures + bins + combat tags
                                            |
                                            v
                       deterministic balanced selection
                                            |
                                            v
                         NPZ shards + curation report
```

## Requirements

- Python 3.11+
- The provided Kuration SDK release ZIP
- `numpy` and `h5py`

### Install on Windows PowerShell

Unzip both archives. Replace the SDK path below with the folder containing its `pyproject.toml`.

```powershell
cd path\to\rare-gameplay-curator
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e "C:\path\to\kuration-sdk-private-release[array]"
python -m pip install -e .
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` in that terminal and activate again.

## Run the structural demo

```powershell
python -m rare_gameplay_curator.curate --demo --out artifacts\demo --n 8
```

Expected behavior: the run should finish, report candidate/selected window counts, and create NPZ shards plus `curation_report.json`. Exact counts can depend on SDK version. Synthetic frames are only for wiring and regression checks.

## Run on real recordings

The public TeaPearce dataset is very large (the SDK documentation describes hundreds of GB). Do not download the entire corpus just to test the code. Obtain recordings from the official dataset, check its current access/license terms, and begin with a small representative subset.

```powershell
python -m rare_gameplay_curator.curate data\*.hdf5 --out artifacts\curated --n 5000
```

Add `--include-frames` only if you need image tensors in output; this substantially increases storage. With multiple source recordings, the pipeline uses a source-level test split so overlapping windows from a recording do not cross the split boundary.

## Evaluation plan

The report currently measures:

- **Selection rate:** selected windows / candidate windows (SDK-reported).
- **Event yield:** kill and death labels in the exported samples.
- **Action-signature diversity:** unique thresholded action vectors / selected frames.
- **Shard count and selected frames:** operational output checks.

For a defensible experiment, compare against a seeded uniform-random baseline using the same recordings and budget. Report event recall (fraction of all candidate kill/death events represented in selected windows), action-signature diversity lift, source/map coverage, duplicate-window rate, runtime, peak memory, and output size. Repeat with multiple seeds. Use a manually reviewed sample to estimate event-label precision. Do not report a lift until baseline and curated runs have both been measured on real data.

## Guardrails and limitations

- A kill/death event is a useful proxy for combat, not a complete collision or physical-interaction label.
- The pipeline only preserves frames/actions present in the source. It does not infer missing labels or guarantee that a selected window teaches a useful physical law.
- Event coverage cannot select an event that is absent from the candidate pool.
- Output diversity is a proxy, not a semantic video similarity score. A future extension can add perceptual embeddings and near-duplicate detection behind the adapter/selection seam.
- Dataset permissions, privacy, and storage costs must be checked before sharing derived shards.
- Keep source manifests, SDK version, seed, parameters, and the generated report with every run for reproducibility.

## Extension hooks

The source adapter is isolated in `src/rare_gameplay_curator/teapearce_adapter.py`. Another simulator/game dataset can supply an adapter that emits the same canonical fields. Selection, output schema, and evaluation can then be extended without changing the SDK core.

## Repository layout

```text
src/rare_gameplay_curator/
  curate.py                 # CLI, Kuration pipeline, report generation
  teapearce_adapter.py      # source-specific HDF5 -> canonical tick adapter
 tests/
 docs/design.md
```

## Acknowledgements

Built against the provided Kuration SDK release. The included TeaPearce adapter is adapted from that SDK's worked example; see `SDK_LICENSE.txt` and the SDK's original license/attribution for applicable terms. Dataset content is not bundled here.
