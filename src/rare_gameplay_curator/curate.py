"""Curate gameplay recordings for world-model training.

Quick demo (no upstream gameplay download):
    python -m rare_gameplay_curator.curate --demo --out artifacts/demo

Real recordings:
    python -m rare_gameplay_curator.curate data/*.hdf5 --out artifacts/curated --n 5000
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from kuration import OutputSpec, input, iter_shards

from .teapearce_adapter import (
    ACTION_WIDTH,
    FrameEncoder,
    TeaPearceHDF5Adapter,
    teapearce_schema,
    write_demo_recording,
)


def output_spec(*, include_frames: bool, frame_shape: tuple[int, ...] | None) -> OutputSpec:
    """Keep labels and action targets together so samples remain trainable."""
    modalities: dict[str, dict[str, Any]] = {
        "action": {"dtype": "float32", "shape": [ACTION_WIDTH], "source": "field:action"},
        "kill": {"dtype": "uint8", "shape": [], "source": "field:kill"},
        "death": {"dtype": "uint8", "shape": [], "source": "field:death"},
        "mouse_dx": {"dtype": "float32", "shape": [], "source": "field:mouse_dx"},
        "mouse_dy": {"dtype": "float32", "shape": [], "source": "field:mouse_dy"},
    }
    if include_frames:
        if frame_shape is None:
            raise ValueError("frame_shape is required when include_frames=True")
        modalities["frame"] = {
            "dtype": "uint8", "shape": list(frame_shape), "source": "encoder:frame"
        }
    return OutputSpec.from_dict({
        "container": "npz", "shard_length": 96, "short_policy": "keep",
        "modalities": modalities,
    })


def summarize_shards(out_dir: str | Path) -> dict[str, Any]:
    """Compute transparent post-export metrics from actual emitted shards."""
    total_steps = kills = deaths = 0
    action_signatures: set[tuple[int, ...]] = set()
    shard_count = 0
    if not (Path(out_dir) / "dataset_manifest.json").exists():
        return {
            "shards_written": 0, "selected_frames": 0,
            "selected_kill_events": 0, "selected_death_events": 0,
            "unique_action_signatures": 0, "action_signature_diversity": 0.0,
            "note": "No dataset manifest found; no output shards were evaluated.",
        }
    for arrays in iter_shards(str(out_dir)):
        shard_count += 1
        actions = arrays.get("action")
        if actions is not None:
            total_steps += int(len(actions))
            # Compact action signature: active key/click indices and dominant mouse bins.
            for action in actions:
                action_signatures.add(tuple(int(i) for i in (action > 0.5).nonzero()[0]))
        kills += int(arrays.get("kill", []).sum())
        deaths += int(arrays.get("death", []).sum())
    return {
        "shards_written": shard_count,
        "selected_frames": total_steps,
        "selected_kill_events": kills,
        "selected_death_events": deaths,
        "unique_action_signatures": len(action_signatures),
        "action_signature_diversity": round(len(action_signatures) / total_steps, 6) if total_steps else 0.0,
        "note": "These are output-only metrics; compare with a full-corpus baseline before claiming improvement.",
    }


def run(sources: list[str], out_dir: str, *, n: int = 5000,
        include_frames: bool = False, seed: int = 21) -> dict[str, Any]:
    if not sources:
        raise ValueError("provide one or more .hdf5 recordings or use --demo")
    if n < 1:
        raise ValueError("n must be at least 1")
    schema = teapearce_schema()
    adapter = TeaPearceHDF5Adapter(schema, include_frames=include_frames)
    for source in sources:
        adapter.check_source(source)

    frame_shape = None
    if include_frames:
        shapes = {adapter.frame_shape(source) for source in sources}
        if len(shapes) != 1:
            raise ValueError(f"all frame shapes must agree, got {sorted(shapes)}")
        frame_shape = shapes.pop()

    # Kuration performs windowing, signal measurement, binning, event-tag coverage,
    # deterministic sampling and source-level splitting. Tag coverage reserves capacity
    # for rare kill/death events when such events exist in the candidate pool.
    pipeline = (
        input(adapter=adapter, sources=sources, schema=schema)
        .window(96, stride=48)
        .measure("action_density", "mean", of="keys_down")
        .measure("mouse_activity", "mean", of="mouse_motion")
        .measure("shoot_rate", "rate", of="left_click")
        .bin("action_density", bins=4)
        .bin("mouse_activity", bins=4)
        .tag("combat.kind", min_frac=0.01)
        .sample("equal", n=n)
        .options(mode="strict", seed=seed)
    )
    if len(sources) > 1:
        pipeline = pipeline.split(test=0.2)

    result = pipeline.output(
        out_dir,
        spec=output_spec(include_frames=include_frames, frame_shape=frame_shape),
        encoder=FrameEncoder() if include_frames else None,
    )
    metrics = summarize_shards(out_dir)
    report = {
        "project": "Rare Gameplay Event Curator",
        "input_recordings": len(sources),
        "requested_windows": n,
        "sdk_selected_windows": result.get("selected"),
        "sdk_candidate_windows": result.get("results", {}).get("segments_total"),
        "output_dir": str(out_dir),
        "metrics": metrics,
        "sdk_result": result,
    }
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "curation_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "sdk_result"}, indent=2))
    return report


def _demo_sources(directory: str) -> list[str]:
    sources = []
    for recording in range(4):
        path = Path(directory) / f"synthetic_recording_{recording}.hdf5"
        write_demo_recording(path, recording=recording)
        sources.append(str(path))
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="*", help="extracted TeaPearce .hdf5 recordings")
    parser.add_argument("--out", default="artifacts/curated", help="output dataset directory")
    parser.add_argument("--n", type=int, default=5000, help="maximum number of windows to select")
    parser.add_argument("--include-frames", action="store_true", help="include image tensors; output becomes much larger")
    parser.add_argument("--demo", action="store_true", help="run on generated structural fixtures, not real gameplay")
    parser.add_argument("--seed", type=int, default=21, help="deterministic sampling seed")
    args = parser.parse_args()
    if args.demo and args.sources:
        parser.error("--demo cannot be combined with source paths")
    if not args.demo and not args.sources:
        parser.error("provide source paths or use --demo")
    if args.demo:
        with tempfile.TemporaryDirectory(prefix="rare_gameplay_demo_") as directory:
            run(_demo_sources(directory), args.out, n=args.n, include_frames=args.include_frames, seed=args.seed)
    else:
        run(args.sources, args.out, n=args.n, include_frames=args.include_frames, seed=args.seed)


if __name__ == "__main__":
    main()
