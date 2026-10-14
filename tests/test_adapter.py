from pathlib import Path

import pytest

h5py = pytest.importorskip("h5py")

from rare_gameplay_curator.teapearce_adapter import (
    ACTION_WIDTH,
    TeaPearceHDF5Adapter,
    teapearce_schema,
    write_demo_recording,
)


def test_adapter_reads_synthetic_structure(tmp_path: Path):
    path = tmp_path / "recording.hdf5"
    write_demo_recording(path, recording=0, n_frames=16)
    adapter = TeaPearceHDF5Adapter(teapearce_schema())
    adapter.check_source(str(path))
    ticks = adapter.ticks(str(path))
    assert len(ticks) == 16
    fields = ticks[0].entities["player"]
    assert fields["action"].shape == (ACTION_WIDTH,)
    assert "kill" in fields and "death" in fields


def test_adapter_rejects_empty_hdf5(tmp_path: Path):
    path = tmp_path / "empty.hdf5"
    with h5py.File(path, "w"):
        pass
    adapter = TeaPearceHDF5Adapter(teapearce_schema())
    with pytest.raises(ValueError, match="no TeaPearce"):
        adapter.check_source(str(path))
