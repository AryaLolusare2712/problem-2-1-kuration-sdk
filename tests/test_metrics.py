from rare_gameplay_curator.curate import summarize_shards


def test_empty_output_metrics_are_zero(tmp_path):
    metrics = summarize_shards(tmp_path)
    assert metrics["shards_written"] == 0
    assert metrics["selected_frames"] == 0
    assert metrics["selected_kill_events"] == 0
    assert metrics["selected_death_events"] == 0
    assert metrics["action_signature_diversity"] == 0.0
