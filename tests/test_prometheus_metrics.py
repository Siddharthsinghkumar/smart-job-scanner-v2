"""Tests that prometheus metrics are exposed correctly."""

import time
import threading
import pytest
from unittest.mock import patch, MagicMock
from prometheus_client import REGISTRY

# Import module
import src.pipeline.pipeline_runner as pipeline_runner

def test_prometheus_metrics_registry():
    # Verify that metrics are registered in the global registry
    metrics = {m.name: m for m in REGISTRY.collect()}
    
    assert 'jobv2_stage_duration_seconds' in metrics
    assert 'jobv2_stage_runs' in metrics or 'jobv2_stage_runs_total' in metrics
    assert 'jobv2_pipeline_running' in metrics
    assert 'jobv2_jobs_processed' in metrics or 'jobv2_jobs_processed_total' in metrics

def test_pipeline_running_gauge():
    # Test that the gauge works
    pipeline_runner.PIPELINE_STATUS.set(1)
    
    val = REGISTRY.get_sample_value('jobv2_pipeline_running')
    assert val == 1.0

def test_stage_metrics_update():
    # Test _update_metrics
    with patch("pathlib.Path.write_text") as mock_write, \
         patch("pathlib.Path.exists", return_value=False):
        
        # Call for a non-pipeline stage
        pipeline_runner._update_metrics("src/pipeline/stage01_test", "success", 1.5, 42)
        
        val_duration_count = REGISTRY.get_sample_value('jobv2_stage_duration_seconds_count', labels={'stage_name': 'src/pipeline/stage01_test', 'status': 'success'})
        assert val_duration_count is not None
        assert val_duration_count >= 1.0
        
        val_runs = REGISTRY.get_sample_value('jobv2_stage_runs_total', labels={'stage_name': 'src/pipeline/stage01_test', 'status': 'success'})
        if val_runs is None:
            val_runs = REGISTRY.get_sample_value('jobv2_stage_runs_total_total', labels={'stage_name': 'src/pipeline/stage01_test', 'status': 'success'})
        assert val_runs is not None
        assert val_runs >= 1.0
        
        val_jobs = REGISTRY.get_sample_value('jobv2_jobs_processed_total')
        if val_jobs is None:
            val_jobs = REGISTRY.get_sample_value('jobv2_jobs_processed_total_total')
        assert val_jobs is not None
        assert val_jobs >= 42.0
