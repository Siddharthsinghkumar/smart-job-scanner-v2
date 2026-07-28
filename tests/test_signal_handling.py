"""Tests that signal handler sets shutdown flag and logs interruption."""

import signal
import sys
import pytest
from unittest.mock import patch, MagicMock

# Import the module to test
import src.pipeline.pipeline_runner as pipeline_runner

def test_signal_handler_sets_flag_and_exits():
    # Reset flag
    pipeline_runner._shutdown_requested = False
    
    with patch("sys.exit") as mock_exit, \
         patch("src.pipeline.pipeline_runner._active_process") as mock_process, \
         patch("src.pipeline.pipeline_runner._append_structured_log") as mock_log, \
         patch("src.pipeline.pipeline_runner._update_metrics") as mock_metrics, \
         patch("builtins.print") as mock_print:
         
        # Mock active process
        mock_process.poll.return_value = None
        
        # Trigger the handler manually for SIGINT
        pipeline_runner._signal_handler(signal.SIGINT, None)
        
        # Check flag
        assert pipeline_runner._shutdown_requested is True
        
        # Check print
        mock_print.assert_called_with("\n[!] Received SIGINT, initiating graceful shutdown...")
        
        # Check process termination
        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once_with(timeout=10)
        
        # Check logging
        mock_log.assert_called_once_with("pipeline", "interrupted", 0.0, 0)
        mock_metrics.assert_called_once_with("pipeline", "interrupted", 0.0, 0)
        
        # Check exit
        mock_exit.assert_called_once_with(128 + signal.SIGINT)

def test_signal_handler_with_timeout():
    pipeline_runner._shutdown_requested = False
    
    with patch("sys.exit") as mock_exit, \
         patch("src.pipeline.pipeline_runner._active_process") as mock_process, \
         patch("src.pipeline.pipeline_runner._append_structured_log") as mock_log, \
         patch("src.pipeline.pipeline_runner._update_metrics") as mock_metrics:
         
        mock_process.poll.return_value = None
        
        import subprocess
        mock_process.wait.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=10)
        
        pipeline_runner._signal_handler(signal.SIGTERM, None)
        
        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()
        mock_exit.assert_called_once_with(128 + signal.SIGTERM)
