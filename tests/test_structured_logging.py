import os
import tempfile
import json
from pathlib import Path
from src.utils.logging_utils import configure_logging

def test_structured_logging_json_output():
    # Use a temp directory for logs
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        # Configure logging for a test stage
        logger = configure_logging("stage_test_123", log_dir=log_dir)
        
        # Log a test message
        logger.info("This is a test message", foo="bar", answer=42)
        
        # Check if log file was created and contains valid JSON
        log_file = log_dir / "stage_test_123.jsonl"
        assert log_file.exists(), "Log file should be created"
        
        # Parse the JSON
        lines = log_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1, "Should have exactly one log line"
        
        log_data = json.loads(lines[0])
        
        # Verify structure
        assert log_data["event"] == "This is a test message"
        assert log_data["foo"] == "bar"
        assert log_data["answer"] == 42
        assert "timestamp" in log_data
        assert log_data["level"] == "info"
