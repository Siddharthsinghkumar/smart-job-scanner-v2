#!/usr/bin/env python3
"""Unified Ollama runner facade for local extraction stage."""

from src.pipeline.stage07_llm_extraction import main, query_model_safe, start_ollama_serve  # noqa: F401

__all__ = ['main', 'query_model_safe', 'start_ollama_serve']
