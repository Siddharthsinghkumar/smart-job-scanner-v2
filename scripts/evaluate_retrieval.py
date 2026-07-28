"""
Evaluate the pipeline's retrieval and extraction quality using RAGAS.
Uses Merlin API (Claude Opus 4.8) via OmniRoute.
"""

import argparse
import json
import os

# Monkeypatch for ragas compatibility with newer langchain versions
import sys
import types
from pathlib import Path

from datasets import Dataset

try:
    import langchain_community
    if not hasattr(langchain_community, "chat_models"):
        langchain_community.chat_models = types.ModuleType("chat_models")
        sys.modules["langchain_community.chat_models"] = langchain_community.chat_models
    if not hasattr(langchain_community.chat_models, "vertexai"):
        vertexai = types.ModuleType("vertexai")
        vertexai.ChatVertexAI = type("ChatVertexAI", (object,), {})
        langchain_community.chat_models.vertexai = vertexai
        sys.modules["langchain_community.chat_models.vertexai"] = vertexai
        
    import langchain_core
    if not hasattr(langchain_core, "pydantic_v1"):
        import pydantic.v1 as pydantic_v1
        langchain_core.pydantic_v1 = pydantic_v1
        sys.modules["langchain_core.pydantic_v1"] = pydantic_v1
except Exception:
    pass

from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision

from src.utils.logging_utils import configure_logging

logger = configure_logging("evaluate_retrieval")

# Ragas requires LangChain chat models for evaluation.
# We configure it to use our OmniRoute proxy for Merlin (Claude Opus 4.8).

def get_eval_llm():
    api_key = os.getenv("MERLIN_API_KEY", "dummy-key")
    # Using omniroute endpoint if applicable, otherwise a standard endpoint
    base_url = os.getenv("MERLIN_BASE_URL", "http://localhost:8080/v1") 
    
    return ChatOpenAI(
        model="claude-opus-4-8",
        api_key=api_key,
        base_url=base_url,
        temperature=0.0
    )

def prepare_dataset(shortlist_file: Path, resume_dir: Path):
    if not shortlist_file.exists():
        logger.error(f"Shortlist file not found: {shortlist_file}")
        return None
        
    with open(shortlist_file, "r") as f:
        shortlist = json.load(f)
        
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    # Just take a sample of up to 5 for evaluation
    for item in shortlist[:5]:
        job_text = item.get("job_text", "")
        best_resume = item.get("best_resume", "")
        
        resume_path = resume_dir / best_resume
        if not resume_path.exists():
            continue
            
        with open(resume_path, "r") as f:
            resume_text = f.read()
            
        # RAGAS mapping:
        # question: Implicit question (e.g., "What are the job requirements?")
        # answer: The LLM extraction or candidate shortlisting reasoning.
        # contexts: The source job description.
        # ground_truth: The ideal candidate profile (the resume).
        
        data["question"].append("What are the key requirements for this job, and is the candidate a match?")
        data["answer"].append(f"Candidate matched with similarity {item.get('similarity')}. Match found.")
        data["contexts"].append([job_text[:1000]])  # truncate for context
        data["ground_truth"].append(resume_text[:1000])
        
    if not data["question"]:
        return None
        
    return Dataset.from_dict(data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shortlist", type=str, default="data/shortlisted_jobs_json/shortlisted_jobs.json")
    parser.add_argument("--resumes", type=str, default="data/dynamic_resumes")
    args = parser.parse_args()
    
    logger.info("Preparing dataset for RAGAS evaluation...")
    dataset = prepare_dataset(Path(args.shortlist), Path(args.resumes))
    
    if dataset is None:
        logger.warning("Could not prepare dataset. Skipping evaluation.")
        print("⚠️ Skipping RAGAS evaluation: Missing shortlist dataset or resumes.")
        return 0
        
    llm = get_eval_llm()
    
    logger.info("Running RAGAS evaluation...")
    result = evaluate(
        dataset,
        metrics=[answer_relevancy, context_precision],
        llm=llm
    )
    
    logger.info("RAGAS Evaluation Results", results=result)
    print("\n--- Evaluation Results ---")
    print(result)
    return 0

if __name__ == "__main__":
    exit(main())
