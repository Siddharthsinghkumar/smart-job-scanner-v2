# Research Summary: First Model Second (Deterministic LLM Workflows)

## Core Philosophy
The **"Blueprint First, Model Second"** framework is a paradigm for building reliable, deterministic LLM agent systems. It addresses the fundamental problem where standard LLM architectures conflate probabilistic, high-level planning with low-level action execution, leading to unpredictable behavior.

The framework centers on decoupling logic from the generative model:
1.  **Blueprint First (Deterministic Logic):** An expert-defined operational procedure is codified into a formal "Execution Blueprint" (e.g., Python source code, State Machines, LangGraph). This blueprint acts as the strict control plane.
2.  **Model Second (Specialized Tool):** The LLM is restricted to acting as a "bounded tool." It is only invoked to perform complex sub-tasks (e.g., extracting job requirements from text) that require natural language reasoning. **The LLM never decides the workflow's path.**

## Relevance to Smart Job Scanner V2
*   **VLM Integration:** When we deploy `SmolVLM-256M-Instruct` or `olmOCR2`, we must *not* ask the VLM to orchestrate the pipeline. We will use a rigid Python state machine to pass the image to the VLM, strictly parse its JSON response, and explicitly route the data to the next step.
*   **Preventing Loops:** This eliminates infinite loops and non-deterministic crashes during OCR extraction.

## Relevance to Job Discovery Engine (Super Project)
*   **LangGraph Orchestration:** The Super Project relies on LangGraph precisely for this reason. The nodes (Scraper, Evaluator, Writer) are the deterministic blueprint. The NVIDIA NIM APIs are merely the specialized tools invoked at each node.
