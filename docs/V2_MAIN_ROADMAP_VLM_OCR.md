# 🚀 Smart Job Scanner V2: The VLM / OCR Roadmap
*Branch: `main`*

> **Superseded 2026-07-08 by [`V2_MILESTONE_ROADMAP.md`](V2_MILESTONE_ROADMAP.md)** — the master plan with milestones M0–M7, audit-finding traceability, and verified reference checks (note: SmolVLM-256M is English-only; a Devanagari-capable candidate is mandatory). This file is kept for the model shortlist and paper pointers.

## 🛑 1. Deprecation of YOLO
The YOLO-based OCR pipeline has been officially deemed a "shit approach" due to its inability to reliably parse dense, highly-variable text blocks (like newspaper ads).
*   All YOLO scripts, models (`yolov8s.pt`), and training datasets have been stripped from the `main` branch.
*   They are archived in the `archive-yolo-experiments` branch for historical reference.

## 🎯 2. The New Approach: Vision Language Models (VLMs)
We are upgrading V2 from basic OCR to intelligent VLM extraction. We will benchmark two state-of-the-art open-source models:

### Model A: `HuggingFaceTB/SmolVLM-256M-Instruct`
*   **Why:** It is incredibly lightweight (256M parameters), blazing fast, and designed to run entirely locally on your hardware without eating up VRAM.
*   **Use Case:** Rapid, first-pass extraction and bounding-box detection on newspaper clippings.

### Model B: `allenai/olmocr2` (olmOCR 2)
*   **Why:** Fine-tuned specifically for parsing complex documents (like PDFs, math, tables) into clean Markdown. It uses Reinforcement Learning with Verifiable Rewards (RLVR) to strictly prevent hallucination.
*   **Use Case:** Deep, high-fidelity extraction where layout context matters.

---

## 📈 3. MVP Milestones for V2 Main Branch

*   **Milestone 1: The VLM Benchmarking Pipeline**
    *   Set up `SmolVLM-256M-Instruct` locally.
    *   Deploy `olmOCR2` toolkit.
    *   Run the multi-language OCR benchmark test suite (from GitHub) against both models using our dataset of newspaper job ads.
*   **Milestone 2: Structuring the Output**
    *   Force the winning VLM to output strictly formatted JSON/Markdown containing: `Company Name`, `Role`, `Requirements`, `Contact Info`.
*   **Milestone 3: Integration into the Super Project**
    *   Once V2 can flawlessly convert a newspaper ad into a clean JSON job description, we plug it into the `/home/sidd/project/job-discovery-engine/` (Super Project).
    *   *The Handoff:* V2 extracts the job -> Passes it to the Super Project -> The Super Project compares it against your `persona-context-engine` FAISS DB -> Dual-Merlin CLI generates the resume and cover letter.

---

## 📚 4. Required Academic Research Ingestion
Two critical papers are being ingested to guide the prompt engineering and state-machine logic for this new VLM pipeline:
1.  **"Blueprint First, Model Second"**: We will strictly use LangGraph (or Python state machines) to dictate the workflow, using the VLMs *only* as bounded tools for extraction, never for decision-making.
2.  **"Knowledge-Induced Factual Hallucinations"**: We will use "KnownPatch" prompting techniques to ensure the VLM doesn't hallucinate non-existent job requirements when encountering unfamiliar domain terms.
