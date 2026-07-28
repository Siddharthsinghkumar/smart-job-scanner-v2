# Research Summary: Understanding New Knowledge-Induced Factual Hallucinations in LLMs

## Core Findings
This research investigates how fine-tuning LLMs (or injecting new context) on unfamiliar information inadvertently causes them to generate incorrect outputs—even regarding previously known facts.

*   **The Mechanism:** Learning new knowledge disrupts the model’s internal attention patterns. It weakens the focus on key entities in the prompt, causing the LLM to over-rely on surrounding irrelevant context.
*   **The Driver:** The degree of unfamiliarity (not the volume of data) triggers these hallucinations.
*   **The Fix ("KnownPatch"):** To mitigate this, researchers introduce "KnownPatch"—re-introducing known, mastered knowledge samples into the prompt or fine-tuning process to ground the model and restore its attention patterns.

## Relevance to Smart Job Scanner V2
*   **VLM Hallucination:** When feeding `olmOCR2` or `SmolVLM` highly irregular job ads (which contain unfamiliar acronyms or chaotic layouts), the VLM might hallucinate standard job requirements that aren't actually in the image. We must use RLVR (Reinforcement Learning with Verifiable Rewards) or strict unit-test evaluations to benchmark and catch this.
*   **Prompt Grounding:** We should anchor our VLM prompts with "Known Patches" (e.g., providing a rigid JSON schema and explicitly instructing it to return `null` if a field is missing, grounding its attention).

## Relevance to Job Discovery Engine (Super Project)
*   **Persona Bleeding:** When generating the Dual-Merlin resume and DM, injecting *new* context (a startup's obscure tech stack) might cause the LLM to hallucinate facts about *your* past experience. This mathematically validates our decision to use strict FAISS metadata tags to ground the LLM in your "known" Persona data before every generation.
