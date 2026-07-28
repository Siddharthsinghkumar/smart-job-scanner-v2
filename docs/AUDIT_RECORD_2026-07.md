# Audit Record — 2026-07-07/08

What was done, what was decided, and where everything lives. Companion to `V2_MILESTONE_ROADMAP.md`.

## What was done

A full adversarial static audit of this repository (commit `f51f606`) per Sid's ETL/headless deliverable spec v2.0, with an orchestration-and-operability mandate. Method: static code inspection + read-only shell verification; no pipeline execution, no network calls, no secret values read.

**19 deliverables under `research/`:** executive summary (`00`), system inventory (`01`), orchestration review (`02`), stage/state audit (`03`), data/SQLite audit (`04`), external APIs & concurrency (`05`), scraping reliability (`06`), Telegram/notification (`07`), observability/operations (`08`), code structure (`09`), improvement roadmap (`10`), evidence appendix (`11`), `findings.json`, `audit-manifest.json`, and 5 CSV matrices (risk register, pipeline stages, dependencies, external APIs, SQLite query surface).

**53 findings:** 3 critical, 14 high, 22 medium, 8 low, 6 informational/positive; 5 explicit hypotheses (H-01..H-05) with the runtime test that would confirm each. Every claim carries file:line evidence.

**Headline:** the failure-signal chain was broken at every link — stages exit 0 on real failure (F-002) → runner marks success → scheduler ignores exit codes → no alerting exists → nightly cleanup deletes the evidence (F-003, F-013). The audit found the system dormant (venv + data absent) — see decisions below for why.

## Decisions taken 2026-07-08 (Sid)

These resolve the audit's open questions and set the rebuild direction:

1. **`4_env/` (10 GB) and `data/` (30 GB) were deleted deliberately** — bloated with unused libraries / irrelevant data. F-001 reclassified: not an outage mystery; the durable fix is a lean `.venv` and no hardcoded interpreter paths.
2. **A copy of the old system is running in Docker** — that answers the audit's "second deployment?" question. It is live production, carries the audited defects, and gets retired at roadmap M4 cutover after a 7-day shadow run.
3. **YOLO block detection is abandoned** ("shit approach") — archived on branch `archive-yolo-experiments` (verified to exist); `main` is clean of it.
4. **New direction:** VLM-based extraction replacing stages 02–05, chosen by benchmark (OCRBench v2 + MDPBench Hindi + own golden set via `Yuliang-Liu/MultimodalOCR` tooling), candidates including SmolVLM-256M (verified **English-only** — cannot be sole engine), olmOCR 2, and a mandatory Devanagari-capable model.
5. **LLM runtime direction:** `llama-cpp-turboquant` (verified real/active, `tqp-v0.2.0`, 3-bit polar KV quant) as a systemd service; pinned + logged system prompts and sampling parameters for local and API calls; KV/context sized by measurement (`n_ctx ≥ p95 × 1.25`), never by round numbers.
6. **BDH (`pathwaycom/bdh`)** verified to be a research architecture with no pretrained weights — demoted to a time-boxed research spike; the production-shaped version of the same instinct is a LoRA fine-tune on our own accumulated extraction pairs.
7. **Research spikes approved:** canary attention probe (3 position-tagged nonces in prompt/context to separate lost-in-the-middle vs bad prompt vs bad model), KnownPatch hallucination eval, source-diversification probe.

## Where everything lives

- `research/` — the 19 audit deliverables (start at `00-executive-summary.md`; registry in `findings.json`).
- `docs/V2_MILESTONE_ROADMAP.md` — master milestone plan M0–M7 with all 53 findings dispositioned (Appendix A).
- `docs/V2_MAIN_ROADMAP_VLM_OCR.md` — earlier short roadmap; superseded, kept for model shortlist + paper pointers.
- `docs/blueprint_first_deterministic_llm.md`, `docs/knowledge_induced_hallucinations.md` — research summaries feeding M2/M6 design.
- Branch `archive-yolo-experiments` — the dead YOLO line.
- Super project: `/home/sidd/project/job-discovery-engine/` — integration contract at roadmap M7.
