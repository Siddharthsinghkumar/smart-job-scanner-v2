# Smart Job Scanner V2 — Master Milestone Roadmap

*Written 2026-07-08. Supersedes `V2_MAIN_ROADMAP_VLM_OCR.md` (kept for the model shortlist and paper notes). Grounded in the 2026-07 audit: `research/00-executive-summary.md`, `research/findings.json` (53 findings), `research/10-improvement-roadmap.md`. Every finding is dispositioned in Appendix A.*

## Ground truth as of 2026-07-08 (decisions, not findings)

- `4_env/` (10 GB) and `data/` (30 GB) were **deleted deliberately** by Sid — dead libraries and stale data. F-001 is resolved-by-decision, not an outage mystery.
- **The old system is still running in a Docker container** — it is the live production instance and it carries the audited defects (F-002 stages lie about success, F-003 silent no-op nights). It stays up only until M4 cutover.
- **YOLO block detection is dead.** Archived on branch `archive-yolo-experiments`; `main` is clean of it. Not coming back.
- `main` is the rebuild branch: VLM-based extraction, benchmark-gated model choice, truth-first orchestration, ending in integration with the super project (`/home/sidd/project/job-discovery-engine/`).

## Verified reference stack (checked 2026-07-08)

| Reference | Verified reality | Role |
|---|---|---|
| `Yuliang-Liu/MultimodalOCR` | Real. OCRBench (1k QA), OCRBench v2 (10k QA, 31 scenarios), **MDPBench (17 languages incl. Hindi)**. Runnable via VLMEvalKit / lmms-eval. | The benchmark harness for M2. |
| `HuggingFaceTB/SmolVLM-256M-Instruct` | Real. 256M params, <1 GB VRAM, strong DocVQA for its size. **English only — no Devanagari.** | Fast-tier candidate for English pages only. Cannot be the sole OCR engine (papers are Hindi+English). |
| `pathwaycom/bdh` | Real, but a **research architecture with no pretrained weights** — GPT-2-scale, `train.py` on toy data, train-it-yourself. | Research spike only (M6). Not on the MVP path. |
| `TheTom/llama-cpp-turboquant` | Real and healthy: llama.cpp fork, ~2k stars, `tqp-v0.2.0` (July 2026), 3-bit polar KV quant (`turbo3`), asymmetric K/V tiers, CUDA/Vulkan, prebuilt Linux binaries, used by LocalAI. | LLM runtime candidate for M3. |

---

## Milestone map

```
M0 hygiene ─→ M1 truth layer ─→ M2 OCR/VLM gate ⛔ ─→ M3 LLM runtime ─→ M4 MVP + cutover ⛔ ─→ M5 state/data ─→ M7 super-project
                                      │
                                      └── M6 research spikes (parallel, time-boxed, never blocking)
```

⛔ = decision gate: Sid's explicit verdict required before proceeding; verdict gets frozen into this file.

---

## M0 — Hygiene & record (≈ half a day)

Goal: repo tells the truth about its own state; environment is lean and rebuildable.

- [ ] Commit `docs/` (this roadmap, audit record, research notes) so the plan survives.
- [ ] Tag current `main` as `legacy-v2` before surgery — the Docker container runs approximately this code; keep a pin to diagnose it against.
- [ ] Slim `requirements.txt`: drop confirmed-dead deps (`pandas`, `torchaudio` — F-033 evidence), split CI-light vs GPU-full requirement sets (F-037/R-21). Target venv well under 2 GB (the 10 GB `4_env` was the symptom that triggered its deletion).
- [ ] New venv at `.venv/`; **remove every hardcoded `4_env/bin/python` path** (Makefile, scripts, docs) — interpreter path comes from env/activation, never literals. This is the durable fix for F-001's class.
- [ ] Relocate disk-loose secrets out of the working tree to `~/.config/smart-job-scanner/` (Telethon session at repo root, bot token file, live gemini config) — F-029. Tree keeps `.example` files only.
- [ ] Delete dead scaffolding: `src/utils/*` (zero importers), `src/llm/llm_utils.py`, orphaned `progress.db` (F-033). Backup DB file before deleting, per standing rule.
- [ ] Check GitHub Actions status once online (closes H-05).

Exit proof: `pip install -r requirements.txt` into fresh `.venv` + `make test` output pasted; `git grep 4_env` returns nothing live.

## M1 — Truth layer: the F-002/F-003 refactor (≈ 1 week)

Goal: **a stage that produced nothing must not exit 0.** This is the design law of the rebuild; every new stage inherits it. Adopting Prefect/VLMs/anything before this just paints failures green in a nicer dashboard (audit's core conclusion).

- [ ] Stage contract + shared `stage_guard` helper: non-zero exit on failure AND on zero yield (unless explicit `--allow-empty`); yield counts printed in a machine-parseable trailer line. Fixes F-002; retro-fit to surviving stages (downloader, 01, 07-11) — do not invest in stages 02–05, they die at M2.
- [ ] Downloader honesty: non-zero on partial/zero yield vs expected source count; flip shipped debug config (`headless: true`, `max_preprocess_workers: 3`) — F-027/F-028.
- [ ] Runner: `flock` single-instance lock (F-005); per-stage retry now actually fires because exit codes are real; `MAX_RETRIES=0` means zero, not infinite (F-039); date-scoped `.done` markers as interim (F-044).
- [ ] **Telegram ops hook**: the system owns a bot — make it report on itself. On any stage failure and as an end-of-run digest (per-stage yields, durations). Fixes F-003's invisibility and F-036's alerting half. Closes H-02 check during runner rework.
- [ ] Cleanup safety: stage11 deletion gated on verified yield of the run, `--yes` flag instead of piped `y`, archive-then-prune for raw PDFs — F-013.
- [ ] Notification delivery: per-job `sent_at` column (watermark no longer advances past failures — F-008), escape URL field (F-009), honor `retry_after` on 429 (F-011), delete the daily live test message (F-010), `--max-per-run` cap for post-outage bursts (R-25), rename/clarify `sent_jobs` semantics (F-012).
- [ ] Metrics: delete the hardwired-zero counters or emit real counts from the yield trailers — no fake telemetry survives (F-006).
- [ ] Delete `src/pipeline/pipeline_runner.py` (drifted duplicate — F-007). One runner.
- [ ] Remove `task_backlog.txt` write-only data-loss path in stage04 even though stage04 is scheduled to die — it's live until M4 (F-016).
- [ ] Regression test per fix above (F-037): watermark failure case, URL escaping, zero-yield exit, lock contention.

Exit proof: kill Ollama mid-run → pipeline exits non-zero, Telegram alert received (screenshot), raw PDFs still on disk.

## M2 — OCR/VLM benchmark gate ⛔ (≈ 1–2 weeks)

Goal: replace stages 02–05 (block detect → refine → EasyOCR → Argos translate) with benchmark-chosen VLM extraction. **No model ships without numbers.**

- [ ] Golden set first: 50–100 newspaper pages/ad crops (mix Hindi + English) with hand-verified transcriptions and expected job-field extractions. This is the yardstick everything else is measured with; without it the public benchmarks only rank models in general, not on our data.
- [ ] Harness: VLMEvalKit or lmms-eval driving OCRBench v2 + **MDPBench Hindi subset** + the golden set. One command, one CSV of scores per model.
- [ ] Candidates (fast tier → fidelity tier):
  - `SmolVLM-256M-Instruct` — English pages only (verified English-only). Speed/VRAM floor.
  - `allenai/olmOCR-2` — document-parsing fine-tune, English-focused.
  - **At least one Devanagari-capable model** — e.g. Qwen2.5-VL (3B/7B) or Surya OCR — mandatory, because Hindi papers are in scope and neither SmolVLM nor olmOCR covers them.
  - Baseline: current EasyOCR(hi,en)+Argos path, so the incumbent has a score too.
- [ ] Metrics: char/word accuracy, job-field extraction F1, **hallucination rate** (KnownPatch-style: rigid JSON schema, `null` for missing fields, count invented fields — per `docs/knowledge_induced_hallucinations.md`), pages/hour, peak VRAM.
- [ ] Structured output from day one: VLM is a bounded tool inside a deterministic Python state machine (per `docs/blueprint_first_deterministic_llm.md`); strict JSON schema parse, explicit routing — the VLM never decides flow.
- [ ] ⛔ Decision gate: model(s) chosen (possibly two-tier: cheap English + heavier Hindi), and verdict on whether stage05 translation survives (a VLM reading Hindi directly may output English fields and kill the translation stage entirely).

Obsoletes on adoption: stage02/03/04 wholesale — F-014, F-015, F-017, F-035 (cache_blocks), F-040 die with them.

Exit proof: benchmark CSV + side-by-side extraction on 10 real pages pasted for verdict.

## M3 — LLM runtime & parameter hygiene (≈ 1 week)

Goal: local LLM inference becomes a managed service with pinned, logged, reproducible parameters — for local **and** API calls.

- [ ] Runtime: `llama-cpp-turboquant` server (`tqp-v0.2.0`) as a **systemd service**; stages only check readiness. Kills stage07's owned Ollama lifecycle including `pkill -f ollama` (F-018). Evaluate `turbo3` (3-bit polar) KV tier vs fp16 KV on extraction quality before enabling — quantized KV is a quality knob, not a free lunch; note the fork's own guidance that K tolerates less compression than V.
- [ ] **Parameter hygiene (applies to local AND Gemini):** a single `configs/llm_params.json` pinning per-task: system prompt (versioned), `temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `seed`, `n_ctx`, KV quant tier, model+quant identity. Every request logs the full param set + prompt version into the interactions DB. Today neither the Ollama path nor Gemini pins or records any of this — results are unreproducible.
- [ ] Extraction defaults to start from: `temperature 0–0.2`, `top_p 0.9`, fixed `seed`, JSON-schema/GBNF-constrained output. Grammar-constrained output also kills the substring-verdict parsing in stage09d.
- [ ] **KV-cache sizing table** (the "how big should n_ctx be" method, reused for the hiring-agent 32k question in M7):
  1. Instrument real token counts: p50/p95/max of (prompt + completion) per task type.
  2. Rule: `n_ctx ≥ p95_total × 1.25` — measured need plus ~20–25% headroom, because quality degrades in the final stretch of the window; never plan to *use* the last 20%.
  3. Memory check: `KV bytes/token = 2 × n_layers × n_kv_heads × head_dim × bytes_per_elem`. Illustrative: Qwen2.5-7B-class GQA ≈ 56 KB/token fp16 → 32k ctx ≈ 1.8 GB; `turbo3` ≈ ×0.22 → ~0.4 GB. Scanner batches (1,000 words ≈ 1.4k tokens + prompt + few-shot ≈ 3–4k p95) need only **8k ctx** — don't pay for 32k here.
- [ ] Gemini client fixes riding along: lock or per-client keys around `genai.configure` (F-021), single sleep instead of double 1.2s, keep key-rotation engine but consolidate (F-022 accepted-tradeoff).
- [ ] Bench `num_parallel`/parallel-request settings on the new runtime (closes H-03).

Exit proof: same input + same params file → byte-identical JSON extraction twice in a row; interactions DB row shows full param set.

## M4 — **MVP**: new pipeline end-to-end + Docker cutover ⛔ (≈ 1–2 weeks)

Goal: the new, shorter, truthful pipeline runs nightly and replaces the Docker legacy container.

- [ ] New DAG (≈7 stages, down from 14): download → pdf2img → **VLM extract (M2 winner) → structured JSON** → resume match → shortlist/analyze → notify → gated cleanup. Every stage under the M1 contract; manifests passed between stages so downstream distinguishes "upstream empty" from "upstream failed" (F-031).
- [ ] Scheduling: **two systemd timers** (`Persistent=true`, `OnFailure=` → Telegram alert). `scheduler.sh` is deleted, not fixed — F-004, F-038 retired with it.
- [ ] **Shadow run: 7 days.** New pipeline and Docker legacy both run; nightly Telegram digest compares job counts/coverage per paper. This is the proof standard — numbers per night, not "it should work."
- [ ] ⛔ Cutover gate: ≥ legacy parity on the shadow week AND 7 consecutive honest-green nights → **stop and remove the Docker container** (its F-002/F-003 defects retire with it). Sid pulls the trigger.
- [ ] Dedup on structured JSON keys (company+role+contact) replaces O(n²) SequenceMatcher prose dedup (F-041).
- [ ] Accepted for MVP: fully synchronous nightly run (F-023) — volumes don't justify async; revisit only if wall-time exceeds the night.

Exit proof: 7 nightly digest screenshots + `docker ps` showing the legacy container gone.

## M5 — State & data consolidation (≈ 1–2 weeks, post-MVP)

Goal: one database, one storage module, run identity, replay paths — retire the nine fragmented state mechanisms.

- [ ] `core/storage.py`: single `scanner.db`, WAL by default, `user_version` migration ladder, named-column row access. Collapses 4 DBs + inline DDL + positional-tuple coupling (F-024, F-025). **Backup all live DBs before migration** (standing rule).
- [ ] Run identity: every pipeline invocation gets a run_id; stage results, yields, and manifests recorded against it (F-030, F-044, R-20). `.done` markers retire.
- [ ] Notifications become an **outbox table** (queued → sent → failed with retry count) — completes the M1 watermark fix into a replayable queue.
- [ ] Resume-by-job-id for LLM analysis (kills index-based resume against regenerated input — F-020); error rows re-drivable.
- [ ] Operator CLI: `scanner status | run | retry-failed | replay-notifications` — converts SSH-and-grep operations into commands (F-036's admin half, R-12).
- [ ] Logging consolidation: one logging config, structured per-run log dir, retention sweep covering everything cleanup currently misses (F-034); disk preflight `df` check + cache retention (F-043, F-035 remnant).
- [ ] Packaging: `pyproject.toml`, absolute-import package, no CWD-relative paths, no module-level side effects (F-032); personal data/tunables out of code into config (F-042).
- [ ] Overlap test for SQLite lock behavior (closes H-01).
- [ ] Optional after everything above: Prefect adoption for run history/UI — only now is it worth it (per audit: orchestrators adopted before exit-code truth just paint failures green). Decide at the time; systemd may remain sufficient at this scale.

## M6 — Research track (parallel, time-boxed; never blocks M2–M4)

**Spike A — Canary attention probe (2–3 days).** Sid's design: three canaries with fresh random values per prompt (e.g. 6-hex nonces) at top / middle / bottom of the prompt — and variant: start / middle / end of the document context — with the instruction to echo top-canary at the top of output, middle mid-output, bottom last.
- Measure per-position recovery rate over N runs × context lengths × models.
- Read-out logic: middle-canary recovery dropping as context grows ⇒ lost-in-the-middle attention problem; all positions failing uniformly ⇒ prompt/guideline problem; failing even at short context ⇒ model problem. Cheap, discriminating, worth building.
- Two cautions to design in: (1) the canary instructions themselves consume attention and can perturb output format — run each config with/without canaries and compare extraction F1 to measure the probe's own cost; (2) canary echo measures positional retrieval + instruction-following, not extraction quality — always report it alongside golden-set F1, never instead of it.
- Deliverable: harness + one-page report; if middle-loss is real, mitigation = smaller batches / restructured prompts (key info at edges), measured again.

**Spike B — BDH (time-boxed 2 days, expectations set).** Verified: research architecture, no pretrained weights, GPT-2-scale, toy `train.py`. It cannot replace a generic LLM for extraction any time soon. The **honest version of the instinct** ("our patterns repeat, a generic LLM is overkill") is: LoRA-fine-tune a 1–3B model on accumulated (batch-text → extracted-jobs-JSON) pairs from our own DB once M4 has produced a few weeks of data. That path keeps the insight and drops the from-scratch training. BDH stays a curiosity spike: run `train.py` on toy data, read the paper, write half a page on whether its sparse-activation ideas matter for us. Not on any critical path.

**Spike C — KnownPatch prompting eval.** Feeds M2's hallucination metric: schema-anchored prompts, explicit `null`-if-missing, known-example grounding; measure invented-field rate with/without (per `docs/knowledge_induced_hallucinations.md`).

**Spike D — Source diversification probe.** epaperwave `wp-json`/feed endpoints (closes H-04), Telegram-first ingestion for papers that have channels, official/RSS sources for a subset — chips at the single-aggregator risk (F-026) and shrinks the 2,821-line heuristic cascade (F-027). Full diversification is post-M5 work; the probe is cheap now.

## M7 — Super-project integration (job-discovery-engine)

Goal: V2 becomes a clean upstream component of the hiring agent.

- [ ] **Contract, not coupling:** V2 emits versioned JSON job records (schema frozen at M4: company, role, requirements, contact, source, confidence, run_id) to an agreed handoff (directory or shared SQLite table). The super project consumes; persona-context-engine FAISS matching and Dual-Merlin generation stay on its side, LangGraph stays the blueprint (per `docs/blueprint_first_deterministic_llm.md`).
- [ ] **The 32k KV question, answered by the M3 method, not by vibes:** instrument the hiring agent's real p95 (prompt + completion) per task; required `n_ctx = p95 × 1.25`. If p95 ≤ ~25k tokens, 32k is enough; if resume+JD+persona context pushes past that, raise ctx rather than trimming into the degraded tail — with `turbo3` KV quant, going 32k → 48k on a 7–8B GQA model costs only a few hundred MB, so headroom is cheap. Build the actual table during integration; don't guess.
- [ ] Shared runtime option: one `llama-cpp-turboquant` systemd service serving both projects (separate model slots), so params/logging hygiene from M3 applies to the hiring agent for free.
- [ ] Same truth law applies across the boundary: the hiring agent must be able to distinguish "V2 ran and found nothing" from "V2 failed" — the run manifest travels with the records.

---

## Appendix A — Finding-by-finding disposition (all 53)

Legend: **fix** = direct remediation · **obsoleted** = component replaced/deleted, finding dies with it · **decision** = resolved by an explicit Sid decision · **tradeoff** = consciously accepted · **spike** = research/verification task.

| ID | Sev | Milestone | Disposition |
|---|---|---|---|
| F-001 venv missing | crit | M0 | decision (deleted deliberately) + fix: lean `.venv`, no hardcoded interpreter paths |
| F-002 stages exit 0 on failure | crit | M1 | fix — stage contract + `stage_guard`, the design law |
| F-003 silent no-op nights | crit | M1 | fix — zero-yield exits + Telegram failure/digest hook |
| F-004 homemade bash scheduler | high | M4 | obsoleted — systemd timers, scheduler.sh deleted |
| F-005 no instance locking | high | M1 | fix — flock |
| F-006 fake zero metrics | high | M1 | fix — real counts or deletion |
| F-007 three orchestrators, one drifted | med | M1 | fix — delete pipeline_runner.py |
| F-008 watermark loses failed sends | high | M1→M5 | fix — sent_at now, outbox table at M5 |
| F-009 unescaped URL in Telegram HTML | med | M1 | fix |
| F-010 daily live test message | low | M1 | fix — delete |
| F-011 no retry_after handling | med | M1 | fix |
| F-012 sent_jobs written pre-send | med | M1→M5 | fix — semantics now, outbox at M5 |
| F-013 cleanup deletes after failed runs | high | M1 | fix — yield-gated, archive-then-prune |
| F-014 failed OCR blocks never retried | med | M2 | obsoleted — stage04 replaced by VLM |
| F-015 two progress-JSON writers | med | M2 | obsoleted |
| F-016 write-only task_backlog.txt | high | M1 | fix now (live until M4), then obsoleted |
| F-017 800-line homemade GPU executor | high | M2 | obsoleted — do not invest further |
| F-018 stage07 owns Ollama incl. pkill | med | M3 | obsoleted — runtime becomes systemd service |
| F-019 three overlapping resume systems | med | M3/M5 | obsoleted by new extraction + run-state |
| F-020 index-based Gemini resume | med | M5 | fix — resume by job_id |
| F-021 genai.configure thread race | med | M3 | fix — lock/per-client keys |
| F-022 homemade rate-limit engine | med | M3 | tradeoff — keep, consolidate |
| F-023 fully synchronous nightly | med | M4 | tradeoff — accepted for MVP volumes |
| F-024 4-DB SQLite sprawl, no migrations | high | M5 | fix — scanner.db + storage.py + user_version |
| F-025 duplicated SQL, positional tuples | med | M5 | fix — storage.py |
| F-026 single-aggregator dependency | high | M6-D→post-M5 | spike now, diversification after MVP |
| F-027 2,821-line heuristic downloader | high | M1/M6-D | fix shipped debug config now; shrink via source diversification |
| F-028 downloader failures invisible | high | M1 | fix — honest exits + expected-count check |
| F-029 secrets disk-loose | med | M0 | fix — relocate to ~/.config |
| F-030 9+ state mechanisms, no run identity | high | M5 | fix — run_id + run-state tables |
| F-031 no inter-stage manifests | high | M4 | fix — manifests in new DAG |
| F-032 no packaging, CWD paths, import side effects | med | M5 | fix — pyproject, package layout |
| F-033 dead scaffolding (utils, orphan DB) | low | M0 | fix — delete |
| F-034 log sprawl, 888 files | med | M5 | fix — logging consolidation + retention |
| F-035 unbounded write-only cache_blocks | med | M2/M5 | obsoleted with stage04; retention sweep at M5 |
| F-036 no health interface/alerting/admin | high | M1/M5 | fix — Telegram hook now, operator CLI at M5 |
| F-037 structural-only tests | med | M1+ | fix — regression test per fix, ongoing law |
| F-038 scheduler bash fragility | med | M4 | obsoleted — deleted with scheduler.sh |
| F-039 MAX_RETRIES=0 = infinite | low | M1 | fix |
| F-040 stage03 silent discards | low | M2 | obsoleted |
| F-041 O(n²) prose dedup | low | M4 | obsoleted — structured-JSON key dedup |
| F-042 personal data hardcoded | med | M5 | fix — config |
| F-043 no disk-space awareness | med | M5 | fix — preflight + retention (30 GB data/ already gone by decision) |
| F-044 .done keyed to command string | med | M1→M5 | fix — date-scoped now, run_id at M5 |
| F-045 POSITIVE git secrets hygiene | info | all | preserve — carry into new code |
| F-046 POSITIVE atomic writes | info | M2/M4 | preserve — requirement on new stages |
| F-047 POSITIVE validation scaffolding | info | M4 | preserve — dry-run/health/CI survive the rebuild |
| F-048 POSITIVE hard-problem pockets | info | M3/M5 | preserve — WAL, key rotation, PDF validation patterns |
| H-01 SQLite lock contention | med | M5 | spike — overlap test during storage migration |
| H-02 tee misordering | low | M1 | spike — verify during runner rework |
| H-03 Ollama num_parallel | info | M3 | spike — bench on new runtime |
| H-04 epaperwave wp-json | info | M6-D | spike — probe |
| H-05 CI green? | low | M0 | spike — check once online |

## Appendix B — Sequencing rationale

1. **M1 before M2/M3:** truth-first is the audit's core conclusion — new models on a lying pipeline produce prettier silent failures.
2. **Benchmark gate before any VLM commitment:** SmolVLM being English-only was caught by a 5-minute verification; the golden set + MDPBench Hindi subset is what prevents shipping a model that can't read half the papers.
3. **MVP (M4) before state consolidation (M5):** cutover kills the Docker legacy system's live defects weeks earlier; storage refactor is safer against a running, observable pipeline.
4. **Research spikes fenced in M6:** BDH and canaries are genuinely interesting and genuinely not load-bearing; time-boxing keeps them from eating the MVP.
