# AirSketch — Deliverables README (start here in a future session)

This `docs/` set lets a future Claude session, given **this repository**, produce two
deliverables about AirSketch: a **detailed academic presentation** and an **academic
book**. Nothing here needs the internet beyond optional citation lookups.

## What to read, for which deliverable

| You want to build… | Read these (in order) |
|---|---|
| **Academic presentation** (.pptx) | `PROJECT_COMPENDIUM.md` → `PRESENTATION_GUIDE.md` |
| **Academic book** (.docx / LaTeX / MD) | `PROJECT_COMPENDIUM.md` → `BOOK_GUIDE.md` |
| **Just understand the project** | `PROJECT_COMPENDIUM.md`, then `../SESSION_HANDOFF.md` |

- **`PROJECT_COMPENDIUM.md`** — the single source of truth for *content* (abstract,
  motivation, architecture, the OpenVINO hand-tracking contribution with its math and
  bugs, every component, Hebrew support, packaging, lessons, tests, results, future
  work, glossary, timeline, and a "what's new since the prior deck" list).
- **`PRESENTATION_GUIDE.md`** — *how* to build the slides (audience, style, a 16–22
  slide outline mapped to Compendium sections, figures, placeholders, build steps).
- **`BOOK_GUIDE.md`** — *how* to write the book (chapter outline mapped to Compendium
  sections, format options, references, figures, process).

## Supporting ground-truth (in the repo root, also authoritative)

- `../SESSION_HANDOFF.md` — full current status + working stance (most up-to-date).
- `../HANDOFF.md` — terse code/module map + gotchas.
- `../OPENVINO_HAND_TRACKING_PLAN.md` — the hand-tracking design + implementation notes.
- `../LAUNCHER.md` — the GUI launcher + exe build.
- `../AirSketch_MidProject_v2.pptx` — the **prior** presentation (the new one must be
  more detailed and include everything new since it; see Compendium §16).
- Source under `../src/airsketch/` — the implementation, for any depth a chapter/slide needs.
- `intel_npu_setup.md`, `model_download.md` — environment/model notes (this folder).

## First steps for the future session

1. Read `PROJECT_COMPENDIUM.md` end to end.
2. Skim `../SESSION_HANDOFF.md` for any status that changed after this file was written.
3. Confirm the open placeholders with the user (course/team/affiliation; slide &
   speaker-note language — prior deck used **Hebrew speaker notes**; output format for
   the book; target Intel SKU; neutral vs Check Point theme).
4. Generate figures by running code where useful (e.g. landmark overlays, Hebrew-UI
   samples) — these were produced during development by rendering to PNG.
5. Build the deliverable per the relevant guide; keep the verified/to-be-verified honesty.

## Honesty guardrails (carry into both deliverables)

- "All seven model components run on OpenVINO" is correct; avoid implying GPU/NPU were
  benchmarked (CPU was measured; GPU/NPU = future work).
- Hand-tracking CPU FPS (~108), CNN accuracy (96%), and test count (175) are verified.
- Hebrew board **OCR** is deferred (no OpenVINO Hebrew OCR model) — don't imply it works.
- Don't fabricate citation details; mark unverified ones `[citation needed]`.
