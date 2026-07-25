# delta-chat

Ingests two revisions of an engineering drawing (native PDF, scanned PDF,
or DWG), computes a structured delta between them, and answers grounded
questions about either revision or the delta via retrieval + an LLM.

AI-assisted coding was used throughout, per the assignment's own note that
this is expected. Every design choice below is one I can defend.

## Quickstart

```bash
make setup                 # pip install core deps (no LLM deps required)
make run                   # ingest sample pair -> compute delta -> write report
cat out/pair_001/delta_report.md
make ask Q="What changed near PSV 9066A?"
make chat                  # interactive REPL instead of one-shot
make markup                # bonus: redline overlay onto PID B
make eval                  # delta P/R/F1 + chat correctness/groundedness scorecard
make test                  # fast unit tests (no PDFs/OCR involved)
```

Everything above runs fully offline with zero API keys, using the default
`LLM_PROVIDER=mock`. See "Running with a real LLM" below to switch to an
open-source local model or hosted Claude.

## Architecture

```
PIDRef (path + revision label)
        |
        v
FormatAdapter.sniff() / .ingest()      <- src/ingest/{base,pdf_native,pdf_scanned,dwg}.py
        |
        v
CanonicalDocument (format-agnostic)     <- src/canonical/model.py
        |
        v
align_documents()                       <- src/delta/align.py   (the hard part)
        |
        v
compute_delta() -> DeltaEntry[]         <- src/delta/engine.py  (deterministic)
        |
        v
DeltaReport (.md + .json)               <- src/delta/report.py
        |
        v
RetrievalIndex (TF-IDF over PID A/B     <- src/chat/index.py
  + delta report chunks)
        |
        v
answer_question()                       <- src/chat/answer.py
  (retrieve -> prompt -> LLMProvider -> cite)
        |
        v
LLMProvider: mock | hf_local | anthropic  <- src/chat/llm.py
```

Every stage is wrapped in a `Trace` span and logs structured JSON with a
request id (`src/observability/`), written under `observability_runs/`.

### The canonical representation

`CanonicalDocument` / `CanonicalItem` (`src/canonical/model.py`) is the one
seam every format normalizes into: a flat list of typed, located, page-
scoped items (`text` / `note` / `tag` / `dimension` / `table_cell` /
`geometry`), each with a bounding box in PDF points (top-down origin) and
an extraction confidence. Nothing downstream ever imports a format-
specific adapter -- the delta engine, the report, and the retrieval index
only ever see `CanonicalDocument`.

### Formats implemented

| Format | Adapter | How |
|---|---|---|
| Native PDF | `src/ingest/pdf_native.py` | `pdfplumber` word-level text + bboxes, extraction_confidence=1.0 |
| Scanned PDF / images | `src/ingest/pdf_scanned.py` | rasterize (`pdf2image`/poppler) + OCR (`pytesseract`), per-word confidence, noise-filtered |
| DWG/DXF | `src/ingest/dwg.py` | **real stub** -- registered, satisfies the interface, `sniff()` correctly claims `.dwg`/`.dxf`, `ingest()` raises a documented `NotImplementedError` explaining the intended `ezdxf`-based design |

DWG was cut deliberately (no sample available, and a real implementation
needs an external DWG->DXF converter binary) in favor of spending the time
budget on alignment, grounding, and eval -- see the module docstring in
`dwg.py` for the intended design if it were built out.

### Alignment -- the hard part

`src/delta/align.py` does greedy nearest-match alignment per page:
`score = 0.45 * position_similarity + 0.55 * text_similarity`, where
position similarity is bbox-center distance normalized by page diagonal
(robust to scan-vs-native size differences) and text similarity is
`difflib.SequenceMatcher` ratio (tolerant of OCR noise). Matches below
`MATCH_THRESHOLD=0.5` aren't matches at all -- both sides become
independent add/remove rather than a forced pairing. This is greedy, not
globally optimal (no Hungarian algorithm) -- a documented, revisitable
trade-off; see the docstring in `align.py`.

### Delta engine -- deterministic by design

`src/delta/engine.py` turns alignment output into typed `DeltaEntry`
records (added/removed/modified) with a confidence that combines match
confidence and extraction confidence. **No LLM call happens here** -- this
is the one part of the assignment that explicitly needs to be
deterministic run-to-run, so non-determinism is isolated entirely to the
chat layer's prose generation (`src/chat/llm.py`).

Entries below `REVIEW_CONFIDENCE_THRESHOLD=0.55` are bucketed into
`needs_review` instead of presented as confident findings -- see "Honest
results" below for why that bucket matters a lot on the scanned sample.

### Retrieval + chat

`src/chat/index.py` is a TF-IDF index (scikit-learn) over three chunk
sources: PID A, PID B, and the delta report (one chunk per delta entry).
TF-IDF instead of an embeddings API is a deliberate choice: P&ID text is
dominated by near-unique identifiers (`26-PIT-9055`, `PSV 9066A`) that
lexical matching handles very well, and it needs no network call or API
key. `src/chat/answer.py` retrieves top-k chunks, builds a context-only
prompt, calls the configured `LLMProvider`, and extracts citations back to
chunk ids -- falling back to citing the top retrieved chunks explicitly if
the model didn't emit bracketed citations, so an answer is never presented
as uncited when grounded context existed.

### Markup overlay (bonus)

`src/markup/overlay.py` rasterizes PID B and draws colored boxes
(green=added, red=removed, orange=modified) with entry-id labels and a
legend. Raster-only by design (works identically regardless of PID B's
source format); a fuller version would draw vector annotations directly
into a native PDF's content stream -- left as documented future work.

### Observability

`src/observability/logging.py` is a ~60-line JSON logger (stdlib
`logging` + a JSON formatter), not a framework dependency -- the actual
requirement ("structured JSON logs with a correlation id") doesn't need
more than that. `src/observability/tracing.py` is a minimal homegrown
tracer with `Trace`/`Span` shapes deliberately close to OpenTelemetry's, so
swapping in real OTel spans later (if this became a served API instead of
a CLI) would be mostly mechanical. Every `run.py pipeline` / `chat` call
writes a `<request_id>.trace.json` (per-stage timings, item counts, token
counts, cost) and a `<request_id>.log.jsonl` under `observability_runs/`.

## Running with a real LLM

Default is `LLM_PROVIDER=mock` (offline, deterministic, zero cost) so
everything above works with no setup. Two real providers are implemented
behind the same `LLMProvider` interface (`src/chat/llm.py`):

**Open-source local model (Gemma / Phi-4-mini / Qwen, etc.):**

```bash
make setup-local-llm                       # pip install torch/transformers/accelerate
export LLM_PROVIDER=hf_local
export LLM_MODEL_NAME=google/gemma-2-2b-it # or microsoft/Phi-4-mini-instruct, Qwen/Qwen2.5-3B-Instruct
make chat
```

Note: "Phi-4" itself is a 14B model; `Phi-4-mini-instruct` (~3.8B) is the
one that actually fits a "4B-class local model" budget on a single
consumer GPU or CPU, so that's what `.env.example` points at alongside
Gemma-2-2B. The first call downloads weights from the Hugging Face Hub
(needs network + disk once); inference after that is fully local. This
provider is real, complete code but **not exercised in this sandbox**
(no GPU/network available here) -- reviewed and tested by inspection and
by running the equivalent `mock` code path through the exact same
`answer_question()` function.

**Hosted Claude:**

```bash
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-...
export LLM_MODEL_NAME=claude-sonnet-5
make chat
```

## Honest results (`make eval`)

Run against `eval/datasets/pair_001` (see
`data/samples/pair_001/PROVENANCE.md` for exactly what's in this pair: 4
deliberately-applied, documented edits -- a PSV setpoint change, an alarm
setpoint change, a tag removal, and a note addition -- burned into a
rasterized/OCR'd Rev B):

```
Delta   precision=0.004  recall=0.750  f1=0.008   (detected=714, ground_truth=4)
        MISSED ground truth: ['gt_04']
Chat    mean_keyword_recall=0.100  citation_source_accuracy=0.800  mean_groundedness=0.725
```

**Read this candidly, not as a headline number:**

- **Recall 0.75** -- 3 of 4 known edits were found (the PSV setpoint
  change, the alarm setpoint change, and the tag removal). The missed one
  (`gt_04`, the added note) was *partially* detected -- its words show up
  as several low-confidence `modified` entries against unrelated nearby
  note text rather than as one clean `added` entry -- because the greedy
  aligner paired new note words with spatially-nearby pre-existing note
  text instead of leaving them unmatched. This is a real, known alignment
  weakness on dense text blocks, not a scoring artifact.
- **Precision 0.004** looks alarming and is meant to: it's computed
  against the *full* high-confidence detected set (714 entries) vs. 4
  labeled ground-truth edits, on a *scanned/OCR* revision of a dense,
  small-font engineering drawing. Plain Tesseract without image
  preprocessing (deskew/threshold/upscale) or a P&ID-tuned OCR model still
  produces a meaningful amount of noise even after the confidence and
  junk-token filtering in `src/ingest/pdf_scanned.py`. On a native-PDF-vs-
  native-PDF pair (no OCR involved) the same code scores far better -- see
  the unit tests in `tests/test_delta_engine.py`, which cover exactly this
  class of case deterministically. This is the single biggest thing I'd
  fix next with more time: real OCR preprocessing, or a P&ID-specific OCR
  model, before the delta engine ever sees the tokens.
- **Chat keyword recall 0.10** -- inspecting the failing answers
  (`out/eval/eval_scorecard.json`) shows the *citation source* is usually
  right (0.80 accuracy: retrieval correctly favors `delta_report` chunks
  for delta questions) but TF-IDF sometimes ranks a noisy single-word
  delta chunk (e.g. "Text removed: 'PSV'") above the substantive one
  (e.g. "PSV 9066A relief set pressure changed from 257 to 262") because
  they share exact-match terms and the noisy chunk is shorter/denser.
  Fix: de-duplicate or down-weight very short, low-information delta
  chunks, or merge them into their parent entry before indexing --
  scoped out of this submission's time budget, noted here rather than
  hidden.
- The `mock` LLM provider used for these numbers doesn't generate free
  prose, it echoes retrieved snippets -- so `mean_keyword_recall` is
  really scoring **retrieval quality**, not answer-generation quality.
  That's intentional (see `chat/llm.py`'s docstring on why `mock` is the
  eval default), but it means these chat numbers are a lower bound: a real
  model (`hf_local` or `anthropic`) sees the same retrieved context and
  would very likely phrase a correct answer even when the ideal chunk
  wasn't top-ranked, as long as it's anywhere in the top-k.

Re-running `make eval` after any change to `src/ingest/pdf_scanned.py`,
`src/delta/align.py`, or `src/chat/index.py` is the way to check whether a
change actually helped, not just eyeballing the delta report.

## What I'd do with another day

1. OCR preprocessing (deskew, adaptive threshold, upscale) or a P&ID-tuned
   OCR model -- this is the single highest-leverage fix, per the eval
   results above.
2. De-duplicate/merge short, low-information delta chunks before indexing
   for retrieval, to fix the chat precision issue identified above.
3. Real DWG ingestion (`ezdxf` + a DWG->DXF conversion step).
4. Swap greedy alignment for `scipy.optimize.linear_sum_assignment` and
   compare precision/recall on a larger, independently-labeled eval set
   (pair_001's ground truth is "recover edits we made," not
   inter-annotator agreement -- a real second labeled pair would be more
   convincing).
5. LLM-as-judge for chat correctness, validated against a small human-
   labeled sample first, instead of the current keyword-recall proxy.

## Repository layout

```
run.py                        CLI: pipeline | chat | markup
src/canonical/model.py        CanonicalDocument / CanonicalItem
src/ingest/                   base.py + pdf_native.py + pdf_scanned.py + dwg.py
src/delta/                    align.py + engine.py + report.py
src/chat/                     index.py + llm.py + answer.py
src/markup/overlay.py         bonus redline overlay
src/observability/            logging.py + tracing.py
eval/                         metrics.py + run_eval.py + datasets/pair_001/
data/samples/pair_001/        pid_a.pdf, pid_b.pdf, make_pid_b.py, PROVENANCE.md
tests/test_delta_engine.py    fast unit tests (no PDFs/OCR)
Makefile, requirements*.txt, .env.example
```
