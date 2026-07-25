# Demo walkthrough

All commands below were actually run to produce this output (not
hand-written) -- re-run them yourself with `make run`, `make ask`, etc.

## 1. Ingest + delta

```
$ make run
```

Ingests `data/samples/pair_001/pid_a.pdf` (native PDF, 2012 text items) and
`pid_b.pdf` (scanned/OCR'd revision, ~900 text items after noise
filtering), aligns them, and writes `out/pair_001/delta_report.md` and
`.json`. Top of the report:

```
# Delta Report: PID-A -> PID-B

## Summary

- Total changes: 714
- Added: 1
- Removed: 636
- Modified: 77
```

(See README "Honest results" for why "714 detected vs. 4 labeled" is
expected and documented on this OCR'd sample, not a bug.)

Signal from the four deliberately-injected edits (see
`data/samples/pair_001/PROVENANCE.md`) shows up in the report, with real
strengths and weaknesses worth being upfront about:

```
- [d0540] (tag, removed, confidence 1.00) Tag removed: "26GT9281"
    -> clean hit: exactly the removed tag, nothing else nearby to confuse it.

- [d0014] (dimension, removed, confidence 1.00) "257" removed
    -> correctly flags that 257 is gone, but the OCR read of the new "262"
       on the scanned side had confidence ~19/100 (below the 40 min-confidence
       filter in src/ingest/pdf_scanned.py) and was dropped, so this shows
       as a removal rather than a clean 257->262 modification -- a real,
       reported limitation, see README "Honest results".

- [d0802] (note, modified, confidence 0.60) "TP DSS NOTE" -> "TEMP BYPASS INSTALLED"
    -> the new margin note was found, but matched against unrelated nearby
       note text instead of reported as a clean addition. See README.
```

This is exactly the kind of output the eval harness (`make eval`) is meant
to catch and score honestly rather than a hand-picked "everything worked"
example -- see README "Honest results" for the full read.

## 2. Grounded chat

```
$ make ask Q="Was any instrument tag removed in the revised drawing?"

Q: Was any instrument tag removed in the revised drawing?
A: Based on the retrieved context: [delta_report:d0651] (delta_report, page 1)
   [REMOVED] page 1, tag: Tag removed: "26-KA-901" (confidence 1.00)
   [delta_report:d0006] ... Tag removed: "26-PY-9077B" (confidence 1.00) ...
Citations: ['delta_report:d0651', 'delta_report:d0006', 'delta_report:d0031']
Groundedness: 0.56  Cost: $0.00000  Latency: 0.3ms  Provider/model: mock/mock
```

Every answer carries citations back to a specific chunk id
(`pid_a:pX:cN`, `pid_b:pX:cN`, or `delta_report:dNNNN`) so a claim can be
traced to a page + location or a specific delta finding -- but note this
particular real answer is a good example of the retrieval-precision
weakness called out in README "Honest results": it surfaces *other* real
tag removals from the same noisy scanned pair instead of the specific
`26GT9281` removal, because those other entries also literally contain the
word "tag" and rank competitively under TF-IDF. The citations are still
accurate (each really is a tag removal in the delta), just not the most
relevant one -- a precision problem, not a hallucination.

## 3. Markup overlay (bonus)

```
$ make markup
Wrote markup:
  out/pair_001/markup/markup_page1.png
  out/pair_001/markup/markup.pdf
```

Colored boxes on PID B: red = removed, green = added, orange = modified,
each labeled with its delta entry id, plus a legend.

## 4. Eval

```
$ make eval
```

Prints a scorecard (delta precision/recall/F1, chat keyword recall /
citation accuracy / groundedness) and writes
`out/eval/eval_scorecard.json`. See README "Honest results" for the actual
numbers and what they mean.

## 5. Observability

Every run writes to `observability_runs/`:

```
$ ls observability_runs/
6f8ef751f7c9.log.jsonl
6f8ef751f7c9.trace.json
```

`*.trace.json` has per-stage timings (ingest_pid_a, ingest_pid_b, delta,
report, retrieval, llm_call) with item counts, token counts, and cost.
`*.log.jsonl` is one structured JSON object per log line, all tagged with
the same `request_id` for correlation.
