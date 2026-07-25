# data/samples/pair_001 -- provenance

**Source document.** `pid_a.pdf` is the P&ID supplied with the take-home
assignment (`Lift_Gas_compressor-P_ID.pdf`, a 3rd Stage HP Gas Lift
Compressor P&ID), used unmodified as Rev A / PID A. It is a born-digital,
single-page PDF with an extractable text layer (~2,000 words with bounding
boxes) -- ingested by `src/ingest/pdf_native.py`.

**Synthesized revision.** No second revision of this document existed, so
`pid_b.pdf` (Rev B / PID B) was generated deliberately by
`data/samples/pair_001/make_pid_b.py`, which:

1. Rasterizes `pid_a.pdf` at 200 DPI (`pdf2image` / poppler) -- modeling the
   assignment's own "a scanned as-built supersedes a drawing" scenario.
2. Burns four documented edits directly into the pixels:
   - **Modified** -- PSV 9066A relief set pressure: `257 bar (g)` -> `262 bar (g)`
   - **Modified** -- TIT-9211 balance-line-cooler high alarm: `H:145` -> `H:150`
   - **Removed** -- instrument tag `26GT9281` (a spare/duplicate gauge tag)
   - **Added** -- new margin note: `37. TEMP BYPASS INSTALLED FOR
     COMMISSIONING; REMOVE PRIOR TO PSV RE-CERT.`
3. Saves the result as an image-only PDF (no text layer) -- ingested by
   `src/ingest/pdf_scanned.py` via OCR.

These four edits are the exact contents of
`eval/datasets/pair_001/ground_truth_delta.json`. Because the edits are
known exactly (we made them), this sample pair measures "did the pipeline
recover edits we know are there," not independent human-labeled ground
truth agreement -- see `eval/metrics.py`'s docstring for why that matters
for how precision/recall should be read.

**Why rasterize-and-burn-in instead of editing the PDF text layer
directly?** An earlier attempt edited `pid_a.pdf`'s content stream directly
(white-out rectangle + redrawn text). It looked correct visually but
failed for the actual purpose: PDF text extraction reads all text objects
in the content stream regardless of z-order, so the "hidden" original text
was still recovered by `pdfplumber` alongside the new text, corrupting the
native-adapter ingestion. Rasterizing first and burning in edits avoids
that entirely -- the edited pixels are the only thing there -- and it has
the added benefit of giving the pipeline a genuine scanned/OCR sample to
exercise, rather than needing a second unrelated sample for that.

**Known limitation of this sample, reported candidly.** Tesseract OCR on
this specific drawing (dense line-art, small font, technical abbreviations)
is noisy even after the confidence/junk-token filtering in
`src/ingest/pdf_scanned.py`. The delta engine's "needs review" bucket and
the eval harness's failure table surface this directly rather than hiding
it -- see `README.md`'s "Honest results" section.
