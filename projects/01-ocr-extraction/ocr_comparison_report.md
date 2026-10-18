# OCR Engine Comparison Report
**Document:** pharmaceutical-sdf-page3-certificate-quality.pdf (Cytiva Certificate of Quality, ÄKTA™ ready Gradient Flow Section With Inlets, Lot 17242818)
**Engines compared:** Tesseract, EasyOCR, PaddleOCR
**Author:** Bashar

## Summary of results

| Engine | Status | Time | Word/region boxes | Notes |
|---|---|---|---|---|
| Tesseract | Success | 8.33s | 220 | Word-level boxes, no transcription errors found, recovered rotated stamp |
| EasyOCR | Success | 1.64s init + 22.11s run (~23.75s) | 47 | Line/region-level boxes, several transcription errors, missed rotated stamp |
| PaddleOCR | Failed to initialize | — | — | Blocked on first-run model download (network policy in this environment) |

## Which OCR tool worked best, and why

Tesseract produced the cleanest output of the two engines that actually ran. Its raw text matched the source document exactly, including special characters that EasyOCR got wrong (correct "121°C" vs EasyOCR's "121*C", correct "29-1846-12" vs EasyOCR's line-fragmented version, correct "EMA/410/01" vs EasyOCR's "EMA/410/O1" with a letter O substituted for the zero). Tesseract also ran roughly 3x faster and returned word-level bounding boxes (220 of them) instead of EasyOCR's coarser line/region-level boxes (47), which matters a lot for building label-anchored field extraction downstream — you need individual word positions to reliably find "Lot Number:" and then the value sitting immediately to its right. Tesseract was also the only engine of the two that recovered the rotated approval stamp ("090177e19abdfceb\Approved\Approved On: 13-Jul-2022 20:47 (GMT)") baked into the page image; EasyOCR produced zero detections in that region.

Key fields specifically requested by the task — Lot Number (17242818), Product Article Number (29-1846-12), and Expiration Date (20230126) — were captured correctly by both Tesseract and EasyOCR. The two engines only diverged on harder cases: symbols, mixed alphanumeric IDs, and the rotated stamp. There's no vendor name field on this particular certificate (it's issued by Cytiva, printed clearly in the footer, and both engines got that plain text right); the real differentiator was everything with a special character or unusual orientation.

## What surprised me

Two things stood out. First, OCR can recover content that native PDF text extraction structurally cannot see. The rotated stamp is not real text in the PDF — it's pixels in a rasterized image — so PyMuPDF's `get_text()` returns nothing for that region no matter how you query it, but Tesseract reads it just fine as a tall, narrow line of text. That's a real capability gap between "extract the text layer" and "OCR the rendered page" that isn't obvious until you compare them side by side.

Second, EasyOCR — a modern deep-learning model — made mistakes that a much older, simpler engine like Tesseract didn't: misreading a degree symbol as an asterisk, a zero as a capital O, and turning "This product is manufactured" into outright gibberish ("CbuiroatCris imanstactured"). Newer/fancier doesn't automatically mean more accurate on this kind of clean, printed pharmaceutical document — Tesseract's classical approach seems to be a very good fit for crisp, standard-font printed text specifically.

## Which engine feels best suited to pharmaceutical compliance workflows

Tesseract, for this document class. Three reasons: (1) it preserved regulatory-citation formatting and units exactly (EMA/410/01, °C, hyphenated part numbers) where accuracy on a single wrong character could matter downstream; (2) it recovered the approval stamp with its timestamp, which is exactly the kind of audit-trail metadata a compliance workflow cares about; and (3) it runs entirely offline with no runtime model download, which matters in a regulated or locked-down IT environment where every external dependency needs its own validation and network access may not be available at all — see the PaddleOCR failure below.

## Frustrations with install/setup or output structure

PaddleOCR never got past initialization in this environment: on first run it needs to download pretrained model weights from an external host (HuggingFace, ModelScope, Baidu AIStudio, or Baidu's BOS object storage), and all four were blocked by this environment's network policy. I confirmed this wasn't a config mistake on my end — direct connection attempts to each host returned explicit connection-refused/policy-denial responses. This is a real production consideration: in a validated or air-gapped pharma IT environment, PaddleOCR's models would need to be pre-downloaded and vendored into the approved environment before it could ever run, which is extra setup and validation overhead that Tesseract (a self-contained package with no runtime downloads) simply doesn't have. In an unrestricted environment such as a normal Colab session, this download would very likely succeed — this failure is specific to this sandbox's network policy, not a fundamental flaw in PaddleOCR.

On the output-structure side, EasyOCR's coarser line/region-level bounding boxes were noticeably more annoying to work with than Tesseract's word-level output — several label/value pairs that Tesseract kept on one clean line got fragmented across multiple EasyOCR lines (for example "AKTA™" split into separate "AKTA" and "TM" entries), and the Gamma Irradiation limits table read out of visual order (40.0/25.0/kGy landed on non-adjacent lines with unrelated content interleaved). That fragmentation would make label-anchored field extraction meaningfully harder to build reliably on EasyOCR's output compared to Tesseract's.

## Would I combine tools or pick one for production?

I'd combine them, but not evenly. Tesseract would be the default/primary engine for this document type — fast, word-level, fully offline, and accurate on clean printed pharmaceutical certificates. I'd reserve a second engine (EasyOCR or, network permitting, PaddleOCR) as a fallback pass specifically triggered by low Tesseract confidence scores or known problem regions, since deep-learning engines can sometimes do better on noisy scans, handwriting, or unusual fonts that a classical engine struggles with. I would not drop Tesseract even after adopting a fancier model, though — this comparison shows it can outperform a newer deep-learning engine on exactly the kind of crisp, standard-font documents that make up most of a compliance archive, and it's the only one of the three with zero external dependencies to validate.
