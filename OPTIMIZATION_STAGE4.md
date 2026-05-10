# Stage 4 Optimization: Candidate Scorer

## The Goal
Reduce False Positives by >90% while strictly preserving 100% of Stage 3 True Positives.

## Insights & Breakthroughs (v17.3)
- **Zero-Text Ads**: Discovered that fragmented job ads can occasionally yield 0 text in OCR but have high detector confidence.
- **Decision Rule**: Soft-rejection is no longer used; Stage 4 uses a **Strict Gate** where candidates must have a "High Signal" tag or valid detector anchor to pass.
- **FP Target**: Reducing S3 candidates from 22,000 down to < 2,200 per PDF.
