# Correction evidence matrix (working draft)

**Status:** preliminary audit; not submitted to journal. Exact printed page/figure locations and underlying workbook values must be checked against the version of record before submission.

| Published item | Original implementation / statement | Audit evidence | Recommended correction | Status |
|---|---|---|---|---|
| Core CMS category | $372.1B, about 7% of $5.2786T in 2024 | Archived NHE24 summary: 372.1 / 5278.6 = 7.0492% | Retain number; name category precisely as government administration and non-medical insurance | Verified |
| 1990–2024 nominal growth | Approximately 872% | 1990 $38.3B; 2024 $372.1B; growth 871.54% | Retain, explicitly label *nominal* growth | Verified |
| Figure 8 / reported r | Paper r=0.998; script fits 1980–2024 nominal levels | Archived summary 1980–2024 r=0.994673 | Reconcile version/data and then correct reported value if warranted; distinguish descriptive correlation from causation | Discrepancy verified; final correction pending |
| Figures 3 and 9 | Captions describe shares of NHE | Original code divides each payer by sum of four selected categories | Replace denominator with full NHE (with 'other' category), or clearly relabel as shares among selected categories | Code defect verified; source workbook values pending |
| Figure 10 | 2025 forecast equals 2024 observed; ±7% labeled 'confidence band' | Direct code inspection; 2024 $5.2786T at 5.8% annual growth implies 2030 $7.403T | Correct year indexing; replace confidence-band label with explicit illustrative scenarios or statistically estimated interval | Code defect verified |
| Administrative waste interpretation | CMS administrative category discussed in context of inefficiency | CMS accounting category does not isolate unnecessary expenditure | Remove any inference that the entire category is waste or recoverable savings | Conceptual clarification |
| Reproducibility | Hardcoded `/mnt/user-data/uploads/` and underscore filenames | Repo uses `data/` and filenames with spaces | Keep original unchanged; use portable scripts and file inventory | Verified |
| Hospital and physician comparisons | Reported expenditure and growth comparisons | Table 02/07 workbook re-extraction required | Do not amend until source workbook values verified | Open |
| References and chronology | Some policy dates and bibliographic details potentially inconsistent | Requires primary-source check against article | Verify and list exact replacements | Open |

## Before writing to the journal

Do not claim that all published results are invalid or that all analyses have been reproduced. Provide the journal a consolidated, author-approved set of exact amendments once the workbook and article checks are complete. Retain a publicly accessible reproducibility audit with a dated changelog.
