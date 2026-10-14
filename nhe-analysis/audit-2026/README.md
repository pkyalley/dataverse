# NHE 2026 audit (preliminary)

This audit supplements the published paper [DOI 10.30574/wjarr.2026.29.1.0020](https://doi.org/10.30574/wjarr.2026.29.1.0020). **It is not a journal-approved correction.** The original code, notebook, figures and source data are preserved unchanged.

Run from the repository root: `python nhe-analysis/audit-2026/reproduce.py`. Uses only Python standard library and the archived `data/NHE24_Summary.csv`; outputs are written under `audit-2026/results/`.

### Verified from archived summary data
- 2024 NHE: $5,278.6B; government administration and non-medical insurance: $372.1B (7.0492% of NHE).
- 1990 category: $38.3B; nominal growth to 2024: 871.54%.
- Nominal levels correlation 1960–2024: r=0.994984; 1980–2024: r=0.994673; annual log-growth correlation 1961–2024: r=0.439827.
- Illustrative 5.8% annual growth from 2024 implies 2030 $7,403.43B; 2033 $8,767.78B. **Not a CMS forecast.**

### Issues requiring correction or further validation
- Original Figures 3/9 normalize four selected payer categories by their own sum, not all NHE.
- Figure 10 starts 2025 at 2024 actual; labels arbitrary ±7% bounds as a confidence band.
- Article r=0.998 differs from archived-data reproduction; verify original vintage and methodology.
- Correlation of trending nominal series does not establish causation or waste; CMS administration category is not total waste.
- Original code hardcodes environment paths and Excel filenames different from checked-in filenames.
- Workbook-based Table 02/03/07 values and all published figure labels still require full independent verification.

See CHANGELOG.md and JOURNAL_CORRECTION_STEPS.md.
