# Measuring Healthcare Administrative Burden: A Reproducible Public-Data Framework and Synthetic Workflow Analytics Demonstration

**Manuscript status:** Working draft v0.1, 2026-10-14. Introduction and methods design only. No empirical or simulated results have been analyzed in this manuscript. Independent research; no employer affiliation or data.

## Abstract
*To be written after source acquisition, analysis, and validation.*

## 1. Introduction
Administrative activity is essential to the functioning of healthcare systems, yet the distinction between necessary coordination and avoidable process friction is often obscured in discussions of cost. Aggregate spending estimates can quantify expenditure categories but cannot, by themselves, reveal whether a particular referral was delayed by an incomplete document, how often an application was returned for correction, or how much staff effort could be prevented through better validation.

The United States National Health Expenditure Accounts (NHEA), maintained by the Centers for Medicare & Medicaid Services (CMS), provide a consistent national accounting framework for spending by type of service and source of funds. These statistics are valuable for examining long-run trends and composition, but their administrative categories should not be interpreted as a direct estimate of waste or as evidence of individual workflow failures (CMS, n.d.).

This study proposes a two-part analytical framework. First, it evaluates which dimensions of administrative burden can be described responsibly with publicly available national data. Second, it designs a transparent, synthetic referral-documentation experiment to test whether explicit quality rules can detect known, artificially introduced documentation defects. The latter is a methodological demonstration, not an observational study of actual healthcare operations.

The central contribution is therefore measurement discipline: a reproducible distinction among observed expenditure, hypothesized process burden, and simulated system performance.

## 2. Research questions
**RQ1.** Which administrative expenditure constructs are directly represented in CMS NHEA, and how do their measured trends depend on definitions, denominators, and inflation treatment?

**RQ2.** What important process-level constructs—such as first-pass referral acceptance, missing documentation, and rework time—cannot be inferred from aggregate spending accounts?

**RQ3.** In a fully synthetic referral dataset with known injected defects, how accurately do transparent validation rules detect missing or inconsistent fields under different assumed defect distributions?

## 3. Methods (planned)
### 3.1 Empirical data and provenance
Obtain official CMS NHEA historical files and definitions. Freeze versions with retrieval date and checksums. Construct year/category records and audit denominator consistency. Present nominal amounts, shares, and sensitivity analyses; inflation adjustment requires a documented deflator.

### 3.2 Construct mapping
For each proposed administrative burden indicator, record operational definition, data source, observation unit, numerator, denominator, validity limitations, and whether it is directly measured, proxied, or unavailable.

### 3.3 Synthetic experiment
Generate fictional referrals with seeded randomization and declared field-level defect mechanisms. No patient names, medical record numbers, real organizational process rules, or actual case records. Compare rule outputs with synthetic ground-truth defect labels and report confusion matrices, precision, recall, and sensitivity to scenario assumptions.

### 3.4 Analysis safeguards
Do not infer causality from national time series; do not call synthetic review minutes observed savings; label all simulations; provide runnable code and source manifests.

## 4. Results
*Reserved until validated analyses are complete.*

## 5. Discussion
*Reserved until results are available.*

## References (initial primary sources)
Centers for Medicare & Medicaid Services. (n.d.). *National Health Expenditure Data: Historical*. https://www.cms.gov/data-research/statistics-trends-and-reports/national-health-expenditure-data/historical

Agency for Healthcare Research and Quality. (n.d.). *HCUPnet data tools*. https://datatools.ahrq.gov/hcupnet/

*Publication dates and final APA details will be checked before submission.*
