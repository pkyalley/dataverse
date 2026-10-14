# HAEA dataset feasibility decision v1 — 2026-10-14

## Decision
**Primary empirical source:** CMS National Health Expenditure Accounts (NHEA), historical annual summary and service/payer tables. Official: https://www.cms.gov/data-research/statistics-trends-and-reports/national-health-expenditure-data/historical

**Purpose:** describe national spending composition and trends, not measure referral defects or prove avoidable waste.

**Official CMS confirmation:** 2024 national health spending was about $5.3T, $15,474 per person, 18.0% of GDP. Official CMS fact sheet: https://www.cms.gov/data-research/statistics-trends-and-reports/national-health-expenditure-data/nhe-fact-sheet

## Source verification and caveats
- CMS provides historical NHE tables, summary, and Definitions, Sources, and Methods documentation. Record exact archived download version, extraction date, and checksums before analysis.
- NHE government administration and non-medical insurance categories do **not** capture all administrative work, and neither category is synonymous with avoidable waste.
- Distinguish nominal from price-adjusted trends; don't reuse old paper's disputed results without fresh independent audit.
- Prior WJARR article audit remains paused; HAEA will not modify its files.
- AHRQ HCUPnet (https://datatools.ahrq.gov/hcupnet/) provides publicly accessible **aggregate** hospital utilization/readmission statistics; this is a secondary context source, not patient-level data or a referral-error dataset.
- CMS Provider Data Catalog clinician utilization (https://data.cms.gov/provider-data/dataset/n0yb-util) is procedure-volume data, **not** a direct measure of referral rework. Defer inclusion until research question warrants it.

## Measurement map
| Construct | Empirical source | Directly observable? | Interpretation |
|---|---|---|---|
| NHE total, GDP share, per capita | CMS NHEA | Yes | National spending context |
| CMS government administration / net cost of health insurance | CMS NHEA | Yes, by defined category | National accounting category, not measured waste |
| Hospital and physician service spending | CMS NHEA | Yes | Spending composition |
| Hospital stays/readmissions | AHRQ HCUPnet | Aggregate summaries | Utilization context, not referral defects |
| Missing referral documents, first-pass acceptance, rework minutes | None selected | **No** | Synthetic demonstration only |

## Data acquisition gate
Before any manuscript numerical results: independently download source; record publication date, exact file name and hash; validate columns, years, missingness and revisions; confirm source terms; run tests; then generate figures.
