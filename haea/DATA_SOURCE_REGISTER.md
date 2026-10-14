# Candidate data source register — feasibility stage

| Source | Official link | Potential role | Limitations / checks |
|---|---|---|---|
| CMS National Health Expenditure Accounts | https://www.cms.gov/data-research/statistics-trends-and-reports/national-health-expenditure-data | Spending trends by category and payer | Aggregate expenditures; administrative category is not a measure of avoidable waste; confirm current historical table version and definitions |
| CMS open data catalog | https://data.cms.gov/ | Provider/payment/utilization context | Select a specific dataset, validate dictionary, year and suppression rules; cannot assume it measures referral rework |
| CMS provider service summaries | https://data.cms.gov/provider-summary-by-type-of-service | Descriptive utilization/payment patterns | Primarily Medicare FFS; not a direct administrative effort measure |
| AHRQ HCUP | https://www.ahrq.gov/data/hcup/index.html | Hospital utilization and quality context | Some encounter-level products require purchase, agreements, or restricted use; do not assume freely downloadable record-level data |
| AHRQ HCUPnet | https://datatools.ahrq.gov/hcupnet/ | Aggregate published hospital statistics | Follow data use terms and small-cell protections; not referral defect data |

**Acquisition gate:** No dataset is accepted until exact file/API endpoint, version/date, license/terms, unit of observation, data dictionary, geographic/coverage restrictions, suppression, and citation are recorded. Never upload restricted raw data to public GitHub.
