# Journal correction matrix — proposed, 14 October 2026

Article: Prince Peter Yalley, *The $372 Billion Problem*, WJARR 29(01), 2052–2064. DOI 10.30574/wjarr.2026.29.1.0020.

**Status:** All ten alternative figures and a summary table have been generated from archived CMS data; this matrix compares their findings with the article's printed sections and captions. It is a **proposed** correction schedule, not an approved corrigendum. Original published code and figures are preserved. See the detailed downloadable author-review matrix for replacement paragraphs.

| Printed location | Original claim | Verified finding / correction |
|---|---|---|
| Fig 1, §4.1, p2055 | NHE $27.1B 1960, $5.28T 2024; GDP 5.1%–18.0% | Retain; describe spending increase as nominal, not causal. |
| Fig 2, §4.2, pp2055–56 | Admin $38.3B 1990 to $372.1B 2024, +872% | Retain rounded values; **specify government administration and non-medical insurance**, not all administration or waste. |
| Fig 3, §4.3, p2056 | 2024 Medicare 26.8%, private 30.2%, Medicaid 22.7%, OOP 20.3%; 'Share of NHE'; 1987–2024 | Correct **total NHE** shares: Medicare **21.18%**, private **31.16%**, Medicaid **17.65%**, OOP **10.54%**, other **19.47%**. Archived detailed workbook has discrete historical years, not full annual 1987–2024; adjust timeline caption. |
| Fig 4, §4.4, p2057 | Hospital $1.51T, physician $0.98T, drugs $0.48T, admin $0.37T | Correct: hospital **$1.6347T**, physician/clinical **$1.1097T**, prescription drugs **$467.0B**, CMS admin category **$372.1B**. 'Fourth largest' only within plotted selected categories. |
| Fig 5, §4.5, pp2057–58 | Growth 1990–2024 admin 872%, hospital 385%, physician 419%, drugs 761% | Correct: **871.54%, 552.84%, 598.36%, 1058.81%**, respectively. Base-1990 indices: 971.54, 652.84, 698.36, 1158.81. Admin grew more slowly than drugs; change comparative conclusion. |
| Fig 6, §4.6, p2058 | NHE per capita $146 in 1960 to $15,474 in 2024 | Retain source-derived amounts; this is aggregate NHE per resident, not household out-of-pocket spending. |
| Fig 7, §4.7, pp2058–59 | Six-variable nominal correlation matrix, admin/NHE r=.998 | Summary annual 1980–2024 **r=.994673 (n=45)**; detailed Table 02 available-year **r=.990677 (n=27)**. Audited replacement matrix has **five** variables, not original six; restore sixth GDP variable if exact scope is required. Correlations are descriptive. |
| Fig 8, §4.8, pp2059–60 | r=.998 (p<.001) 1980–2024 | Replacement scatter uses detailed workbook's **27** available years: **r=.990677**. Alternatively annual summary has **45** years: **r=.994673**. State source, n, and non-causal limitation; do not carry over p-value without matching test. |
| Fig 9, §4.9, p2060 | Medicare $782B 26.8%; private $882B 30.2%; Medicaid $663B 22.7%; OOP $591B 20.3% | Correct: Medicare **$1118.0B (21.18%)**, private **$1644.6B (31.16%)**, Medicaid **$931.7B (17.65%)**, OOP **$556.6B (10.54%)**, other **$1027.7B (19.47%)**. Denominator $5278.6B NHE. |
| Fig 10, §4.10, pp2060–61 | CMS forecast $7.7T by 2030 at 5.8%, admin $540B; arbitrary ±7% 'confidence' bounds | Correctly indexed **illustrative** 5.8% compounding from 2024 $5.2786T gives **$7.4034T (2030)** and **$8.7678T (2033)**. At constant 7%, 2030 category would be **$518.2B**, not $540B. Not an official CMS forecast or confidence interval. |
| Table 1 hypotheses, p2054 | H1 872% vs hospital 385%; H5 r=.998; H4 payer claims | Update H1 hospital **552.84%**; H5 correlation with stated observation sample; reassess H4 using full-NHE denominator. |
| Table 3 key metrics, pp2061–62 | 1990 NHE about $724B and GDP share 12.3%; 2024 Medicare 26.8%, private 30.2% | Archived summary 1990 NHE **$718.7B**, GDP share **12.05%**. Use full-NHE payer shares above. Table 3 still needs cell-by-cell visual transcription. |
| Methods §§2–3, pp2054–55 | Analysis 'utilized 25 NHE Excel tables' | Original code directly uses summary and Tables 02, 03, 07; distinguish archived sources from tables actually analyzed. |
| Abstract/discussion/conclusion | r=.998 and administrative inefficiency interpretations | Update r after source choice; do not infer causal drivers or avoidable waste from nominal aggregate trends. |
| References/policy chronology | Potential inconsistent dates/DOIs | Verify against primary publications before proposing exact replacements. |

## Scope and evidence

Source files: `data/NHE24_Summary.csv` and archived CMS Table 02/03/07 Excel workbooks. Reproduction: `reproduce.py`, `reproduce_workbooks.py`, `results/summary_audit.json`, `results/workbook_reproduction.json`, `results/payer_2024.csv`, `results/category_comparison.csv`, `results/table1_summary_corrected.csv`, and `figures/corrected_fig1...corrected_fig10...`. The table named `table1_summary_corrected.csv` is a newly generated summary, not the paper's hypothesis Table 1.

**Remaining before a definitive journal submission:** Verify published Table 3 cell by cell; ensure Figure 7's sixth original variable and Figure 3's historical year coverage are reconciled; verify policy dates and references. Author approval and editor instructions are required. Do not represent this matrix as an already published correction.
