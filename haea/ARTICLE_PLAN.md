# HAEA accompanying research article — development plan

**Working title:** Measuring Healthcare Administrative Burden: A Reproducible Public-Data Framework and Synthetic Workflow Analytics Demonstration

**Status:** Study design; not a completed empirical paper. Independent research, no RRH affiliation or data.

## Intended contribution
(1) A transparent taxonomy distinguishing aggregate spending from process-level administrative burden; (2) reproducible descriptive analysis of appropriately sourced public data; (3) an openly documented synthetic-data workflow quality demonstration; (4) an explicit account of what cannot be inferred.

## Draft manuscript structure
1. **Abstract** — write only after results are validated; distinguish empirical from synthetic.
2. **Introduction** — burden measurement problem, construct validity, research gap, scope.
3. **Background** — CMS administrative categories; process-level rework; limitations of public data.
4. **Research questions** — public-data descriptive questions and synthetic method performance.
5. **Data and provenance** — exact federal files, extraction/version dates, licensing, variable definitions; synthetic generator seed/assumptions.
6. **Methods** — denominators, time series, sensitivity checks; defect-rule evaluation against synthetic truth.
7. **Results A** — verified public-data findings only, with tables/figures from code.
8. **Results B** — explicitly simulated quality metrics and algorithm diagnostics.
9. **Discussion** — interpretation, boundaries, transportability, governance implications.
10. **Limitations** — no real-world process observations, no causal savings estimate, aggregate-data construct mismatch.
11. **Conclusion** — proportional to demonstrated findings.
12. **Reproducibility appendix** — dataset manifest, code, environment, tests.

## Publication gates
- Do not draft a numerical abstract before data validation.
- Do not describe CMS government administration/net insurance costs as total administrative waste.
- Do not treat synthetic results as hospital observations.
- No RRH patient/operational data, internal documents, or employer resources.
- Do not import disputed numerical claims from the earlier WJARR paper.
- Complete independent source and numerical review before submission.

## Manuscript and code synchronization
Each figure and table will be mapped to a script, input dataset version, definitions, and test. Keep claims register with citation and analysis reference. Decide whether the empirical study and synthetic demonstration should be separate manuscripts after feasibility review.
