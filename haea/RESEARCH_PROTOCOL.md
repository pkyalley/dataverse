# HAEA research protocol — v0.1 (2026-10-14)

## Working title
**Measuring Healthcare Administrative Burden: A Reproducible Public-Data Framework and Synthetic Workflow Analytics Demonstration**

## Motivation and boundaries
Administrative spending is not synonymous with administrative waste. CMS's category 'government administration and net cost of health insurance' is narrower than all administrative activities, and it cannot quantify avoidable referral/documentation rework. Public spending data can describe trends and composition but not identify the root causes of a local workflow defect.

## Study A: public-data measurement framework
**Aim A1:** Inventory definitions, observation units, reporting periods, revisions, and coverage of federal public datasets relevant to administrative costs and healthcare utilization.
**Aim A2:** Reproduce descriptive longitudinal trends and ratios with clearly documented denominators, nominal vs inflation-adjusted units, and sensitivity to source revision.
**Aim A3:** Assess whether available variables can meaningfully proxy administrative burden, distinguishing direct measurements from weak proxies and unavailable constructs.

**Candidate research questions**
- How do the CMS-defined administrative expenditure categories evolve as amounts and shares of NHE?
- How sensitive are reported trends to price adjustment, endpoints, and category definition?
- Which administrative process outcomes cannot be estimated from aggregate public expenditure data?

**Hypotheses:** Exploratory/descriptive until a dataset-specific preregistration defines estimands and testing rules. Do not predeclare a causal effect.

**Unit of analysis:** Year/category for CMS NHE; other units only after separate data dictionary review.
**Initial methods:** source/version logging; descriptive statistics; real vs nominal trends; denominator audits; transparent plots; optional uncertainty only when statistically justified.
**Quality controls:** assertions on year coverage, duplicate keys, missingness, arithmetic reconciliation, source revisions, and reproducibility.

## Study B: synthetic workflow methods demonstration
**Aim B1:** Define an explicit referral-documentation data dictionary without using or reconstructing any employer workflow.
**Aim B2:** Simulate independent fictional referrals with transparent data-generating assumptions and seeds.
**Aim B3:** Build rule-based quality checks and compute illustrative metrics (missing fields, validation failures, rework counts, cycle times).
**Aim B4:** Evaluate algorithm behavior against synthetic ground truth (precision, recall, false-positive rate), and run sensitivity analyses.

**Important:** Synthetic records demonstrate feasibility of methods, not real-world prevalence, clinical safety, time savings, or external validity.

## Proposed data model for synthetic demonstration
- referral_id (fictional random identifier)
- created_at, first_review_at, completed_at (synthetic timestamps)
- document_type (synthetic controlled vocabulary)
- required_document_present (boolean)
- date_valid (boolean)
- referral_status (synthetic state)
- defect_type (synthetic classification)
- rework_events (nonnegative integer)
- reviewer_minutes (synthetic numeric value)
No name, DOB, MRN, real location, staff identifier, or employer-derived fields.

## Outcome definitions (to validate)
- Documentation completeness = referrals passing all defined required-field checks / eligible referrals.
- First-pass acceptance = referrals accepted without correction / reviewed referrals.
- Rework incidence = referrals with >=1 correction / reviewed referrals.
- Median cycle time = median(completed_at - created_at) for completed synthetic referrals, with censoring handled separately.
- Review effort = total simulated reviewer minutes; **not observed cost savings**.

## Publication standards
Provide a complete source manifest, reproducible code, analysis environment, synthetic-data provenance, and limitation statement. Seek methodological review before final hypotheses or submission.

## Pending decisions
Identify an actually downloadable, suitable public dataset with explicit version and dictionary; decide whether Study A and Study B merit separate manuscripts. No data collection has started.
