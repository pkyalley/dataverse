# Workbook inspection — next reproduction stage

The checked-in CMS Table 02, 03 and 07 files are Excel workbooks. Their raw bytes are accessible from GitHub, but a verified independent local workbook execution has not yet been completed in this audit.

```bash
python -m pip install openpyxl
python nhe-analysis/audit-2026/inspect_workbooks.py
```

The script produces `results/workbook_inventory.json`, including SHA-256 hashes, worksheet names, and candidate year-header rows. It **does not** claim to have corrected published workbook-derived values. A human or subsequent validated script must check the units, labels, selected years, and formulas before replacing any figures.

See `CORRECTION_MATRIX.md` for issue-by-issue status.
