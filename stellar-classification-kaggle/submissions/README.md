# Submission Registry

This folder keeps the central audit trail for competition submissions.

- `submission_registry.csv`: compact table for sorting and comparison.
- `submission_registry.jsonl`: full metadata records, one submission per line.
- `*.metadata.json` files live next to each generated submission in `outputs/`.

Submission CSV files remain ignored by git. Register existing outputs with:

```powershell
python scripts/08_register_submissions.py --write
```

