# Forward Risk Manager

This repo starts with a data conversion pipeline from QuantConnect Research exports into tidy CSVs that are easy to feed into PyTorch/DGL.

## Data You Should Export
From QuantConnect Research, export daily history for:
- Prices for your symbol universe (S&P 400 constituents via MDY proxy).
- MDY ETF constituents (date + holdings/weights).
- Optional: Coarse Universe data for market cap.

## Expected Tidy Outputs
The converter writes:
- `data/processed/prices.csv`
  - Columns: `date,ticker,open,high,low,close,adj_close,volume`
- `data/processed/constituents.csv`
  - Columns: `date,ticker,is_member,weight,sector,market_cap`

## Converter Usage
Place your raw exports under `data/raw/` (any filenames). Then run:

```bash
python scripts/qc_export_to_tidy.py \
  --prices data/raw/prices \
  --constituents data/raw/constituents \
  --coarse data/raw/coarse \
  --out-dir data/processed
```

If your export columns have different names, override them, for example:

```bash
python scripts/qc_export_to_tidy.py \
  --prices data/raw/prices \
  --constituents data/raw/constituents \
  --price-date-col time \
  --price-ticker-col symbol \
  --adj-close-col adjusted_close \
  --constituent-ticker-col constituent_symbol \
  --weight-col weight
```

## Notes
- If you don’t have `adj_close`, the converter falls back to `close`.
- If constituents don’t include `is_member`, it defaults to 1.
- `market_cap` is added if coarse data is provided and the column can be inferred.
