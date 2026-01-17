
# Input Structure Recognition (PCA Regime Analysis)

This repository explores **input-structure recognition** in market OHLCV data.
Instead of focusing on prediction, we analyze whether the input features form a
low-rank structure and how that structure changes over time (regime stability / transition).

## Key Ideas
- Raw price/volume series often hide structure.
- Feature engineering creates a representation space where structure becomes visible.
- PCA summarizes the global structure (dominant axes).
- Rolling PCA tracks time-varying structural stability.
- Subspace drift and anchor similarity measure whether the PCA subspace orientation is preserved or changes.

## What This Code Does
1) Fetch OHLCV from Upbit (e.g., KRW-BTC 60min)
2) Generate minimal features:
   - returns, range volatility, rolling volatility
   - volume log/change/z-score
   - momentum
3) Global PCA:
   - explained variance ratio
   - cumulative explained variance
   - participation ratio (effective dimension)
4) Rolling PCA:
   - PR (effective dimension over time)
   - cum2 / cum5 over time
5) Subspace drift:
   - detects sudden changes in the top-k PCA subspace orientation
6) Anchor similarity:
   - compares rolling subspace identity vs a fixed reference ("anchor") structure

## How to Run
```bash
pip install -r requirements.txt
python compare.py
