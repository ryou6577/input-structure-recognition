
# Input Structure Recognition (PCA Regime Analysis)

This repository explores **input-structure recognition** in market OHLCV data.
Instead of focusing on prediction, we analyze whether the input features form a
low-rank structure and how that structure changes over time (regime stability / transition).

## Perspective
This project treats market data as a **structure-generation process**, not merely a prediction target.  
By designing input features as a representation space, we attempt to reveal whether the market exhibits a **low-rank structure**, how stable that structure is over time, and when it **drifts or transitions** into a new regime.  
In this view, feature engineering is not an auxiliary step—it is the core mechanism that makes structure observable.


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

<img width="1029" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/34a3dc5a-bd92-4139-a8e8-e091a0b10569" />

<img width="893" height="480" alt="Figure_2" src="https://github.com/user-attachments/assets/3caafb35-6bad-41fc-ae75-2ccd840625d4" />

## How to Run
```bash
pip install -r requirements.txt
python compare.py

