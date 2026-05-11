<img width="2048" height="1280" alt="Preview" src="https://github.com/user-attachments/assets/dc84400f-8d88-4e86-9e5b-924080a59968" />

# SGD-NEER-shadow-model

Transparent shadow S$NEER proxy for studying Singapore dollar exchange-rate policy.

## What this model does

The script builds a trade-weighted geometric index from observable Yahoo Finance SGD crosses, where a higher index means broad SGD strength. It compares that proxy with an estimated MAS-style policy centre and band.

## Accuracy notes

MAS does not disclose the official S$NEER basket, weights, policy centre, slope, or band width. This project therefore avoids claiming to replicate the official index exactly. The basket weights and policy band are explicit modelling assumptions that can be adjusted in `SGD.py`.

The current version improves accuracy by:

- using SGD crosses in their natural quote direction, where higher values mean SGD appreciation;
- constructing a weighted geometric NEER proxy instead of regressing one bilateral FX pair against the others;
- removing the `scikit-learn` dependency and computing fit statistics locally;
- labelling the policy centre and band as estimated, not official;
- adding broader Asian trade-partner currencies such as TWD and IDR.

## Run

```bash
python3 SGD.py
```

The chart is saved as `sgd_neer_dashboard.png`.
